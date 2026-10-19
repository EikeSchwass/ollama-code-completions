using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace OllamaCodeCompletions
{
    /// <summary>
    /// Minimal fill-in-the-middle client for two server backends:
    /// <list type="bullet">
    /// <item>Ollama: <c>/api/generate</c> with its native <c>suffix</c> parameter, models via <c>/api/tags</c>.</item>
    /// <item>llama.cpp <c>llama-server</c>: <c>/infill</c> with <c>input_prefix</c>/<c>input_suffix</c>,
    /// models via <c>/v1/models</c>.</item>
    /// </list>
    /// Both apply the right FIM tokens for whichever code model is loaded (qwen2.5-coder,
    /// codellama, codegemma, deepseek-coder, …) so we don't have to hard-code them.
    /// </summary>
    internal sealed class OllamaClient
    {
        private static readonly HttpClient s_http = new HttpClient
        {
            // Per-request timeouts are still enforced via the CancellationToken; this is just a
            // hard upper bound to prevent stuck connections from leaking forever.
            Timeout = TimeSpan.FromMinutes(2),
        };

        public sealed class CompletionRequest
        {
            public ServerBackend Backend { get; set; }
            public string ServerUrl { get; set; }
            public string Model { get; set; }
            public string Prefix { get; set; }
            public string Suffix { get; set; }
            public int MaxPredict { get; set; } = 128;
            public int TimeoutSeconds { get; set; } = 30;
            public bool UseAuth { get; set; }
            public string Username { get; set; }
            public string Password { get; set; }
            public IReadOnlyList<string> StopTokens { get; set; }
        }

        /// <summary>
        /// Returns the suggested completion text, or null if the request failed,
        /// was cancelled, or the server returned an empty response. Never throws.
        /// </summary>
        public async Task<string> GetCompletionAsync(CompletionRequest req, CancellationToken ct)
        {
            if (req == null) return null;
            if (string.IsNullOrWhiteSpace(req.ServerUrl)) return null;
            // A single-model llama-server ignores "model"; Ollama requires it.
            if (req.Backend == ServerBackend.Ollama && string.IsNullOrWhiteSpace(req.Model)) return null;

            string url = req.ServerUrl.TrimEnd('/') + (req.Backend == ServerBackend.LlamaServer ? "/infill" : "/api/generate");

            // Build the stop-token array: always include the standard FIM/EOT sentinels,
            // then append any caller-supplied tokens (e.g. "\n" for single-line mode).
            var stopArray = new JArray
            {
                "<|endoftext|>", "<|fim_pad|>", "<|file_sep|>", "<EOT>",
                "<|im_end|>", "<|im_start|>"
            };
            if (req.StopTokens != null)
            {
                foreach (string token in req.StopTokens)
                    stopArray.Add(token);
            }

            JObject payload = req.Backend == ServerBackend.LlamaServer
                ? BuildInfillPayload(req, stopArray)
                : BuildGeneratePayload(req, stopArray);

            // Per-request linked timeout so a hung server doesn't pin a suggestion forever.
            using (var timeoutCts = new CancellationTokenSource(TimeSpan.FromSeconds(Math.Max(1, req.TimeoutSeconds))))
            using (var linked = CancellationTokenSource.CreateLinkedTokenSource(ct, timeoutCts.Token))
            using (var http = new HttpRequestMessage(HttpMethod.Post, url))
            {
                http.Content = new StringContent(payload.ToString(Formatting.None), Encoding.UTF8, "application/json");

                if (req.UseAuth)
                {
                    string raw = (req.Username ?? string.Empty) + ":" + (req.Password ?? string.Empty);
                    string b64 = Convert.ToBase64String(Encoding.UTF8.GetBytes(raw));
                    http.Headers.Authorization = new AuthenticationHeaderValue("Basic", b64);
                }

                Logger.Log("Http", $"POST {url} backend={req.Backend} model={req.Model} predict={req.MaxPredict} timeout={req.TimeoutSeconds}s");
                var sw = Stopwatch.StartNew();
                try
                {
                    using (var resp = await s_http.SendAsync(http, HttpCompletionOption.ResponseContentRead, linked.Token).ConfigureAwait(false))
                    {
                        sw.Stop();
                        Logger.Log("Http", $"{(int)resp.StatusCode} in {sw.ElapsedMilliseconds}ms");

                        if (!resp.IsSuccessStatusCode) return null;

                        string body = await resp.Content.ReadAsStringAsync().ConfigureAwait(false);
                        var parsed = JObject.Parse(body);
                        string field = req.Backend == ServerBackend.LlamaServer ? "content" : "response";
                        string completion = (string)parsed[field];
                        if (completion == null)
                        {
                            Logger.Log("Http", $"{field} field missing or non-string");
                            completion = string.Empty;
                        }
                        return PostProcess(completion);
                    }
                }
                catch (OperationCanceledException) when (ct.IsCancellationRequested)
                {
                    // User kept typing — propagate so the session can stay quiet.
                    throw;
                }
                catch (OperationCanceledException)
                {
                    // Linked token cancelled by timeoutCts — request exceeded TimeoutSeconds.
                    Logger.Log("Http", $"timeout after {req.TimeoutSeconds}s");
                    return null;
                }
                catch (HttpRequestException ex)
                {
                    Logger.LogException("Http", ex);
                    return null;
                }
                catch (JsonReaderException ex)
                {
                    Logger.LogException("Http", ex);
                    return null;
                }
            }
        }

        private static JObject BuildGeneratePayload(CompletionRequest req, JArray stopArray)
        {
            return new JObject
            {
                ["model"] = req.Model,
                ["prompt"] = req.Prefix ?? string.Empty,
                ["suffix"] = req.Suffix ?? string.Empty,
                ["stream"] = false,
                ["options"] = new JObject
                {
                    ["temperature"] = 0.2,
                    ["top_p"] = 0.95,
                    ["num_predict"] = req.MaxPredict,
                    ["stop"] = stopArray
                }
            };
        }

        private static JObject BuildInfillPayload(CompletionRequest req, JArray stopArray)
        {
            var payload = new JObject
            {
                ["input_prefix"] = req.Prefix ?? string.Empty,
                ["input_suffix"] = req.Suffix ?? string.Empty,
                ["stream"] = false,
                ["cache_prompt"] = true,
                ["temperature"] = 0.2,
                ["top_p"] = 0.95,
                ["n_predict"] = req.MaxPredict,
                ["stop"] = stopArray
            };
            // A single-model llama-server ignores "model"; only router-mode servers use it.
            if (!string.IsNullOrWhiteSpace(req.Model))
                payload["model"] = req.Model;
            return payload;
        }

        // ---- Server discovery + diagnostics (used by the Options page UI) ----

        /// <summary>
        /// Returns the alphabetically-sorted list of model names available on the
        /// configured server (GET /api/tags or /v1/models). Throws on any failure — the
        /// caller is expected to surface the message to the user.
        /// </summary>
        public static async Task<List<string>> ListModelsAsync(
            ServerBackend backend, string serverUrl, bool useAuth, string username, string password, CancellationToken ct)
        {
            if (string.IsNullOrWhiteSpace(serverUrl))
                throw new ArgumentException("Server URL is required.", nameof(serverUrl));

            string url = serverUrl.TrimEnd('/') + ModelsPath(backend);

            using (var http = new HttpRequestMessage(HttpMethod.Get, url))
            {
                ApplyBasicAuth(http, useAuth, username, password);
                Logger.Log("Http", $"GET {url}");
                var sw = Stopwatch.StartNew();
                using (var resp = await s_http.SendAsync(http, HttpCompletionOption.ResponseContentRead, ct).ConfigureAwait(false))
                {
                    sw.Stop();
                    Logger.Log("Http", $"{(int)resp.StatusCode} in {sw.ElapsedMilliseconds}ms");

                    if (!resp.IsSuccessStatusCode)
                        throw new HttpRequestException($"HTTP {(int)resp.StatusCode} {resp.ReasonPhrase}");

                    string body = await resp.Content.ReadAsStringAsync().ConfigureAwait(false);
                    var result = ParseModelNames(backend, JObject.Parse(body));
                    if (result == null)
                        throw new HttpRequestException($"Unexpected response shape (no '{ModelsField(backend)}' field).");
                    result.Sort(StringComparer.OrdinalIgnoreCase);
                    return result;
                }
            }
        }

        /// <summary>Result of <see cref="TestConnectionAsync"/>.</summary>
        public sealed class TestConnectionResult
        {
            public bool Success { get; }
            public string Message { get; }
            public TestConnectionResult(bool success, string message)
            {
                Success = success;
                Message = message ?? string.Empty;
            }
        }

        /// <summary>
        /// GETs the model list (/api/tags or /v1/models) to verify reachability + auth,
        /// then checks the configured model is on the server. A single-model llama-server
        /// ignores the configured model, so there a mismatch is only reported as a
        /// failure when several models are served. Expected failures (HTTP, JSON, auth) come back as
        /// <see cref="TestConnectionResult"/> with <see cref="TestConnectionResult.Success"/>=false;
        /// truly unexpected exceptions propagate to the caller.
        /// </summary>
        public static async Task<TestConnectionResult> TestConnectionAsync(
            ServerBackend backend, string serverUrl, string model, bool useAuth, string username, string password, CancellationToken ct)
        {
            if (string.IsNullOrWhiteSpace(serverUrl))
                return new TestConnectionResult(false, "Server URL is empty.");

            string url = serverUrl.TrimEnd('/') + ModelsPath(backend);

            using (var http = new HttpRequestMessage(HttpMethod.Get, url))
            {
                ApplyBasicAuth(http, useAuth, username, password);
                Logger.Log("Http", $"GET {url} (test)");
                try
                {
                    using (var resp = await s_http.SendAsync(http, HttpCompletionOption.ResponseContentRead, ct).ConfigureAwait(false))
                    {
                        Logger.Log("Http", $"{(int)resp.StatusCode} (test)");
                        if (resp.StatusCode == System.Net.HttpStatusCode.Unauthorized)
                            return new TestConnectionResult(false, "Authentication failed (HTTP 401).");
                        if (!resp.IsSuccessStatusCode)
                            return new TestConnectionResult(false, $"HTTP {(int)resp.StatusCode} {resp.ReasonPhrase}");

                        string body = await resp.Content.ReadAsStringAsync().ConfigureAwait(false);
                        var models = ParseModelNames(backend, JObject.Parse(body));
                        if (models == null)
                            return new TestConnectionResult(false, $"Unexpected response shape (no '{ModelsField(backend)}' field).");

                        if (string.IsNullOrWhiteSpace(model))
                            return new TestConnectionResult(true, $"{models.Count} model(s) available.");

                        foreach (string name in models)
                        {
                            if (string.Equals(name, model, StringComparison.OrdinalIgnoreCase))
                                return new TestConnectionResult(true, $"Model '{model}' is available.");
                        }
                        if (backend == ServerBackend.LlamaServer && models.Count == 1)
                            return new TestConnectionResult(true, $"Server serves '{models[0]}' (configured model is ignored).");
                        return new TestConnectionResult(false, $"Connected, but model '{model}' is not on the server.");
                    }
                }
                catch (HttpRequestException ex)
                {
                    Logger.LogException("Http", ex);
                    return new TestConnectionResult(false, ex.Message);
                }
                catch (JsonReaderException ex)
                {
                    Logger.LogException("Http", ex);
                    return new TestConnectionResult(false, $"Bad response: {ex.Message}");
                }
            }
        }

        private static string ModelsPath(ServerBackend backend) =>
            backend == ServerBackend.LlamaServer ? "/v1/models" : "/api/tags";

        private static string ModelsField(ServerBackend backend) =>
            backend == ServerBackend.LlamaServer ? "data" : "models";

        /// <summary>
        /// Extracts model names from <c>models[].name</c> (Ollama) or <c>data[].id</c>
        /// (llama-server, OpenAI-style), or null if the list field is absent.
        /// </summary>
        private static List<string> ParseModelNames(ServerBackend backend, JObject parsed)
        {
            var list = parsed[ModelsField(backend)] as JArray;
            if (list == null) return null;
            string nameField = backend == ServerBackend.LlamaServer ? "id" : "name";
            var result = new List<string>();
            foreach (var m in list)
            {
                string name = (string)m[nameField];
                if (!string.IsNullOrEmpty(name)) result.Add(name);
            }
            return result;
        }

        /// <summary>Adds an <c>Authorization: Basic …</c> header when auth is enabled.</summary>
        private static void ApplyBasicAuth(HttpRequestMessage http, bool useAuth, string username, string password)
        {
            if (!useAuth) return;
            string raw = (username ?? string.Empty) + ":" + (password ?? string.Empty);
            string b64 = Convert.ToBase64String(Encoding.UTF8.GetBytes(raw));
            http.Headers.Authorization = new AuthenticationHeaderValue("Basic", b64);
        }

        /// <summary>Strip stray FIM/EOT sentinels that occasionally leak through.</summary>
        private static string PostProcess(string s)
        {
            if (string.IsNullOrEmpty(s)) return s;
            string[] junk =
            {
                "<|endoftext|>", "<|fim_pad|>", "<|file_sep|>", "<EOT>",
                "<|fim_middle|>", "<|fim_prefix|>", "<|fim_suffix|>",
                "<|im_end|>", "<|im_start|>",
            };
            foreach (var j in junk)
            {
                int idx = s.IndexOf(j, StringComparison.Ordinal);
                if (idx >= 0) s = s.Substring(0, idx);
            }
            return s;
        }
    }
}
