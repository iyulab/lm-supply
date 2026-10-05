using System.Diagnostics;
using System.Net.Http.Headers;
using System.Net.Http.Json;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.Json.Serialization;
using LMSupply.Exceptions;
using LMSupply.Json;

namespace LMSupply.Llama.Server;

/// <summary>
/// OpenAI-compatible HTTP client for llama-server.
/// </summary>
public sealed class LlamaServerClient : IDisposable
{
    private readonly HttpClient _httpClient;
    private readonly string _baseUrl;
    private readonly bool _ownsHttpClient;
    private readonly int _maxContextLength;
    private readonly string? _apiKey;
    private readonly TimeSpan? _requestTimeout;
    private Func<ServerState>? _serverState;

    /// <summary>The wire options every request and response uses (snake_case, nulls omitted, source-generated metadata).</summary>
    internal static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
        TypeInfoResolver = LlamaJsonContext.Default
    };

    /// <summary>
    /// Creates a new client for the specified server URL.
    /// </summary>
    /// <param name="baseUrl">llama-server base URL.</param>
    /// <param name="httpClient">Optional shared HttpClient. If null, a new one is created and owned.
    /// <paramref name="requestTimeout"/> only applies to the client created in that case — a caller
    /// supplying their own <paramref name="httpClient"/> owns its timeout too.</param>
    /// <param name="maxContextLength">Model context window size — carried in ContextLengthExceededException on overflow.</param>
    /// <param name="requestTimeout">Limit on each request this client sends when it owns its
    /// HttpClient. Defaults to 5 minutes, not .NET's 100-second default — local/CPU-bound inference,
    /// especially a multi-step tool-calling loop where each round's prompt grows with prior tool
    /// results, routinely exceeds 100 seconds per completion on hardware without a GPU.
    /// <see cref="Timeout.InfiniteTimeSpan"/> means no limit. It covers the same span
    /// <see cref="HttpClient.Timeout"/> would — up to the response headers for streamed calls, the
    /// whole body otherwise — and expiry throws the same shape (<see cref="TaskCanceledException"/>
    /// with an inner <see cref="TimeoutException"/>). Ignored when <paramref name="httpClient"/> is
    /// supplied.</param>
    /// <param name="apiKey">Key the server was started with (<see cref="LlamaServerProcess.ApiKey"/>); sent as
    /// <c>Authorization: Bearer</c> on every request. <see langword="null"/> sends none.</param>
    public LlamaServerClient(
        string baseUrl,
        HttpClient? httpClient = null,
        int maxContextLength = 0,
        TimeSpan? requestTimeout = null,
        string? apiKey = null)
    {
        _baseUrl = baseUrl.TrimEnd('/');
        _maxContextLength = maxContextLength;
        _apiKey = string.IsNullOrEmpty(apiKey) ? null : apiKey;

        if (httpClient != null)
        {
            _httpClient = httpClient;
            _ownsHttpClient = false;
        }
        else
        {
            // The limit is applied per request (SendWithTimeoutAsync), not on the HttpClient, so that
            // views over this client (WithRequestTimeout) can carry their own limit on the same
            // connection pool.
            _httpClient = new HttpClient { Timeout = Timeout.InfiniteTimeSpan };
            _ownsHttpClient = true;
            _requestTimeout = ValidateRequestTimeout(requestTimeout ?? LlamaServerConfig.DefaultRequestTimeout);
        }
    }

    private LlamaServerClient(LlamaServerClient owner, TimeSpan requestTimeout)
    {
        _baseUrl = owner._baseUrl;
        _maxContextLength = owner._maxContextLength;
        _apiKey = owner._apiKey;
        _httpClient = owner._httpClient;
        _ownsHttpClient = false;
        _requestTimeout = ValidateRequestTimeout(requestTimeout);
        _serverState = owner._serverState;
    }

    /// <summary>
    /// A client for the same server that shares this client's connections but limits each request
    /// to <paramref name="requestTimeout"/>. Disposing the view does not dispose the connections.
    /// </summary>
    /// <remarks>
    /// A pooled server is shared by every caller that loads the same model; each caller's lease
    /// gets its own view, so one caller's limit never becomes another's.
    /// </remarks>
    internal LlamaServerClient WithRequestTimeout(TimeSpan requestTimeout) => new(this, requestTimeout);

    /// <summary>The per-request limit this client applies; <see langword="null"/> when the caller supplied the HttpClient (its own timeout governs).</summary>
    internal TimeSpan? RequestTimeout => _requestTimeout;

    private static TimeSpan ValidateRequestTimeout(TimeSpan value) =>
        value == Timeout.InfiniteTimeSpan || value > TimeSpan.Zero
            ? value
            : throw new ArgumentOutOfRangeException(nameof(value), value,
                "Request timeout must be positive or Timeout.InfiniteTimeSpan.");

    private Task<HttpResponseMessage> PostWithTimeoutAsync(string url, HttpContent content, CancellationToken cancellationToken) =>
        SendWithTimeoutAsync(
            new HttpRequestMessage(HttpMethod.Post, url) { Content = content },
            HttpCompletionOption.ResponseContentRead,
            cancellationToken);

    /// <summary>
    /// Sends with this client's request limit, over exactly the span <see cref="HttpClient.Timeout"/>
    /// covers, and reports expiry the way HttpClient does.
    /// </summary>
    private async Task<HttpResponseMessage> SendWithTimeoutAsync(
        HttpRequestMessage request,
        HttpCompletionOption completionOption,
        CancellationToken cancellationToken)
    {
        // Per request, not DefaultRequestHeaders: a caller-supplied HttpClient is not ours to change.
        if (_apiKey is not null)
            request.Headers.Authorization = new AuthenticationHeaderValue("Bearer", _apiKey);

        if (_requestTimeout is not { } limit || limit == Timeout.InfiniteTimeSpan)
            return await _httpClient.SendAsync(request, completionOption, cancellationToken);

        using var cts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        cts.CancelAfter(limit);
        try
        {
            return await _httpClient.SendAsync(request, completionOption, cts.Token);
        }
        catch (OperationCanceledException ex) when (!cancellationToken.IsCancellationRequested && cts.IsCancellationRequested)
        {
            var message = $"The request to llama-server was canceled because the request timeout of {limit.TotalSeconds:0.###} seconds elapsed.";
            throw new TaskCanceledException(message, new TimeoutException(message, ex));
        }
    }

    /// <summary>
    /// Generates a streaming chat completion.
    /// </summary>
    private async IAsyncEnumerable<string> GenerateChatCoreAsync(
        IEnumerable<ChatCompletionMessage> messages,
        ChatCompletionOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        options ??= new ChatCompletionOptions();

        var request = BuildChatRequest(messages, options, stream: true);

        var json = JsonSerializer.Serialize(request, JsonOptions.TypeInfoOf(request));
        using var content = new StringContent(json, Encoding.UTF8, "application/json");

        using var httpRequest = new HttpRequestMessage(HttpMethod.Post, $"{_baseUrl}/v1/chat/completions")
        {
            Content = content
        };

        using var response = await SendWithTimeoutAsync(
            httpRequest,
            HttpCompletionOption.ResponseHeadersRead,
            cancellationToken);

        await EnsureSuccessOrThrowContextExceptionAsync(response, _maxContextLength, cancellationToken);

        await using var stream = await response.Content.ReadAsStreamAsync(cancellationToken);
        using var reader = new StreamReader(stream);

        while (await reader.ReadLineAsync(cancellationToken) is { } line)
        {
            if (string.IsNullOrEmpty(line))
                continue;

            if (!line.StartsWith("data: ", StringComparison.Ordinal))
                continue;

            var data = line[6..];

            if (data == "[DONE]")
                break;

            ChatCompletionChunk? chunk;
            try
            {
                chunk = JsonSerializer.Deserialize(data, JsonOptions.TypeInfo<ChatCompletionChunk>());
            }
            catch (Exception ex)
            {
                Trace.TraceInformation($"[LlamaServerClient] Chat SSE chunk deserialization failed: {ex.Message}");
                continue;
            }

            // Only yield content tokens — reasoning_content (thinking) is intentionally skipped.
            // b8994+: Gemma 4 separates thinking into reasoning_content; content holds the answer.
            var delta = chunk?.Choices?.FirstOrDefault()?.Delta?.Content;
            if (!string.IsNullOrEmpty(delta))
            {
                yield return delta;
            }
        }
    }

    /// <summary>
    /// Generates a streaming chat completion with structured data.
    /// Returns text deltas, tool call deltas, and finish reason.
    /// </summary>
    private async IAsyncEnumerable<ChatStreamData> GenerateChatStreamCoreAsync(
        IEnumerable<ChatCompletionMessage> messages,
        ChatCompletionOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        options ??= new ChatCompletionOptions();

        var request = BuildChatRequest(messages, options, stream: true);

        var json = JsonSerializer.Serialize(request, JsonOptions.TypeInfoOf(request));
        using var content = new StringContent(json, Encoding.UTF8, "application/json");

        using var httpRequest = new HttpRequestMessage(HttpMethod.Post, $"{_baseUrl}/v1/chat/completions")
        {
            Content = content
        };

        using var response = await SendWithTimeoutAsync(
            httpRequest,
            HttpCompletionOption.ResponseHeadersRead,
            cancellationToken);

        await EnsureSuccessOrThrowContextExceptionAsync(response, _maxContextLength, cancellationToken);

        await using var stream = await response.Content.ReadAsStreamAsync(cancellationToken);
        using var reader = new StreamReader(stream);

        while (await reader.ReadLineAsync(cancellationToken) is { } line)
        {
            if (string.IsNullOrEmpty(line))
                continue;

            if (!line.StartsWith("data: ", StringComparison.Ordinal))
                continue;

            var data = line[6..];

            if (data == "[DONE]")
                break;

            ChatCompletionChunk? chunk;
            try
            {
                chunk = JsonSerializer.Deserialize(data, JsonOptions.TypeInfo<ChatCompletionChunk>());
            }
            catch (Exception ex)
            {
                Trace.TraceInformation($"[LlamaServerClient] Chat stream chunk deserialization failed: {ex.Message}");
                continue;
            }

            // usage and timings share the last chunk on current builds (b11146: the include_usage chunk with empty
            // choices), but neither is assumed to arrive with the other.
            var usage = ReportedUsage(chunk);
            if (usage is not null || chunk?.Timings is not null)
                yield return new ChatStreamData { Usage = usage, Timings = chunk?.Timings };

            var choice = chunk?.Choices?.FirstOrDefault();
            if (choice is null)
                continue;

            if (choice.Delta?.Content is not null ||
                choice.Delta?.ReasoningContent is not null ||
                choice.Delta?.ToolCalls is { Count: > 0 } ||
                choice.FinishReason is not null)
            {
                yield return new ChatStreamData
                {
                    TextDelta = choice.Delta?.Content,
                    ReasoningDelta = choice.Delta?.ReasoningContent,
                    ToolCallDeltas = choice.Delta?.ToolCalls,
                    FinishReason = choice.FinishReason
                };
            }
        }
    }

    /// <summary>
    /// The server's token accounting from a streamed chunk: the OpenAI-compatible <c>usage</c> object when present,
    /// else llama.cpp's <c>timings</c>. Null when the chunk carries neither.
    /// </summary>
    private static ChatCompletionUsage? ReportedUsage(ChatCompletionChunk? chunk)
    {
        if (chunk?.Usage is { } usage)
            return usage;
        if (chunk?.Timings is { PromptN: { } prompt, PredictedN: { } predicted })
            return new ChatCompletionUsage { PromptTokens = prompt, CompletionTokens = predicted, TotalTokens = prompt + predicted };
        return null;
    }

    /// <summary>
    /// Generates a non-streaming chat completion.
    /// </summary>
    public async Task<string> GenerateChatCompleteAsync(
        IEnumerable<ChatCompletionMessage> messages,
        ChatCompletionOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        var sb = new StringBuilder();

        await foreach (var token in GenerateChatAsync(messages, options, cancellationToken))
        {
            sb.Append(token);
        }

        return sb.ToString();
    }

    /// <summary>
    /// Generates a non-streaming chat completion with full response including tool calls.
    /// </summary>
    private async Task<ChatCompletionFullResponse> GenerateChatWithToolsCoreAsync(
        IEnumerable<ChatCompletionMessage> messages,
        ChatCompletionOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        options ??= new ChatCompletionOptions();

        var request = BuildChatRequest(messages, options, stream: false);

        var json = JsonSerializer.Serialize(request, JsonOptions.TypeInfoOf(request));
        using var content = new StringContent(json, Encoding.UTF8, "application/json");

        using var response = await PostWithTimeoutAsync(
            $"{_baseUrl}/v1/chat/completions",
            content,
            cancellationToken);

        await EnsureSuccessOrThrowContextExceptionAsync(response, _maxContextLength, cancellationToken);

        var responseJson = await response.Content.ReadAsStringAsync(cancellationToken);
        var result = JsonSerializer.Deserialize(responseJson, JsonOptions.TypeInfo<ChatCompletionFullResponse>());

        return result ?? new ChatCompletionFullResponse();
    }

    /// <summary>
    /// Generates a streaming text completion.
    /// </summary>
    public async IAsyncEnumerable<string> GenerateAsync(
        string prompt,
        CompletionOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        await foreach (var data in GenerateStreamAsync(prompt, options, cancellationToken).ConfigureAwait(false))
        {
            if (!string.IsNullOrEmpty(data.TextDelta))
                yield return data.TextDelta;
        }
    }

    /// <summary>
    /// Generates a streaming text completion with the reason it ended: every chunk carries a text delta, and
    /// the last one carries <see cref="CompletionStreamData.FinishReason"/> — <c>"length"</c> when the server
    /// stopped at <c>n_predict</c>, <c>"stop"</c> at the end-of-sequence token or a stop word.
    /// </summary>
    private async IAsyncEnumerable<CompletionStreamData> GenerateStreamCoreAsync(
        string prompt,
        CompletionOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        options ??= new CompletionOptions();

        var request = new CompletionRequest
        {
            Prompt = prompt,
            NPredict = options.MaxTokens,
            Temperature = options.Temperature,
            TopP = options.TopP,
            TopK = options.TopK > 0 ? options.TopK : null,
            MinP = options.MinP > 0 ? options.MinP : null,
            RepeatPenalty = options.RepeatPenalty != 1.0f ? options.RepeatPenalty : null,
            FrequencyPenalty = options.FrequencyPenalty != 0 ? options.FrequencyPenalty : null,
            PresencePenalty = options.PresencePenalty != 0 ? options.PresencePenalty : null,
            DryMultiplier = options.DryMultiplier,
            DryBase = options.DryBase,
            DryAllowedLength = options.DryAllowedLength,
            DryPenaltyLastN = options.DryPenaltyLastN,
            RepeatLastN = options.RepeatLastN,
            Seed = options.Seed != -1 ? options.Seed : null,
            Stream = true,
            Stop = options.StopSequences?.ToList(),
            Grammar = options.Grammar,
            JsonSchema = ParseStructuredSchema(options.Grammar, options.JsonSchema)
        };

        var json = JsonSerializer.Serialize(request, JsonOptions.TypeInfoOf(request));
        using var content = new StringContent(json, Encoding.UTF8, "application/json");

        using var httpRequest = new HttpRequestMessage(HttpMethod.Post, $"{_baseUrl}/completion")
        {
            Content = content
        };

        using var response = await SendWithTimeoutAsync(
            httpRequest,
            HttpCompletionOption.ResponseHeadersRead,
            cancellationToken);

        await EnsureSuccessOrThrowContextExceptionAsync(response, _maxContextLength, cancellationToken);

        await using var stream = await response.Content.ReadAsStreamAsync(cancellationToken);
        using var reader = new StreamReader(stream);

        while (await reader.ReadLineAsync(cancellationToken) is { } line)
        {
            if (string.IsNullOrEmpty(line))
                continue;

            if (!line.StartsWith("data: ", StringComparison.Ordinal))
                continue;

            var data = line[6..];

            CompletionChunk? chunk;
            try
            {
                chunk = JsonSerializer.Deserialize(data, JsonOptions.TypeInfo<CompletionChunk>());
            }
            catch (Exception ex)
            {
                Trace.TraceInformation($"[LlamaServerClient] Completion SSE chunk deserialization failed: {ex.Message}");
                continue;
            }

            if (chunk is null)
                continue;

            if (chunk.Stop)
            {
                // The final chunk carries why generation ended (and, on some builds, a last piece of text).
                yield return new CompletionStreamData
                {
                    TextDelta = string.IsNullOrEmpty(chunk.Content) ? null : chunk.Content,
                    FinishReason = MapStopType(chunk.StopType),
                    PromptTokens = chunk.TokensEvaluated,
                    CompletionTokens = chunk.TokensPredicted,
                    Timings = chunk.Timings,
                };
                break;
            }

            if (!string.IsNullOrEmpty(chunk.Content))
            {
                yield return new CompletionStreamData { TextDelta = chunk.Content };
            }
        }
    }

    /// <summary>
    /// Maps llama-server's native <c>stop_type</c> to the OpenAI-style finish reason the rest of the library uses:
    /// <c>limit</c> (reached <c>n_predict</c>) → <c>"length"</c>; <c>eos</c> and <c>word</c> → <c>"stop"</c>.
    /// An absent or unknown value (an older server) is <c>null</c> — not guessed.
    /// </summary>
    internal static string? MapStopType(string? stopType) => stopType switch
    {
        "limit" => "length",
        "eos" or "word" => "stop",
        _ => null,
    };

    /// <summary>
    /// Checks if the server is healthy.
    /// </summary>
    public async Task<bool> CheckHealthAsync(CancellationToken cancellationToken = default)
    {
        try
        {
            using var response = await SendWithTimeoutAsync(
                new HttpRequestMessage(HttpMethod.Get, $"{_baseUrl}/health"),
                HttpCompletionOption.ResponseContentRead,
                cancellationToken);
            return response.IsSuccessStatusCode;
        }
        catch (Exception ex) when (ex is not OperationCanceledException || !cancellationToken.IsCancellationRequested)
        {
            Trace.TraceInformation($"[LlamaServerClient] Health check failed: {ex.Message}");
            return false;
        }
    }

    #region Tokenize API

    /// <summary>
    /// Counts the number of tokens in the given text using the server's tokenizer.
    /// </summary>
    private Task<int> CountTokensCoreAsync(
        string text,
        CancellationToken cancellationToken = default)
        => TokenizeCountAsync(text, addSpecial: false, cancellationToken);

    /// <summary>
    /// Tokenizes <paramref name="text"/> on the server. With <paramref name="addSpecial"/>, the tokens the model adds
    /// around a prompt (a BOS token, for models that use one) are counted too — as generation counts a prompt.
    /// </summary>
    private async Task<int> TokenizeCountAsync(string text, bool addSpecial, CancellationToken cancellationToken)
    {
        var request = new TokenizeRequest { Content = text, AddSpecial = addSpecial ? true : null };
        var json = JsonSerializer.Serialize(request, JsonOptions.TypeInfoOf(request));
        using var content = new StringContent(json, Encoding.UTF8, "application/json");

        using var response = await PostWithTimeoutAsync(
            $"{_baseUrl}/tokenize",
            content,
            cancellationToken);

        response.EnsureSuccessStatusCode();

        var responseJson = await response.Content.ReadAsStringAsync(cancellationToken);
        var result = JsonSerializer.Deserialize(responseJson, JsonOptions.TypeInfo<TokenizeResponse>());

        return result?.Tokens?.Count ?? 0;
    }

    /// <summary>
    /// Counts the tokens of the prompt a chat request renders to: the messages through the model's chat template,
    /// with the request's tools, tool choice and thinking setting — what <c>/v1/chat/completions</c> would send the
    /// model. Uses the server's <c>/apply-template</c>, then <c>/tokenize</c>. Null when the server has no
    /// <c>/apply-template</c>.
    /// </summary>
    private async Task<int?> CountChatPromptTokensCoreAsync(
        IEnumerable<ChatCompletionMessage> messages,
        ChatCompletionOptions? options,
        CancellationToken cancellationToken = default)
    {
        var request = BuildChatRequest(messages, options ?? new ChatCompletionOptions(), stream: false);
        var json = JsonSerializer.Serialize(request, JsonOptions.TypeInfoOf(request));
        using var content = new StringContent(json, Encoding.UTF8, "application/json");

        using var response = await PostWithTimeoutAsync(
            $"{_baseUrl}/apply-template",
            content,
            cancellationToken);

        if (response.StatusCode == System.Net.HttpStatusCode.NotFound)
            return null;

        await EnsureSuccessOrThrowContextExceptionAsync(response, _maxContextLength, cancellationToken);

        var responseJson = await response.Content.ReadAsStringAsync(cancellationToken);
        var rendered = JsonSerializer.Deserialize(responseJson, JsonOptions.TypeInfo<ApplyTemplateResponse>());
        if (rendered?.Prompt is not { } prompt)
            return null;

        // Generation tokenizes the rendered prompt with the model's special tokens (a BOS, where the model uses one).
        return await TokenizeCountAsync(prompt, addSpecial: true, cancellationToken);
    }

    #endregion

    #region Embedding API

    /// <summary>
    /// Generates embeddings for a single text input.
    /// Requires server started with --embedding flag.
    /// </summary>
    public async Task<float[]> GenerateEmbeddingAsync(
        string input,
        CancellationToken cancellationToken = default)
    {
        var result = await GenerateEmbeddingsBatchAsync([input], cancellationToken);
        return result[0];
    }

    /// <summary>
    /// Generates embeddings for multiple text inputs in batch.
    /// Requires server started with --embedding flag.
    /// </summary>
    private async Task<float[][]> GenerateEmbeddingsBatchCoreAsync(
        IReadOnlyList<string> inputs,
        CancellationToken cancellationToken = default)
    {
        var request = new EmbeddingRequest
        {
            Input = inputs.Count == 1 ? inputs[0] : inputs
        };

        var json = JsonSerializer.Serialize(request, JsonOptions.TypeInfoOf(request));
        using var content = new StringContent(json, Encoding.UTF8, "application/json");

        using var response = await PostWithTimeoutAsync(
            $"{_baseUrl}/v1/embeddings",
            content,
            cancellationToken);

        response.EnsureSuccessStatusCode();

        var responseJson = await response.Content.ReadAsStringAsync(cancellationToken);
        var embeddingResponse = JsonSerializer.Deserialize(responseJson, JsonOptions.TypeInfo<EmbeddingResponse>());

        if (embeddingResponse?.Data == null || embeddingResponse.Data.Count == 0)
        {
            throw new InvalidOperationException("No embeddings returned from server");
        }

        // Sort by index to ensure correct order
        return embeddingResponse.Data
            .OrderBy(d => d.Index)
            .Select(d => d.Embedding)
            .ToArray();
    }

    #endregion

    #region Reranking API

    /// <summary>
    /// Reranks documents by relevance to a query.
    /// Requires server started with --embedding and --pooling rank flags.
    /// </summary>
    private async Task<IReadOnlyList<RerankResult>> RerankCoreAsync(
        string query,
        IReadOnlyList<string> documents,
        int topN = 10,
        CancellationToken cancellationToken = default)
    {
        var request = new RerankRequest
        {
            Query = query,
            Documents = documents.ToList(),
            TopN = Math.Min(topN, documents.Count)
        };

        var json = JsonSerializer.Serialize(request, JsonOptions.TypeInfoOf(request));
        using var content = new StringContent(json, Encoding.UTF8, "application/json");

        using var response = await PostWithTimeoutAsync(
            $"{_baseUrl}/v1/rerank",
            content,
            cancellationToken);

        response.EnsureSuccessStatusCode();

        var responseJson = await response.Content.ReadAsStringAsync(cancellationToken);
        var rerankResponse = JsonSerializer.Deserialize(responseJson, JsonOptions.TypeInfo<RerankResponse>());

        return rerankResponse?.Results ?? [];
    }

    #endregion

    /// <summary>
    /// Ensures the HTTP response is successful, converting context overflow errors
    /// to <see cref="ContextLengthExceededException"/>.
    /// </summary>
    private static async Task EnsureSuccessOrThrowContextExceptionAsync(
        HttpResponseMessage response,
        int maxContextLength,
        CancellationToken cancellationToken)
    {
        if (response.IsSuccessStatusCode)
            return;

        // Read error body for context overflow detection
        var errorBody = await response.Content.ReadAsStringAsync(cancellationToken);

        if (response.StatusCode == System.Net.HttpStatusCode.BadRequest &&
            IsContextOverflowError(errorBody))
        {
            throw new ContextLengthExceededException(null, maxContextLength);
        }

        // Not a recognized context-overflow shape — surface the body we already read instead
        // of discarding it and falling through to EnsureSuccessStatusCode()'s generic message.
        throw new InferenceBackendException(response.StatusCode, errorBody);
    }

    private static bool IsContextOverflowError(string errorBody)
    {
        // llama-server returns errors like:
        // "the prompt is too long" / "input exceeds context" / "context length exceeded"
        return errorBody.Contains("too long", StringComparison.OrdinalIgnoreCase) ||
               errorBody.Contains("context", StringComparison.OrdinalIgnoreCase) &&
               (errorBody.Contains("exceed", StringComparison.OrdinalIgnoreCase) ||
                errorBody.Contains("overflow", StringComparison.OrdinalIgnoreCase) ||
                errorBody.Contains("too large", StringComparison.OrdinalIgnoreCase));
    }

    /// <summary>
    /// Ties this client to the server process it talks to, so a request that fails because that process exited reports
    /// <see cref="InferenceBackendExitedException"/> (exit code and the server's last output) instead of the bare
    /// transport failure. Views made afterwards share it.
    /// </summary>
    internal void AttachServer(LlamaServerProcess server) =>
        AttachServerState(() => new ServerState(server.IsRunning, server.ExitCode, server.RecentLog));

    /// <summary>The seam behind <see cref="AttachServer"/>: reads the server's state when a request fails.</summary>
    internal void AttachServerState(Func<ServerState> state) => _serverState = state;

    /// <summary>What a failed request needs to know about its server.</summary>
    internal readonly record struct ServerState(bool IsRunning, int? ExitCode, string RecentLog);

    /// <summary>
    /// The exception to throw for <paramref name="failure"/>: <see cref="InferenceBackendExitedException"/> when the
    /// attached server has exited, otherwise <see langword="null"/> (rethrow as is). The process may still be dying when
    /// the connection is refused or cut, so its exit gets a moment to register.
    /// </summary>
    private async Task<Exception?> TranslateFailureAsync(Exception failure)
    {
        if (_serverState is not { } read || failure is not (HttpRequestException or IOException))
            return null;

        var state = read();
        for (var i = 0; i < 20 && state.IsRunning; i++)
        {
            await Task.Delay(100).ConfigureAwait(false);
            state = read();
        }

        return state.IsRunning ? null : new InferenceBackendExitedException(state.ExitCode, state.RecentLog, failure);
    }

    private async Task<T> GuardAsync<T>(Func<Task<T>> call)
    {
        try
        {
            return await call().ConfigureAwait(false);
        }
        catch (Exception ex)
        {
            if (await TranslateFailureAsync(ex).ConfigureAwait(false) is { } translated)
                throw translated;
            throw;
        }
    }

    private async IAsyncEnumerable<T> GuardStream<T>(
        IAsyncEnumerable<T> source,
        [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        var enumerator = source.GetAsyncEnumerator(cancellationToken);
        try
        {
            while (true)
            {
                T item;
                try
                {
                    if (!await enumerator.MoveNextAsync().ConfigureAwait(false))
                        yield break;
                    item = enumerator.Current;
                }
                catch (Exception ex)
                {
                    if (await TranslateFailureAsync(ex).ConfigureAwait(false) is { } translated)
                        throw translated;
                    throw;
                }

                yield return item;
            }
        }
        finally
        {
            await enumerator.DisposeAsync().ConfigureAwait(false);
        }
    }

    /// <inheritdoc cref="GenerateChatCoreAsync"/>
    public IAsyncEnumerable<string> GenerateChatAsync(
        IEnumerable<ChatCompletionMessage> messages,
        ChatCompletionOptions? options = null,
        CancellationToken cancellationToken = default) =>
        GuardStream(GenerateChatCoreAsync(messages, options, cancellationToken), cancellationToken);

    /// <inheritdoc cref="GenerateChatStreamCoreAsync"/>
    public IAsyncEnumerable<ChatStreamData> GenerateChatStreamAsync(
        IEnumerable<ChatCompletionMessage> messages,
        ChatCompletionOptions? options = null,
        CancellationToken cancellationToken = default) =>
        GuardStream(GenerateChatStreamCoreAsync(messages, options, cancellationToken), cancellationToken);

    /// <inheritdoc cref="GenerateChatWithToolsCoreAsync"/>
    public Task<ChatCompletionFullResponse> GenerateChatWithToolsAsync(
        IEnumerable<ChatCompletionMessage> messages,
        ChatCompletionOptions? options = null,
        CancellationToken cancellationToken = default) =>
        GuardAsync(() => GenerateChatWithToolsCoreAsync(messages, options, cancellationToken));

    /// <inheritdoc cref="GenerateStreamCoreAsync"/>
    public IAsyncEnumerable<CompletionStreamData> GenerateStreamAsync(
        string prompt,
        CompletionOptions? options = null,
        CancellationToken cancellationToken = default) =>
        GuardStream(GenerateStreamCoreAsync(prompt, options, cancellationToken), cancellationToken);

    /// <inheritdoc cref="CountTokensCoreAsync"/>
    public Task<int> CountTokensAsync(string text, CancellationToken cancellationToken = default) =>
        GuardAsync(() => CountTokensCoreAsync(text, cancellationToken));

    /// <inheritdoc cref="CountChatPromptTokensCoreAsync"/>
    public Task<int?> CountChatPromptTokensAsync(
        IEnumerable<ChatCompletionMessage> messages,
        ChatCompletionOptions? options = null,
        CancellationToken cancellationToken = default) =>
        GuardAsync(() => CountChatPromptTokensCoreAsync(messages, options, cancellationToken));

    /// <inheritdoc cref="GenerateEmbeddingsBatchCoreAsync"/>
    public Task<float[][]> GenerateEmbeddingsBatchAsync(IReadOnlyList<string> inputs, CancellationToken cancellationToken = default) =>
        GuardAsync(() => GenerateEmbeddingsBatchCoreAsync(inputs, cancellationToken));

    /// <inheritdoc cref="RerankCoreAsync"/>
    public Task<IReadOnlyList<RerankResult>> RerankAsync(
        string query,
        IReadOnlyList<string> documents,
        int topN = 10,
        CancellationToken cancellationToken = default) =>
        GuardAsync(() => RerankCoreAsync(query, documents, topN, cancellationToken));

    public void Dispose()
    {
        if (_ownsHttpClient)
        {
            _httpClient.Dispose();
        }
    }

    /// <summary>
    /// Builds the chat completion request body shared by all three chat paths (streaming,
    /// structured-streaming, non-streaming with tools), so thinking control and every other option
    /// map identically. Forwards <c>enable_thinking</c> via <c>chat_template_kwargs</c> when
    /// <see cref="ChatCompletionOptions.EnableThinking"/> is set; omits it (model default) when null.
    /// </summary>
    internal static ChatCompletionRequest BuildChatRequest(
        IEnumerable<ChatCompletionMessage> messages,
        ChatCompletionOptions options,
        bool stream)
    {
        return new ChatCompletionRequest
        {
            Messages = messages.ToList(),
            MaxTokens = options.MaxTokens,
            Temperature = options.Temperature,
            TopP = options.TopP,
            TopK = options.TopK > 0 ? options.TopK : null,
            MinP = options.MinP > 0 ? options.MinP : null,
            RepeatPenalty = options.RepeatPenalty != 1.0f ? options.RepeatPenalty : null,
            FrequencyPenalty = options.FrequencyPenalty != 0 ? options.FrequencyPenalty : null,
            PresencePenalty = options.PresencePenalty != 0 ? options.PresencePenalty : null,
            DryMultiplier = options.DryMultiplier,
            DryBase = options.DryBase,
            DryAllowedLength = options.DryAllowedLength,
            DryPenaltyLastN = options.DryPenaltyLastN,
            RepeatLastN = options.RepeatLastN,
            Seed = options.Seed != -1 ? options.Seed : null,
            Stream = stream,
            // The server's token accounting on a streamed response; without it the last chunk carries no usage.
            StreamOptions = stream ? new ChatStreamOptions { IncludeUsage = true } : null,
            Stop = options.StopSequences?.ToList(),
            Grammar = options.Grammar,
            ResponseFormat = BuildChatResponseFormat(options.Grammar, options.JsonSchema),
            Tools = options.Tools?.ToList(),
            ToolChoice = BuildToolChoiceNode(options.ToolChoice),
            ChatTemplateKwargs = options.EnableThinking is { } enableThinking
                ? new Dictionary<string, object> { ["enable_thinking"] = enableThinking }
                : null
        };
    }

    /// <summary>
    /// Builds the OpenAI-compatible <c>tool_choice</c> wire value: omitted (null, server default
    /// "auto") for <see cref="LlamaToolChoiceMode.Auto"/> or an unset choice, a bare string for
    /// <c>none</c>/<c>required</c>, or <c>{"type":"function","function":{"name":...}}</c> to force
    /// one named function.
    /// </summary>
    internal static JsonNode? BuildToolChoiceNode(LlamaToolChoice? toolChoice) => toolChoice?.Mode switch
    {
        null or LlamaToolChoiceMode.Auto => null,
        LlamaToolChoiceMode.None => JsonValue.Create("none"),
        LlamaToolChoiceMode.Required => JsonValue.Create("required"),
        LlamaToolChoiceMode.Function => new JsonObject
        {
            ["type"] = "function",
            ["function"] = new JsonObject { ["name"] = toolChoice.FunctionName }
        },
        _ => null
    };

    /// <summary>
    /// Parses <paramref name="jsonSchema"/> (the JSON-schema string carried by
    /// <c>GenerationOptions.JsonSchema</c>) into a node so it serializes as a JSON <b>object</b>, and
    /// enforces llama-server's rule that a single request cannot carry both a grammar and a json_schema
    /// constraint. Returns <c>null</c> when no schema is set. Throws <see cref="ArgumentException"/> on
    /// invalid JSON or a grammar+schema conflict so the contract surfaces at call time instead of as an
    /// opaque HTTP 400 from the server.
    /// </summary>
    /// <remarks>
    /// The public option is a JSON <i>string</i>; serializing that string verbatim would emit a quoted
    /// JSON string, which llama-server rejects (it expects an object). This is the single conversion
    /// seam shared by the native <c>/completion</c> path and the chat <c>response_format</c> builder.
    /// </remarks>
    internal static JsonNode? ParseStructuredSchema(string? grammar, string? jsonSchema)
    {
        if (string.IsNullOrWhiteSpace(jsonSchema))
            return null;

        if (!string.IsNullOrWhiteSpace(grammar))
            throw new ArgumentException(
                "Grammar and JsonSchema cannot both be set: llama-server rejects a request that " +
                "specifies both a grammar and a json_schema constraint. Set only one.");

        try
        {
            return JsonNode.Parse(jsonSchema)
                ?? throw new ArgumentException("JsonSchema must be a JSON object or array, not null.", nameof(jsonSchema));
        }
        catch (JsonException ex)
        {
            throw new ArgumentException(
                $"JsonSchema is not valid JSON and cannot be sent to llama-server: {ex.Message}",
                nameof(jsonSchema), ex);
        }
    }

    /// <summary>
    /// Builds the OpenAI-compatible <c>response_format</c> for the chat endpoint from a JSON-schema
    /// string, or <c>null</c> when no schema is set. llama-server's <c>/v1/chat/completions</c> reads
    /// the schema from <c>response_format.json_schema.schema</c>; a root-level <c>json_schema</c> field
    /// (which the native <c>/completion</c> endpoint uses) is not honored here.
    /// See <see cref="ParseStructuredSchema"/> for validation.
    /// </summary>
    internal static ChatResponseFormat? BuildChatResponseFormat(string? grammar, string? jsonSchema)
        => ParseStructuredSchema(grammar, jsonSchema) is { } schema
            ? new ChatResponseFormat { JsonSchema = new ChatJsonSchema { Schema = schema } }
            : null;
}

#region Request/Response Models

/// <summary>
/// Chat completion message.
/// </summary>
public sealed class ChatCompletionMessage
{
    [JsonPropertyName("role")]
    public required string Role { get; init; }

    [JsonPropertyName("content")]
    public string? Content { get; init; }

    [JsonPropertyName("tool_calls")]
    public List<ToolCallMessage>? ToolCalls { get; init; }

    [JsonPropertyName("tool_call_id")]
    public string? ToolCallId { get; init; }

    public static ChatCompletionMessage System(string content) => new() { Role = "system", Content = content };
    public static ChatCompletionMessage User(string content) => new() { Role = "user", Content = content };
    public static ChatCompletionMessage Assistant(string content) => new() { Role = "assistant", Content = content };
    public static ChatCompletionMessage Tool(string toolCallId, string content) => new() { Role = "tool", Content = content, ToolCallId = toolCallId };
}

/// <summary>
/// Options for chat completion.
/// </summary>
public sealed class ChatCompletionOptions
{
    public int MaxTokens { get; init; } = 256;
    public float Temperature { get; init; } = 0.7f;
    public float TopP { get; init; } = 0.9f;
    public int TopK { get; init; } = 50;
    public float MinP { get; init; } = 0.05f;
    public float RepeatPenalty { get; init; } = 1.1f;
    public float FrequencyPenalty { get; init; }
    public float PresencePenalty { get; init; }

    // Advanced anti-repetition (standard llama-server samplers; null = server default, omitted from request)
    public float? DryMultiplier { get; init; }
    public float? DryBase { get; init; }
    public int? DryAllowedLength { get; init; }
    public int? DryPenaltyLastN { get; init; }
    public int? RepeatLastN { get; init; }

    public int Seed { get; init; } = -1;
    public IReadOnlyList<string>? StopSequences { get; init; }

    /// <summary>
    /// Grammar constraint in GBNF format (Phase 3).
    /// </summary>
    public string? Grammar { get; init; }

    /// <summary>
    /// JSON schema for structured output (Phase 3).
    /// When set, output will be constrained to match this schema.
    /// </summary>
    public string? JsonSchema { get; init; }

    /// <summary>
    /// Tool definitions available for the model.
    /// When provided, the model may respond with tool_calls.
    /// </summary>
    public IReadOnlyList<ToolDefinition>? Tools { get; init; }

    /// <summary>
    /// Controls whether/which tool the model must call. Null = auto (model decides). Only
    /// meaningful when <see cref="Tools"/> is set.
    /// </summary>
    public LlamaToolChoice? ToolChoice { get; init; }

    /// <summary>
    /// Thinking control forwarded to the GGUF chat template via <c>chat_template_kwargs.enable_thinking</c>.
    /// Null = omit (model template default: Qwen3 thinks, Gemma does not). False = suppress reasoning
    /// (direct answer); True = force reasoning on. Only applies to the chat path (/v1/chat/completions),
    /// which is where the template runs — the raw /completion path has no template to control.
    /// </summary>
    public bool? EnableThinking { get; init; }
}

/// <summary>
/// Options for text completion.
/// </summary>
public sealed class CompletionOptions
{
    public int MaxTokens { get; init; } = 256;
    public float Temperature { get; init; } = 0.7f;
    public float TopP { get; init; } = 0.9f;
    public int TopK { get; init; } = 50;
    public float MinP { get; init; } = 0.05f;
    public float RepeatPenalty { get; init; } = 1.1f;
    public float FrequencyPenalty { get; init; }
    public float PresencePenalty { get; init; }

    // Advanced anti-repetition (standard llama-server samplers; null = server default, omitted from request)
    public float? DryMultiplier { get; init; }
    public float? DryBase { get; init; }
    public int? DryAllowedLength { get; init; }
    public int? DryPenaltyLastN { get; init; }
    public int? RepeatLastN { get; init; }

    public int Seed { get; init; } = -1;
    public IReadOnlyList<string>? StopSequences { get; init; }

    /// <summary>
    /// Grammar constraint in GBNF format (Phase 3).
    /// </summary>
    public string? Grammar { get; init; }

    /// <summary>
    /// JSON schema for structured output (Phase 3).
    /// When set, output will be constrained to match this schema.
    /// </summary>
    public string? JsonSchema { get; init; }
}

internal sealed class ChatCompletionRequest
{
    public List<ChatCompletionMessage>? Messages { get; set; }
    public ChatStreamOptions? StreamOptions { get; set; }
    public int? MaxTokens { get; set; }
    public float? Temperature { get; set; }
    public float? TopP { get; set; }
    public int? TopK { get; set; }
    public float? MinP { get; set; }
    public float? RepeatPenalty { get; set; }
    public float? FrequencyPenalty { get; set; }
    public float? PresencePenalty { get; set; }

    // Advanced anti-repetition (snake_case: dry_multiplier, dry_base, dry_allowed_length,
    // dry_penalty_last_n, repeat_last_n). Null -> omitted (WhenWritingNull) -> server default.
    public float? DryMultiplier { get; set; }
    public float? DryBase { get; set; }
    public int? DryAllowedLength { get; set; }
    public int? DryPenaltyLastN { get; set; }
    public int? RepeatLastN { get; set; }

    public int? Seed { get; set; }
    public bool Stream { get; set; }
    public List<string>? Stop { get; set; }

    /// <summary>
    /// Grammar constraint in GBNF format (Phase 3).
    /// </summary>
    public string? Grammar { get; set; }

    /// <summary>
    /// Structured-output constraint for the OpenAI-compatible chat endpoint. llama-server's
    /// <c>/v1/chat/completions</c> reads the schema from <c>response_format.json_schema.schema</c>
    /// (OpenAI form), NOT a root-level <c>json_schema</c> field — sending the latter (as a string) is
    /// rejected with HTTP 400. Built from <c>GenerationOptions.JsonSchema</c> in BuildChatRequest.
    /// </summary>
    [JsonPropertyName("response_format")]
    public ChatResponseFormat? ResponseFormat { get; set; }

    /// <summary>
    /// Tool definitions (OpenAI-compatible).
    /// </summary>
    public List<ToolDefinition>? Tools { get; set; }

    /// <summary>
    /// OpenAI-compatible <c>tool_choice</c> — a bare string (<c>"none"</c>/<c>"required"</c>) or
    /// <c>{"type":"function","function":{"name":...}}</c>, or omitted entirely for the server's
    /// "auto" default. Built from <see cref="ChatCompletionOptions.ToolChoice"/> in BuildChatRequest
    /// via <see cref="LlamaServerClient.BuildToolChoiceNode"/>.
    /// </summary>
    public JsonNode? ToolChoice { get; set; }

    /// <summary>
    /// Re-use KV cache from previous request if possible.
    /// Reduces first token latency for prompts with common prefixes.
    /// </summary>
    public bool CachePrompt { get; set; } = true;

    /// <summary>
    /// Extra kwargs forwarded to the GGUF chat template by llama-server. Used to pass
    /// <c>enable_thinking</c> to thinking-aware templates (Qwen3) so a thinking-default-on model can
    /// be told to emit a direct answer instead of a reasoning block. Null = omit (model template
    /// default applies). Unknown to older llama-server builds, which ignore unrecognized fields.
    /// </summary>
    public Dictionary<string, object>? ChatTemplateKwargs { get; set; }
}

internal sealed class CompletionRequest
{
    public string? Prompt { get; set; }
    public int? NPredict { get; set; }
    public float? Temperature { get; set; }
    public float? TopP { get; set; }
    public int? TopK { get; set; }
    public float? MinP { get; set; }
    public float? RepeatPenalty { get; set; }
    public float? FrequencyPenalty { get; set; }
    public float? PresencePenalty { get; set; }

    // Advanced anti-repetition (snake_case: dry_multiplier, dry_base, dry_allowed_length,
    // dry_penalty_last_n, repeat_last_n). Null -> omitted (WhenWritingNull) -> server default.
    public float? DryMultiplier { get; set; }
    public float? DryBase { get; set; }
    public int? DryAllowedLength { get; set; }
    public int? DryPenaltyLastN { get; set; }
    public int? RepeatLastN { get; set; }

    public int? Seed { get; set; }
    public bool Stream { get; set; }
    public List<string>? Stop { get; set; }

    /// <summary>
    /// Grammar constraint in GBNF format (Phase 3).
    /// </summary>
    public string? Grammar { get; set; }

    /// <summary>
    /// JSON schema (as a JSON <b>object</b>) for structured output on the native <c>/completion</c>
    /// endpoint. Serialized to the root <c>json_schema</c> field, which llama-server expects to be an
    /// object — so <c>GenerationOptions.JsonSchema</c> (a JSON string) is parsed to a node via
    /// <see cref="LlamaServerClient.ParseStructuredSchema"/> before assignment.
    /// </summary>
    public JsonNode? JsonSchema { get; set; }

    /// <summary>
    /// Re-use KV cache from previous request if possible.
    /// Reduces first token latency for prompts with common prefixes.
    /// </summary>
    public bool CachePrompt { get; set; } = true;
}

/// <summary>
/// OpenAI-compatible <c>response_format</c> for structured chat output. llama-server's
/// <c>/v1/chat/completions</c> reads the schema from <c>response_format.json_schema.schema</c>.
/// </summary>
internal sealed class ChatResponseFormat
{
    [JsonPropertyName("type")]
    public string Type { get; set; } = "json_schema";

    [JsonPropertyName("json_schema")]
    public ChatJsonSchema? JsonSchema { get; set; }
}

/// <summary>
/// The nested schema payload of <see cref="ChatResponseFormat"/> (OpenAI form:
/// <c>{ "name": ..., "schema": {...} }</c>).
/// </summary>
internal sealed class ChatJsonSchema
{
    [JsonPropertyName("name")]
    public string Name { get; set; } = "response";

    [JsonPropertyName("schema")]
    public JsonNode? Schema { get; set; }
}

internal sealed class ChatCompletionChunk
{
    public List<ChatCompletionChoice>? Choices { get; set; }

    /// <summary>OpenAI-compatible usage — on the last chunk when <c>stream_options.include_usage</c> is set.</summary>
    public ChatCompletionUsage? Usage { get; set; }

    /// <summary>llama.cpp's own accounting, on the last chunk of every build that has it.</summary>
    public LlamaServerTimings? Timings { get; set; }
}

internal sealed class ChatStreamOptions
{
    public bool IncludeUsage { get; set; }
}

/// <summary>
/// llama.cpp's <c>timings</c> object, as the server reports it. Prompt figures cover only the tokens evaluated for this
/// request (<c>prompt_n</c>), not those reused from the prompt cache (<c>cache_n</c>), so a rate derived from the
/// OpenAI-compatible <c>usage.prompt_tokens</c> is wrong whenever the cache hits — use the server's rates.
/// </summary>
public sealed class LlamaServerTimings
{
    /// <summary>Prompt tokens reused from the server's prompt cache (<c>cache_n</c>).</summary>
    [JsonPropertyName("cache_n")]
    public int? CacheN { get; set; }

    /// <summary>Prompt tokens evaluated for this request (<c>prompt_n</c>).</summary>
    [JsonPropertyName("prompt_n")]
    public int? PromptN { get; set; }

    /// <summary>Time spent evaluating the prompt, in milliseconds (<c>prompt_ms</c>).</summary>
    [JsonPropertyName("prompt_ms")]
    public double? PromptMs { get; set; }

    /// <summary>Prompt evaluation rate the server measured (<c>prompt_per_second</c>).</summary>
    [JsonPropertyName("prompt_per_second")]
    public double? PromptPerSecond { get; set; }

    /// <summary>Tokens generated, reasoning included (<c>predicted_n</c>).</summary>
    [JsonPropertyName("predicted_n")]
    public int? PredictedN { get; set; }

    /// <summary>Time spent generating, in milliseconds (<c>predicted_ms</c>).</summary>
    [JsonPropertyName("predicted_ms")]
    public double? PredictedMs { get; set; }

    /// <summary>Generation rate the server measured (<c>predicted_per_second</c>).</summary>
    [JsonPropertyName("predicted_per_second")]
    public double? PredictedPerSecond { get; set; }
}

internal sealed class ChatCompletionChoice
{
    public ChatCompletionDelta? Delta { get; set; }
    public string? FinishReason { get; set; }
}

internal sealed class ChatCompletionDelta
{
    public string? Content { get; set; }

    [JsonPropertyName("reasoning_content")]
    public string? ReasoningContent { get; set; }

    [JsonPropertyName("tool_calls")]
    public List<ToolCallDelta>? ToolCalls { get; set; }
}

/// <summary>
/// Tool call delta in streaming response.
/// </summary>
public sealed class ToolCallDelta
{
    [JsonPropertyName("index")]
    public int Index { get; set; }

    [JsonPropertyName("id")]
    public string? Id { get; set; }

    [JsonPropertyName("type")]
    public string? Type { get; set; }

    [JsonPropertyName("function")]
    public FunctionCallDelta? Function { get; set; }
}

/// <summary>
/// Function call delta in streaming response.
/// </summary>
public sealed class FunctionCallDelta
{
    [JsonPropertyName("name")]
    public string? Name { get; set; }

    [JsonPropertyName("arguments")]
    public string? Arguments { get; set; }
}

internal sealed class CompletionChunk
{
    public string? Content { get; set; }
    public bool Stop { get; set; }

    /// <summary>On the final chunk: <c>eos</c>, <c>limit</c>, <c>word</c> or <c>none</c>.</summary>
    public string? StopType { get; set; }

    /// <summary>On the final chunk: tokens generated.</summary>
    public int? TokensPredicted { get; set; }

    /// <summary>On the final chunk: prompt tokens evaluated.</summary>
    public int? TokensEvaluated { get; set; }

    /// <summary>On the final chunk: llama.cpp's timings.</summary>
    public LlamaServerTimings? Timings { get; set; }
}

/// <summary>
/// Structured streaming data from a raw text completion (<c>/completion</c>).
/// </summary>
public sealed class CompletionStreamData
{
    /// <summary>
    /// Text content delta.
    /// </summary>
    public string? TextDelta { get; init; }

    /// <summary>
    /// Finish reason (present only on the final chunk): <c>"length"</c> or <c>"stop"</c>, or <c>null</c> when the
    /// server did not say.
    /// </summary>
    public string? FinishReason { get; init; }

    /// <summary>Prompt tokens the server evaluated (final chunk), or null when it did not say.</summary>
    public int? PromptTokens { get; init; }

    /// <summary>Tokens the server generated (final chunk), or null when it did not say.</summary>
    public int? CompletionTokens { get; init; }

    /// <summary>The server's timings for the whole completion (final chunk), or null when it did not send them.</summary>
    public LlamaServerTimings? Timings { get; init; }
}

/// <summary>
/// Tool call in a message (OpenAI format).
/// </summary>
public sealed class ToolCallMessage
{
    [JsonPropertyName("id")]
    public string? Id { get; set; }

    [JsonPropertyName("type")]
    public string Type { get; set; } = "function";

    [JsonPropertyName("function")]
    public FunctionCallMessage? Function { get; set; }
}

/// <summary>
/// Function call details.
/// </summary>
public sealed class FunctionCallMessage
{
    [JsonPropertyName("name")]
    public string? Name { get; set; }

    [JsonPropertyName("arguments")]
    public string? Arguments { get; set; }
}

/// <summary>
/// Tool definition for the request (OpenAI format).
/// </summary>
public sealed class ToolDefinition
{
    [JsonPropertyName("type")]
    public string Type { get; set; } = "function";

    [JsonPropertyName("function")]
    public FunctionDefinition? Function { get; set; }
}

/// <summary>
/// Function definition for a tool.
/// </summary>
public sealed class FunctionDefinition
{
    [JsonPropertyName("name")]
    public string? Name { get; set; }

    [JsonPropertyName("description")]
    public string? Description { get; set; }

    [JsonPropertyName("parameters")]
    public JsonElement? Parameters { get; set; }
}

/// <summary>
/// Which <see cref="LlamaToolChoice"/> mode a value represents, mirroring the OpenAI-compatible
/// <c>tool_choice</c> wire values llama-server accepts.
/// </summary>
public enum LlamaToolChoiceMode
{
    /// <summary>Model decides freely whether to call a tool.</summary>
    Auto = 0,

    /// <summary>Suppress tool calls even though <see cref="ChatCompletionOptions.Tools"/> is set.</summary>
    None = 1,

    /// <summary>Force the model to call at least one tool.</summary>
    Required = 2,

    /// <summary>Force the model to call one specific, named function.</summary>
    Function = 3
}

/// <summary>
/// Controls whether/which tool the model must call. Only meaningful when
/// <see cref="ChatCompletionOptions.Tools"/> is set.
/// </summary>
public sealed class LlamaToolChoice
{
    /// <summary>Model decides freely whether to call a tool. Equivalent to leaving <see cref="ChatCompletionOptions.ToolChoice"/> unset.</summary>
    public static readonly LlamaToolChoice Auto = new(LlamaToolChoiceMode.Auto, null);

    /// <summary>Suppress tool calls even though <see cref="ChatCompletionOptions.Tools"/> is set.</summary>
    public static readonly LlamaToolChoice None = new(LlamaToolChoiceMode.None, null);

    /// <summary>Force the model to call at least one tool.</summary>
    public static readonly LlamaToolChoice Required = new(LlamaToolChoiceMode.Required, null);

    /// <summary>Force the model to call the named function.</summary>
    public static LlamaToolChoice Function(string name) =>
        new(LlamaToolChoiceMode.Function, name ?? throw new ArgumentNullException(nameof(name)));

    /// <summary>Which mode this instance represents.</summary>
    public LlamaToolChoiceMode Mode { get; }

    /// <summary>The forced function name when <see cref="Mode"/> is <see cref="LlamaToolChoiceMode.Function"/>; null otherwise.</summary>
    public string? FunctionName { get; }

    private LlamaToolChoice(LlamaToolChoiceMode mode, string? functionName)
    {
        Mode = mode;
        FunctionName = functionName;
    }
}

/// <summary>
/// Structured streaming data from a chat completion.
/// Exposed to LMSupply.Generator via InternalsVisibleTo for conversion to ChatStreamChunk.
/// </summary>
public sealed class ChatStreamData
{
    /// <summary>
    /// Text content delta.
    /// </summary>
    public string? TextDelta { get; init; }

    /// <summary>
    /// Reasoning/thinking content delta (b8994+: emitted in separate field before content).
    /// </summary>
    public string? ReasoningDelta { get; init; }

    /// <summary>
    /// Tool call deltas (raw SSE format).
    /// </summary>
    public List<ToolCallDelta>? ToolCallDeltas { get; init; }

    /// <summary>
    /// Finish reason (present only on the final chunk).
    /// </summary>
    public string? FinishReason { get; init; }

    /// <summary>
    /// The server's token accounting for the whole completion (reasoning included), on the chunk that carries it.
    /// </summary>
    public ChatCompletionUsage? Usage { get; init; }

    /// <summary>
    /// The server's timings for the whole completion, on the chunk that carries them. Null on builds that do not send them.
    /// </summary>
    public LlamaServerTimings? Timings { get; init; }
}

/// <summary>
/// Full (non-streaming) chat completion response.
/// </summary>
public sealed class ChatCompletionFullResponse
{
    public List<ChatCompletionFullChoice>? Choices { get; set; }

    /// <summary>
    /// Token accounting the server reports for the completion (OpenAI-compatible <c>usage</c> object).
    /// </summary>
    public ChatCompletionUsage? Usage { get; set; }

    /// <summary>
    /// llama.cpp's <c>timings</c> for the completion. Null on builds that do not send them.
    /// </summary>
    public LlamaServerTimings? Timings { get; set; }
}

/// <summary>
/// OpenAI-compatible <c>usage</c> object of a chat completion response.
/// </summary>
public sealed class ChatCompletionUsage
{
    [JsonPropertyName("prompt_tokens")]
    public int PromptTokens { get; set; }

    [JsonPropertyName("completion_tokens")]
    public int CompletionTokens { get; set; }

    [JsonPropertyName("total_tokens")]
    public int TotalTokens { get; set; }
}

/// <summary>
/// Choice in a full chat completion response.
/// </summary>
public sealed class ChatCompletionFullChoice
{
    public ChatCompletionResponseMessage? Message { get; set; }

    [JsonPropertyName("finish_reason")]
    public string? FinishReason { get; set; }
}

/// <summary>
/// Message in a full chat completion response.
/// </summary>
public sealed class ChatCompletionResponseMessage
{
    [JsonPropertyName("role")]
    public string? Role { get; set; }

    [JsonPropertyName("content")]
    public string? Content { get; set; }

    [JsonPropertyName("reasoning_content")]
    public string? ReasoningContent { get; set; }

    [JsonPropertyName("tool_calls")]
    public List<ToolCallMessage>? ToolCalls { get; set; }
}

#endregion

#region Embedding Request/Response Models

internal sealed class EmbeddingRequest
{
    /// <summary>
    /// Input text(s) to embed. Can be a single string or array of strings.
    /// </summary>
    public required object Input { get; set; }

    /// <summary>
    /// Model identifier (optional, defaults to loaded model).
    /// </summary>
    public string Model { get; set; } = "default";

    /// <summary>
    /// Encoding format: "float" (default) or "base64".
    /// </summary>
    public string EncodingFormat { get; set; } = "float";
}

internal sealed class EmbeddingResponse
{
    public string Object { get; set; } = "list";
    public List<EmbeddingData> Data { get; set; } = [];
    public string Model { get; set; } = "";
    public EmbeddingUsage Usage { get; set; } = new();
}

internal sealed class EmbeddingData
{
    public string Object { get; set; } = "embedding";
    public float[] Embedding { get; set; } = [];
    public int Index { get; set; }
}

internal sealed class EmbeddingUsage
{
    public int PromptTokens { get; set; }
    public int TotalTokens { get; set; }
}

#endregion

#region Reranking Request/Response Models

internal sealed class RerankRequest
{
    /// <summary>
    /// The search query.
    /// </summary>
    public required string Query { get; set; }

    /// <summary>
    /// Documents to rerank.
    /// </summary>
    public required List<string> Documents { get; set; }

    /// <summary>
    /// Maximum number of results to return.
    /// </summary>
    public int TopN { get; set; } = 10;
}

internal sealed class RerankResponse
{
    public List<RerankResult> Results { get; set; } = [];
}

/// <summary>
/// Result from reranking operation.
/// </summary>
public sealed class RerankResult
{
    /// <summary>
    /// Original index of the document in the input list.
    /// </summary>
    public int Index { get; set; }

    /// <summary>
    /// Relevance score (higher is more relevant).
    /// </summary>
    public float RelevanceScore { get; set; }
}

#endregion

#region Tokenize Response Models

internal sealed class TokenizeRequest
{
    public required string Content { get; init; }

    /// <summary>Whether to add the model's special tokens (BOS); the server's default is not to.</summary>
    public bool? AddSpecial { get; init; }
}

internal sealed class TokenizeResponse
{
    public List<int>? Tokens { get; set; }
}

internal sealed class ApplyTemplateResponse
{
    public string? Prompt { get; set; }
}

#endregion
