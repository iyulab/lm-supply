using System.Diagnostics;
using System.Net.Http.Json;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.Json.Serialization;
using LMSupply.Exceptions;

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
    /// <param name="requestTimeout">Timeout for the internally-created HttpClient. Defaults to 5
    /// minutes, not .NET's 100-second default — local/CPU-bound inference, especially a multi-step
    /// tool-calling loop where each round's prompt grows with prior tool results, routinely exceeds
    /// 100 seconds per completion on hardware without a GPU. Ignored when <paramref name="httpClient"/>
    /// is supplied.</param>
    public LlamaServerClient(
        string baseUrl,
        HttpClient? httpClient = null,
        int maxContextLength = 0,
        TimeSpan? requestTimeout = null)
    {
        _baseUrl = baseUrl.TrimEnd('/');
        _maxContextLength = maxContextLength;

        if (httpClient != null)
        {
            _httpClient = httpClient;
            _ownsHttpClient = false;
        }
        else
        {
            _httpClient = new HttpClient { Timeout = requestTimeout ?? TimeSpan.FromMinutes(5) };
            _ownsHttpClient = true;
        }
    }

    /// <summary>
    /// Generates a streaming chat completion.
    /// </summary>
    public async IAsyncEnumerable<string> GenerateChatAsync(
        IEnumerable<ChatCompletionMessage> messages,
        ChatCompletionOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        options ??= new ChatCompletionOptions();

        var request = BuildChatRequest(messages, options, stream: true);

        var json = JsonSerializer.Serialize(request, JsonOptions);
        using var content = new StringContent(json, Encoding.UTF8, "application/json");

        using var httpRequest = new HttpRequestMessage(HttpMethod.Post, $"{_baseUrl}/v1/chat/completions")
        {
            Content = content
        };

        using var response = await _httpClient.SendAsync(
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
                chunk = JsonSerializer.Deserialize<ChatCompletionChunk>(data, JsonOptions);
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
    public async IAsyncEnumerable<ChatStreamData> GenerateChatStreamAsync(
        IEnumerable<ChatCompletionMessage> messages,
        ChatCompletionOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        options ??= new ChatCompletionOptions();

        var request = BuildChatRequest(messages, options, stream: true);

        var json = JsonSerializer.Serialize(request, JsonOptions);
        using var content = new StringContent(json, Encoding.UTF8, "application/json");

        using var httpRequest = new HttpRequestMessage(HttpMethod.Post, $"{_baseUrl}/v1/chat/completions")
        {
            Content = content
        };

        using var response = await _httpClient.SendAsync(
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
                chunk = JsonSerializer.Deserialize<ChatCompletionChunk>(data, JsonOptions);
            }
            catch (Exception ex)
            {
                Trace.TraceInformation($"[LlamaServerClient] Chat stream chunk deserialization failed: {ex.Message}");
                continue;
            }

            if (ReportedUsage(chunk) is { } usage)
                yield return new ChatStreamData { Usage = usage };

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
    public async Task<ChatCompletionFullResponse> GenerateChatWithToolsAsync(
        IEnumerable<ChatCompletionMessage> messages,
        ChatCompletionOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        options ??= new ChatCompletionOptions();

        var request = BuildChatRequest(messages, options, stream: false);

        var json = JsonSerializer.Serialize(request, JsonOptions);
        using var content = new StringContent(json, Encoding.UTF8, "application/json");

        using var response = await _httpClient.PostAsync(
            $"{_baseUrl}/v1/chat/completions",
            content,
            cancellationToken);

        await EnsureSuccessOrThrowContextExceptionAsync(response, _maxContextLength, cancellationToken);

        var responseJson = await response.Content.ReadAsStringAsync(cancellationToken);
        var result = JsonSerializer.Deserialize<ChatCompletionFullResponse>(responseJson, JsonOptions);

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
    public async IAsyncEnumerable<CompletionStreamData> GenerateStreamAsync(
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

        var json = JsonSerializer.Serialize(request, JsonOptions);
        using var content = new StringContent(json, Encoding.UTF8, "application/json");

        using var httpRequest = new HttpRequestMessage(HttpMethod.Post, $"{_baseUrl}/completion")
        {
            Content = content
        };

        using var response = await _httpClient.SendAsync(
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
                chunk = JsonSerializer.Deserialize<CompletionChunk>(data, JsonOptions);
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
            var response = await _httpClient.GetAsync($"{_baseUrl}/health", cancellationToken);
            return response.IsSuccessStatusCode;
        }
        catch (Exception ex)
        {
            Trace.TraceInformation($"[LlamaServerClient] Health check failed: {ex.Message}");
            return false;
        }
    }

    #region Tokenize API

    /// <summary>
    /// Counts the number of tokens in the given text using the server's tokenizer.
    /// </summary>
    public async Task<int> CountTokensAsync(
        string text,
        CancellationToken cancellationToken = default)
    {
        var request = new TokenizeRequest { Content = text };
        var json = JsonSerializer.Serialize(request, JsonOptions);
        using var content = new StringContent(json, Encoding.UTF8, "application/json");

        using var response = await _httpClient.PostAsync(
            $"{_baseUrl}/tokenize",
            content,
            cancellationToken);

        response.EnsureSuccessStatusCode();

        var responseJson = await response.Content.ReadAsStringAsync(cancellationToken);
        var result = JsonSerializer.Deserialize<TokenizeResponse>(responseJson, JsonOptions);

        return result?.Tokens?.Count ?? 0;
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
    public async Task<float[][]> GenerateEmbeddingsBatchAsync(
        IReadOnlyList<string> inputs,
        CancellationToken cancellationToken = default)
    {
        var request = new EmbeddingRequest
        {
            Input = inputs.Count == 1 ? inputs[0] : inputs
        };

        var json = JsonSerializer.Serialize(request, JsonOptions);
        using var content = new StringContent(json, Encoding.UTF8, "application/json");

        using var response = await _httpClient.PostAsync(
            $"{_baseUrl}/v1/embeddings",
            content,
            cancellationToken);

        response.EnsureSuccessStatusCode();

        var responseJson = await response.Content.ReadAsStringAsync(cancellationToken);
        var embeddingResponse = JsonSerializer.Deserialize<EmbeddingResponse>(responseJson, JsonOptions);

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
    public async Task<IReadOnlyList<RerankResult>> RerankAsync(
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

        var json = JsonSerializer.Serialize(request, JsonOptions);
        using var content = new StringContent(json, Encoding.UTF8, "application/json");

        using var response = await _httpClient.PostAsync(
            $"{_baseUrl}/v1/rerank",
            content,
            cancellationToken);

        response.EnsureSuccessStatusCode();

        var responseJson = await response.Content.ReadAsStringAsync(cancellationToken);
        var rerankResponse = JsonSerializer.Deserialize<RerankResponse>(responseJson, JsonOptions);

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
    public LlamaTimings? Timings { get; set; }
}

internal sealed class ChatStreamOptions
{
    public bool IncludeUsage { get; set; }
}

/// <summary>llama.cpp's <c>timings</c>: <c>prompt_n</c> tokens evaluated, <c>predicted_n</c> generated (reasoning included).</summary>
internal sealed class LlamaTimings
{
    public int? PromptN { get; set; }
    public int? PredictedN { get; set; }
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
}

internal sealed class TokenizeResponse
{
    public List<int>? Tokens { get; set; }
}

#endregion
