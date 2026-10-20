using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using LMSupply.Json;

namespace LMSupply.Llama.Server;

public sealed partial class LlamaServerClient
{
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

            if (chunk?.PromptProgress is { } progress)
                yield return new ChatStreamData { PromptProgress = progress };

            var choice = chunk?.Choices?.FirstOrDefault();
            if (choice is null)
                continue;

            var logprobs = ShapeLogprobs(choice.Logprobs?.Content, options.TopLogprobs);
            if (choice.Delta?.Content is not null ||
                choice.Delta?.ReasoningContent is not null ||
                choice.Delta?.ToolCalls is { Count: > 0 } ||
                logprobs is not null ||
                choice.FinishReason is not null)
            {
                yield return new ChatStreamData
                {
                    TextDelta = choice.Delta?.Content,
                    ReasoningDelta = choice.Delta?.ReasoningContent,
                    ToolCallDeltas = choice.Delta?.ToolCalls,
                    Logprobs = logprobs,
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

        foreach (var choice in result?.Choices ?? [])
        {
            if (choice.Logprobs is { } logprobs)
                logprobs.Content = ShapeLogprobs(logprobs.Content, options.TopLogprobs);
        }

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
            JsonSchema = ParseStructuredSchema(options.Grammar, options.JsonSchema),
            NProbs = WireTopLogprobs(options.TopLogprobs)
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

            var logprobs = ShapeLogprobs(chunk.CompletionProbabilities, options.TopLogprobs);

            if (chunk.Stop)
            {
                // The final chunk carries why generation ended (and, on some builds, a last piece of text).
                yield return new CompletionStreamData
                {
                    TextDelta = string.IsNullOrEmpty(chunk.Content) ? null : chunk.Content,
                    Logprobs = logprobs,
                    FinishReason = MapStopType(chunk.StopType),
                    PromptTokens = chunk.TokensEvaluated,
                    CompletionTokens = chunk.TokensPredicted,
                    Timings = chunk.Timings,
                };
                break;
            }

            if (!string.IsNullOrEmpty(chunk.Content) || logprobs is not null)
            {
                yield return new CompletionStreamData
                {
                    TextDelta = string.IsNullOrEmpty(chunk.Content) ? null : chunk.Content,
                    Logprobs = logprobs
                };
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
    /// The candidate count to ask llama-server for, from <c>TopLogprobs</c>; null asks for no log probabilities. The
    /// server computes them only for a count above zero (<c>n_probs</c>, which <c>top_logprobs</c> maps to), so 0 —
    /// the sampled token's own log probability — asks for one candidate, which <see cref="ShapeLogprobs"/> drops.
    /// </summary>
    internal static int? WireTopLogprobs(int? topLogprobs) => topLogprobs switch
    {
        null => null,
        < 0 => throw new ArgumentOutOfRangeException(
            nameof(topLogprobs), topLogprobs, "TopLogprobs must be 0 or greater, or null to request no log probabilities."),
        0 => 1,
        { } count => count,
    };

    /// <summary>
    /// The per-token log probabilities a response carries, as the caller asked for them: null when it carries none,
    /// and with every candidate list emptied when <paramref name="topLogprobs"/> is 0 (see <see cref="WireTopLogprobs"/>).
    /// </summary>
    internal static List<LlamaTokenLogprob>? ShapeLogprobs(List<LlamaTokenLogprob>? content, int? topLogprobs)
    {
        if (content is not { Count: > 0 })
            return null;

        if (topLogprobs == 0)
        {
            foreach (var token in content)
                token.TopLogprobs = [];
        }

        return content;
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
        var topLogprobs = WireTopLogprobs(options.TopLogprobs);
        if (topLogprobs is not null && stream && options.Tools is { Count: > 0 })
            throw new ArgumentException(
                "TopLogprobs cannot be combined with Tools on a streamed chat completion: llama-server rejects a " +
                "streamed request that asks for log probabilities and offers tools. Use a non-streamed completion.");

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
            // Reading a long prompt can take minutes before the first token; progress is the only sign of life meanwhile.
            ReturnProgress = stream ? true : null,
            Stop = options.StopSequences?.ToList(),
            Grammar = options.Grammar,
            ResponseFormat = BuildChatResponseFormat(options.Grammar, options.JsonSchema),
            Tools = options.Tools?.ToList(),
            ToolChoice = BuildToolChoiceNode(options.ToolChoice),
            ChatTemplateKwargs = options.EnableThinking is { } enableThinking
                ? new Dictionary<string, object> { ["enable_thinking"] = enableThinking }
                : null,
            // The server refuses top_logprobs without logprobs: true.
            Logprobs = topLogprobs is not null ? true : null,
            TopLogprobs = topLogprobs
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
