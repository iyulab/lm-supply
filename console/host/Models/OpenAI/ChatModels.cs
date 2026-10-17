using System.Text.Json;
using System.Text.Json.Serialization;

namespace LMSupply.Console.Host.Models.OpenAI;

/// <summary>
/// OpenAI /v1/chat/completions request
/// </summary>
public sealed record ChatCompletionRequest
{
    /// <summary>
    /// Model ID (e.g., "microsoft/Phi-4-mini-instruct-onnx" or "default")
    /// </summary>
    public string Model { get; init; } = "default";

    /// <summary>
    /// Messages in the conversation
    /// </summary>
    public required IReadOnlyList<ChatCompletionMessage> Messages { get; init; }

    /// <summary>
    /// Sampling temperature (0-2)
    /// </summary>
    public float? Temperature { get; init; }

    /// <summary>
    /// Nucleus sampling (0-1)
    /// </summary>
    [JsonPropertyName("top_p")]
    public float? TopP { get; init; }

    /// <summary>
    /// Maximum tokens to generate
    /// </summary>
    [JsonPropertyName("max_tokens")]
    public int? MaxTokens { get; init; }

    /// <summary>
    /// Whether to stream the response
    /// </summary>
    public bool Stream { get; init; }

    /// <summary>
    /// Stop sequences
    /// </summary>
    public IReadOnlyList<string>? Stop { get; init; }

    /// <summary>
    /// Frequency penalty (-2 to 2)
    /// </summary>
    [JsonPropertyName("frequency_penalty")]
    public float? FrequencyPenalty { get; init; }

    /// <summary>
    /// Presence penalty (-2 to 2)
    /// </summary>
    [JsonPropertyName("presence_penalty")]
    public float? PresencePenalty { get; init; }

    /// <summary>
    /// User identifier for abuse detection
    /// </summary>
    public string? User { get; init; }

    /// <summary>
    /// Tools available for the model to call
    /// </summary>
    [JsonPropertyName("tools")]
    public IReadOnlyList<Tool>? Tools { get; init; }

    /// <summary>
    /// Controls which tool is called ("auto", "none", or a specific tool)
    /// </summary>
    [JsonPropertyName("tool_choice")]
    public JsonElement? ToolChoice { get; init; }

    /// <summary>
    /// Format of the response output
    /// </summary>
    [JsonPropertyName("response_format")]
    public ResponseFormat? ResponseFormat { get; init; }

    /// <summary>
    /// Number of completions to generate
    /// </summary>
    public int? N { get; init; }

    /// <summary>
    /// Seed for deterministic sampling
    /// </summary>
    public int? Seed { get; init; }

    /// <summary>
    /// Whether to return log probabilities
    /// </summary>
    public bool? Logprobs { get; init; }

    /// <summary>
    /// Number of top log probabilities to return (requires Logprobs=true)
    /// </summary>
    [JsonPropertyName("top_logprobs")]
    public int? TopLogprobs { get; init; }

    /// <summary>
    /// Maximum completion tokens (alternative to max_tokens)
    /// </summary>
    [JsonPropertyName("max_completion_tokens")]
    public int? MaxCompletionTokens { get; init; }

    /// <summary>
    /// Draft model ID for speculative decoding acceleration (LMSupply extension).
    /// When provided, a smaller draft model pre-generates candidate tokens verified by the main model.
    /// </summary>
    [JsonPropertyName("draft_model")]
    public string? DraftModel { get; init; }
}

/// <summary>
/// Chat message (request form — Content may be string or ContentPart array)
/// </summary>
public sealed record ChatCompletionMessage
{
    /// <summary>
    /// Role: system, user, assistant, tool
    /// </summary>
    public required string Role { get; init; }

    /// <summary>
    /// Message content — either a JSON string or a JSON array of ContentPart objects.
    /// Use ExtractText() or MultimodalHelper to read the actual text.
    /// </summary>
    public JsonElement? Content { get; init; }

    /// <summary>
    /// Optional name for the participant
    /// </summary>
    public string? Name { get; init; }

    /// <summary>
    /// Tool calls made by the assistant (in assistant messages)
    /// </summary>
    [JsonPropertyName("tool_calls")]
    public IReadOnlyList<ToolCall>? ToolCalls { get; init; }

    /// <summary>
    /// Tool call ID being responded to (in tool messages)
    /// </summary>
    [JsonPropertyName("tool_call_id")]
    public string? ToolCallId { get; init; }

    /// <summary>
    /// Extracts the plain text from Content, ignoring image parts.
    /// Returns empty string if Content is null or not a string/array.
    /// </summary>
    public string GetTextContent()
    {
        if (Content is null) return "";
        if (Content.Value.ValueKind == JsonValueKind.String)
            return Content.Value.GetString() ?? "";
        if (Content.Value.ValueKind == JsonValueKind.Array)
        {
            var sb = new System.Text.StringBuilder();
            foreach (var part in Content.Value.EnumerateArray())
            {
                if (part.TryGetProperty("type", out var t) && t.GetString() == "text" &&
                    part.TryGetProperty("text", out var text))
                    sb.Append(text.GetString());
            }
            return sb.ToString();
        }
        return "";
    }
}

/// <summary>
/// Chat message in response form — Content is always a plain string.
/// </summary>
public sealed record ChatCompletionResponseMessage
{
    /// <summary>
    /// Role (always "assistant" in responses)
    /// </summary>
    public required string Role { get; init; }

    /// <summary>
    /// Plain-text response content
    /// </summary>
    public required string? Content { get; init; }

    /// <summary>
    /// Tool calls requested by the model
    /// </summary>
    [JsonPropertyName("tool_calls")]
    public IReadOnlyList<ToolCall>? ToolCalls { get; init; }
}

/// <summary>
/// OpenAI /v1/chat/completions response
/// </summary>
public sealed record ChatCompletionResponse
{
    public required string Id { get; init; }
    [JsonPropertyName("object")]
    public string ObjectType { get; init; } = "chat.completion";
    public long Created { get; init; } = DateTimeOffset.UtcNow.ToUnixTimeSeconds();
    public required string Model { get; init; }
    public required IReadOnlyList<ChatCompletionChoice> Choices { get; init; }
    public Usage? Usage { get; init; }
    [JsonPropertyName("system_fingerprint")]
    public string? SystemFingerprint { get; init; }
}

/// <summary>
/// Chat completion choice
/// </summary>
public sealed record ChatCompletionChoice
{
    public int Index { get; init; }
    public required ChatCompletionResponseMessage Message { get; init; }
    [JsonPropertyName("finish_reason")]
    public string? FinishReason { get; init; }
    /// <summary>The log probabilities of the generated tokens, when the request set <c>logprobs</c>; null otherwise.</summary>
    public ChatCompletionLogprobs? Logprobs { get; init; }
}

/// <summary>
/// The log probabilities of a choice's tokens (OpenAI <c>logprobs</c>).
/// </summary>
public sealed record ChatCompletionLogprobs
{
    public required IReadOnlyList<ChatCompletionTokenLogprob> Content { get; init; }

    internal static ChatCompletionLogprobs? From(IReadOnlyList<LMSupply.Generator.Models.TokenLogprob>? logprobs) =>
        logprobs is null ? null : new ChatCompletionLogprobs
        {
            Content = [.. logprobs.Select(t => new ChatCompletionTokenLogprob
            {
                Token = t.Token,
                Logprob = t.Logprob,
                Bytes = t.Bytes is null ? null : [.. t.Bytes.Select(b => (int)b)],
                TopLogprobs = [.. t.TopLogprobs.Select(c => new ChatCompletionTopLogprob
                {
                    Token = c.Token,
                    Logprob = c.Logprob,
                    Bytes = c.Bytes is null ? null : [.. c.Bytes.Select(b => (int)b)],
                })],
            })],
        };
}

/// <summary>
/// One generated token and its log probability, with the most likely candidates at its position.
/// </summary>
public sealed record ChatCompletionTokenLogprob
{
    public required string Token { get; init; }
    public double Logprob { get; init; }
    public IReadOnlyList<int>? Bytes { get; init; }
    [JsonPropertyName("top_logprobs")]
    public required IReadOnlyList<ChatCompletionTopLogprob> TopLogprobs { get; init; }
}

/// <summary>
/// A candidate token at a position and its log probability.
/// </summary>
public sealed record ChatCompletionTopLogprob
{
    public required string Token { get; init; }
    public double Logprob { get; init; }
    public IReadOnlyList<int>? Bytes { get; init; }
}

/// <summary>
/// Streaming chunk response
/// </summary>
public sealed record ChatCompletionChunk
{
    public required string Id { get; init; }
    [JsonPropertyName("object")]
    public string ObjectType { get; init; } = "chat.completion.chunk";
    public long Created { get; init; } = DateTimeOffset.UtcNow.ToUnixTimeSeconds();
    public required string Model { get; init; }
    public required IReadOnlyList<ChatCompletionChunkChoice> Choices { get; init; }
}

/// <summary>
/// Streaming chunk choice
/// </summary>
public sealed record ChatCompletionChunkChoice
{
    public int Index { get; init; }
    public required ChatCompletionDelta Delta { get; init; }
    [JsonPropertyName("finish_reason")]
    public string? FinishReason { get; init; }
    /// <summary>The log probabilities of the tokens in this chunk, when the request set <c>logprobs</c>.</summary>
    public ChatCompletionLogprobs? Logprobs { get; init; }
}

/// <summary>
/// Streaming delta content
/// </summary>
public sealed record ChatCompletionDelta
{
    public string? Role { get; init; }
    public string? Content { get; init; }
}

// ---------------------------------------------------------------------------
// Tool definition (for requests)
// ---------------------------------------------------------------------------

public sealed record Tool
{
    public string Type { get; init; } = "function";
    public required ToolFunction Function { get; init; }
}

public sealed record ToolFunction
{
    public required string Name { get; init; }
    public string? Description { get; init; }
    [JsonPropertyName("parameters")]
    public JsonElement? Parameters { get; init; }
}

// ---------------------------------------------------------------------------
// Tool call (in responses)
// ---------------------------------------------------------------------------

public sealed record ToolCall
{
    public required string Id { get; init; }
    public string Type { get; init; } = "function";
    public required ToolCallFunction Function { get; init; }
}

public sealed record ToolCallFunction
{
    public required string Name { get; init; }
    public required string Arguments { get; init; }
}

// ---------------------------------------------------------------------------
// response_format
// ---------------------------------------------------------------------------

public sealed record ResponseFormat
{
    public required string Type { get; init; }  // "text", "json_object", "json_schema"
    [JsonPropertyName("json_schema")]
    public JsonSchemaFormat? JsonSchema { get; init; }
}

public sealed record JsonSchemaFormat
{
    public required string Name { get; init; }
    public JsonElement? Schema { get; init; }
    public bool Strict { get; init; }
}

// ---------------------------------------------------------------------------
// Content parts for multimodal messages
// ---------------------------------------------------------------------------

public sealed record ContentPart
{
    public required string Type { get; init; }  // "text" or "image_url"
    public string? Text { get; init; }
    [JsonPropertyName("image_url")]
    public ImageUrlPart? ImageUrl { get; init; }
}

public sealed record ImageUrlPart
{
    public required string Url { get; init; }
    public string Detail { get; init; } = "auto";
}
