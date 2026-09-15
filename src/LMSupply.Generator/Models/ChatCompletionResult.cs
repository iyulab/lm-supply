namespace LMSupply.Generator.Models;

/// <summary>
/// Result of a chat completion that may contain text and/or tool calls.
/// </summary>
public sealed record ChatCompletionResult
{
    /// <summary>
    /// Text content of the response (may be null if only tool calls).
    /// </summary>
    public string? Content { get; init; }

    /// <summary>
    /// Tool calls made by the model (may be null if only text).
    /// </summary>
    public IReadOnlyList<ChatToolCall>? ToolCalls { get; init; }

    /// <summary>
    /// Whether the model made tool calls.
    /// </summary>
    public bool HasToolCalls => ToolCalls is { Count: > 0 };

    /// <summary>
    /// Finish reason from the model.
    /// </summary>
    public string? FinishReason { get; init; }

    /// <summary>
    /// Reasoning ("thinking") content the model produced before the answer, when the backend
    /// separates it from <see cref="Content"/> (llama-server b8994+ <c>reasoning_content</c>; Gemma 4,
    /// Qwen3 and similar). Null when the model did not reason or the backend does not separate it.
    /// A consumer that sees an empty <see cref="Content"/> with <c>FinishReason == "length"</c> and a
    /// non-empty value here knows the token budget went to reasoning — see
    /// <see cref="GenerationOptions.Thinking"/>.
    /// </summary>
    public string? Reasoning { get; init; }

    /// <summary>
    /// Token accounting reported by the backend for this completion, or null when the backend does
    /// not report it (the ONNX path does not; llama-server does).
    /// </summary>
    public ChatTokenUsage? Usage { get; init; }
}

/// <summary>
/// Token accounting for one chat completion, as reported by the backend.
/// </summary>
public sealed record ChatTokenUsage
{
    /// <summary>Tokens in the prompt (all input messages after formatting).</summary>
    public int PromptTokens { get; init; }

    /// <summary>Tokens generated, reasoning included.</summary>
    public int CompletionTokens { get; init; }

    /// <summary>Prompt plus completion tokens.</summary>
    public int TotalTokens { get; init; }
}
