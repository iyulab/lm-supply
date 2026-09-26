namespace LMSupply.Generator.Models;

/// <summary>
/// A structured chunk from a streaming chat completion.
/// May contain text content, tool call deltas, and/or a finish reason.
/// </summary>
public sealed record ChatStreamChunk
{
    /// <summary>
    /// Text content delta, or null if this chunk contains only tool calls or reasoning.
    /// </summary>
    public string? Text { get; init; }

    /// <summary>
    /// Reasoning/thinking content delta (b8994+ servers with Gemma 4 and similar models).
    /// Only populated when <see cref="GenerationOptions.ExtractReasoningTokens"/> is true.
    /// </summary>
    public string? ReasoningDelta { get; init; }

    /// <summary>
    /// Tool call deltas in this chunk, or null if no tool calls.
    /// Multiple deltas may arrive across multiple chunks for the same tool call (identified by Index).
    /// </summary>
    public IReadOnlyList<ChatToolCallDelta>? ToolCalls { get; init; }

    /// <summary>
    /// Finish reason, present only on the final chunk.
    /// Values: "stop", "tool_calls", "length".
    /// </summary>
    public string? FinishReason { get; init; }

    /// <summary>
    /// Token accounting reported by the backend, present only on the final chunk (the one carrying
    /// <see cref="FinishReason"/>). <see cref="ChatTokenUsage.CompletionTokens"/> includes reasoning the
    /// stream may not have shown. Null when the backend does not report usage (the ONNX path does not;
    /// llama-server does), or when the stream was cut client-side by the output-token safety limit.
    /// </summary>
    public ChatTokenUsage? Usage { get; init; }
}

/// <summary>
/// A streaming delta for a tool call.
/// Tool calls are accumulated across multiple chunks by matching on <see cref="Index"/>.
/// </summary>
public sealed record ChatToolCallDelta
{
    /// <summary>
    /// Zero-based index identifying which tool call this delta belongs to.
    /// </summary>
    public required int Index { get; init; }

    /// <summary>
    /// Tool call ID (typically only present in the first delta for a given index).
    /// </summary>
    public string? Id { get; init; }

    /// <summary>
    /// Function name (typically only present in the first delta for a given index).
    /// </summary>
    public string? Name { get; init; }

    /// <summary>
    /// Partial JSON arguments string (accumulated across deltas).
    /// </summary>
    public string? Arguments { get; init; }
}
