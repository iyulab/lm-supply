namespace LMSupply.Console.Host.Models.Requests;

/// <summary>
/// Chat request
/// </summary>
public sealed record ChatRequest
{
    /// <summary>
    /// Model ID (e.g. "microsoft/Phi-4-mini-instruct-onnx" or "default")
    /// </summary>
    public string ModelId { get; init; } = "default";

    /// <summary>
    /// Messages
    /// </summary>
    public required IReadOnlyList<ChatMessageDto> Messages { get; init; }

    /// <summary>
    /// Generation options
    /// </summary>
    public ChatOptionsDto? Options { get; init; }
}

/// <summary>
/// Chat message
/// </summary>
public sealed record ChatMessageDto
{
    /// <summary>
    /// Role (system, user, assistant)
    /// </summary>
    public required string Role { get; init; }

    /// <summary>
    /// Content
    /// </summary>
    public required string Content { get; init; }
}

/// <summary>
/// Generation options
/// </summary>
public sealed record ChatOptionsDto
{
    public int MaxTokens { get; init; } = 2048;
    public float Temperature { get; init; } = 0.7f;
    public float TopP { get; init; } = 0.9f;
    public int TopK { get; init; } = 50;
    public float RepetitionPenalty { get; init; } = 1.0f;
    public IReadOnlyList<string>? StopSequences { get; init; }
}
