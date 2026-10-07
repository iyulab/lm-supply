namespace LMSupply.Console.Host.Models.Requests;

/// <summary>
/// Embedding request
/// </summary>
public sealed record EmbedRequest
{
    /// <summary>
    /// Model ID
    /// </summary>
    public string ModelId { get; init; } = "default";

    /// <summary>
    /// Text to embed (single)
    /// </summary>
    public string? Text { get; init; }

    /// <summary>
    /// Texts to embed (batch)
    /// </summary>
    public IReadOnlyList<string>? Texts { get; init; }
}

/// <summary>
/// Similarity request
/// </summary>
public sealed record SimilarityRequest
{
    /// <summary>
    /// Model ID
    /// </summary>
    public string ModelId { get; init; } = "default";

    /// <summary>
    /// First text
    /// </summary>
    public required string Text1 { get; init; }

    /// <summary>
    /// Second text
    /// </summary>
    public required string Text2 { get; init; }
}
