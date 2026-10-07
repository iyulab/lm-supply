namespace LMSupply.Console.Host.Models.Requests;

/// <summary>
/// Translation request
/// </summary>
public sealed record TranslateRequest
{
    /// <summary>
    /// Model ID (e.g. "default", "ko-en", "ja-en")
    /// </summary>
    public string ModelId { get; init; } = "default";

    /// <summary>
    /// Text to translate (single)
    /// </summary>
    public string? Text { get; init; }

    /// <summary>
    /// Texts to translate (batch)
    /// </summary>
    public IReadOnlyList<string>? Texts { get; init; }
}
