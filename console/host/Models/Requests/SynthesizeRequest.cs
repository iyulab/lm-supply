namespace LMSupply.Console.Host.Models.Requests;

/// <summary>
/// TTS request
/// </summary>
public sealed record SynthesizeRequest
{
    /// <summary>
    /// Model ID
    /// </summary>
    public string ModelId { get; init; } = "default";

    /// <summary>
    /// Text to synthesize
    /// </summary>
    public required string Text { get; init; }

    /// <summary>
    /// Speech speed (0.5 to 2.0)
    /// </summary>
    public float Speed { get; init; } = 1.0f;
}
