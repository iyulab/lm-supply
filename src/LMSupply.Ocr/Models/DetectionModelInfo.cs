namespace LMSupply.Ocr.Models;

/// <summary>
/// Metadata about a text detection model (DBNet).
/// </summary>
/// <param name="RepoId">HuggingFace repository ID.</param>
/// <param name="Alias">Short alias for the model.</param>
/// <param name="DisplayName">Human-readable display name.</param>
/// <param name="ModelFile">ONNX model file name.</param>
/// <param name="InputWidth">Expected input image width.</param>
/// <param name="InputHeight">Expected input image height.</param>
public record DetectionModelInfo(
    string RepoId,
    string Alias,
    string DisplayName,
    string ModelFile,
    int InputWidth = 960,
    int InputHeight = 960)
{
    /// <summary>
    /// Optional subfolder within the HuggingFace repository.
    /// </summary>
    public string? Subfolder { get; init; }

    /// <summary>
    /// Mean values for input normalization (RGB order).
    /// Default uses ImageNet mean scaled to 0-255 range.
    /// </summary>
    public float[] Mean { get; init; } = [123.675f, 116.28f, 103.53f];

    /// <summary>
    /// Standard deviation values for input normalization (RGB order).
    /// Default uses ImageNet std scaled to 0-255 range.
    /// </summary>
    public float[] Std { get; init; } = [58.395f, 57.12f, 57.375f];

    /// <summary>
    /// Whether the model uses dynamic input shape.
    /// </summary>
    public bool DynamicInput { get; init; } = true;
}
