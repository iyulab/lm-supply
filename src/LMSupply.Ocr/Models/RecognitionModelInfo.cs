namespace LMSupply.Ocr.Models;

/// <summary>
/// Metadata about a text recognition model (CRNN).
/// </summary>
/// <param name="RepoId">HuggingFace repository ID.</param>
/// <param name="Alias">Short alias for the model.</param>
/// <param name="DisplayName">Human-readable display name.</param>
/// <param name="ModelFile">ONNX model file name.</param>
/// <param name="DictFile">Character dictionary file name.</param>
/// <param name="LanguageCodes">ISO language codes supported by this model.</param>
public record RecognitionModelInfo(
    string RepoId,
    string Alias,
    string DisplayName,
    string ModelFile,
    string DictFile,
    IReadOnlyList<string> LanguageCodes)
{
    /// <summary>
    /// Optional subfolder within the HuggingFace repository.
    /// </summary>
    public string? Subfolder { get; init; }

    /// <summary>
    /// Expected input image height for recognition.
    /// Width is dynamic based on text length.
    /// Default is 48 pixels (PaddleOCR v3 standard).
    /// </summary>
    public int InputHeight { get; init; } = 48;

    /// <summary>
    /// Maximum input width ratio relative to height.
    /// Default is 25 (max width = 48 * 25 = 1200 pixels).
    /// </summary>
    public int MaxWidthRatio { get; init; } = 25;

    /// <summary>
    /// Mean values for input normalization.
    /// Default uses 0.5 normalization.
    /// </summary>
    public float[] Mean { get; init; } = [0.5f, 0.5f, 0.5f];

    /// <summary>
    /// Standard deviation values for input normalization.
    /// Default uses 0.5 normalization.
    /// </summary>
    public float[] Std { get; init; } = [0.5f, 0.5f, 0.5f];

    /// <summary>
    /// Whether to use space character in recognition.
    /// </summary>
    public bool UseSpace { get; init; } = true;
}
