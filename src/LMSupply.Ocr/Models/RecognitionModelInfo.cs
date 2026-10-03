namespace LMSupply.Ocr.Models;

/// <summary>
/// Metadata about a text recognition model (CRNN).
/// </summary>
/// <param name="RepoId">HuggingFace repository ID.</param>
/// <param name="AliasName">Short alias name for the model.</param>
/// <param name="DisplayName">Human-readable display name.</param>
/// <param name="ModelFile">ONNX model file name.</param>
/// <param name="DictFile">Character dictionary file name.</param>
/// <param name="LanguageCodes">ISO language codes supported by this model.</param>
public record RecognitionModelInfo(
    string RepoId,
    string AliasName,
    string DisplayName,
    string ModelFile,
    string DictFile,
    IReadOnlyList<string> LanguageCodes) : IModelInfoBase
{
    /// <summary>
    /// Gets the model description (derived from DisplayName).
    /// </summary>
    public string? Description => DisplayName;

    // IModelInfoBase explicit implementation — include subfolder for uniqueness
    // because multiple models share the same RepoId with different subfolders.
    string IModelInfoBase.Id => Subfolder is not null ? $"{RepoId}/{Subfolder}" : RepoId;
    /// <summary>
    /// Optional subfolder within the HuggingFace repository.
    /// </summary>
    public string? Subfolder { get; init; }

    /// <summary>
    /// The licence of the model's weights as the catalog curates it — an SPDX identifier where one exists. For a
    /// conversion repository (an ONNX export of another model) it is the licence of the model it converts, which the
    /// conversion's own card often leaves out. <see langword="null"/> when not curated (a model loaded from its own files).
    /// </summary>
    public string? License { get; init; }

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
