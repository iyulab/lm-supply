namespace LMSupply.Ocr.Models;

/// <summary>
/// Combined metadata about an OCR pipeline (detection + recognition).
/// </summary>
/// <param name="DetectionModel">Information about the text detection model.</param>
/// <param name="RecognitionModel">Information about the text recognition model.</param>
public record OcrModelInfo(
    DetectionModelInfo DetectionModel,
    RecognitionModelInfo RecognitionModel) : IModelInfoBase
{
    /// <summary>
    /// Gets the combined model identifier.
    /// </summary>
    public string Id => $"{DetectionModel.AliasName}+{RecognitionModel.AliasName}";

    /// <summary>
    /// Gets the alias name for this OCR pipeline configuration.
    /// </summary>
    public string AliasName => Id;

    /// <summary>
    /// Gets the description.
    /// </summary>
    public string? Description => $"OCR pipeline: {DetectionModel.DisplayName} + {RecognitionModel.DisplayName}";

    /// <summary>
    /// The licences of the pipeline's two models: the shared one when they agree, both named otherwise.
    /// <see langword="null"/> when neither is curated.
    /// </summary>
    public string? License =>
        DetectionModel.License == RecognitionModel.License
            ? DetectionModel.License
            : $"{DetectionModel.License ?? "unknown"} (detection); {RecognitionModel.License ?? "unknown"} (recognition)";

    /// <summary>
    /// Gets the supported language codes from the recognition model.
    /// </summary>
    public IReadOnlyList<string> SupportedLanguages => RecognitionModel.LanguageCodes;
}
