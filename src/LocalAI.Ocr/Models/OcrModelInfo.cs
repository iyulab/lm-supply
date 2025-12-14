namespace LocalAI.Ocr.Models;

/// <summary>
/// Combined metadata about an OCR pipeline (detection + recognition).
/// </summary>
/// <param name="DetectionModel">Information about the text detection model.</param>
/// <param name="RecognitionModel">Information about the text recognition model.</param>
public record OcrModelInfo(
    DetectionModelInfo DetectionModel,
    RecognitionModelInfo RecognitionModel)
{
    /// <summary>
    /// Gets a combined model identifier.
    /// </summary>
    public string ModelId => $"{DetectionModel.Alias}+{RecognitionModel.Alias}";

    /// <summary>
    /// Gets the supported language codes from the recognition model.
    /// </summary>
    public IReadOnlyList<string> SupportedLanguages => RecognitionModel.LanguageCodes;
}
