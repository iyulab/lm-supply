using LMSupply.Ocr.Models;

namespace LMSupply.Ocr;

/// <summary>
/// Interface for optical character recognition (OCR) models.
/// </summary>
public interface IOcr : IAsyncDisposable
{
    /// <summary>
    /// Gets the detection model identifier.
    /// </summary>
    string DetectionModelId { get; }

    /// <summary>
    /// Gets the recognition model identifier.
    /// </summary>
    string RecognitionModelId { get; }

    /// <summary>
    /// Gets the supported language codes for recognition.
    /// </summary>
    IReadOnlyList<string> SupportedLanguages { get; }

    /// <summary>
    /// Pre-loads the models to avoid cold start latency.
    /// </summary>
    /// <param name="cancellationToken">Cancellation token.</param>
    Task WarmupAsync(CancellationToken cancellationToken = default);

    /// <summary>
    /// Gets information about the loaded OCR pipeline.
    /// </summary>
    /// <returns>Combined model information or null if not available.</returns>
    OcrModelInfo? GetModelInfo();

    /// <summary>
    /// Performs OCR on an image from a file path.
    /// </summary>
    /// <param name="imagePath">Path to the image file.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>OCR result containing detected text regions.</returns>
    Task<OcrResult> RecognizeAsync(string imagePath, CancellationToken cancellationToken = default);

    /// <summary>
    /// Performs OCR on an image from a stream.
    /// </summary>
    /// <param name="imageStream">Stream containing image data.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>OCR result containing detected text regions.</returns>
    Task<OcrResult> RecognizeAsync(Stream imageStream, CancellationToken cancellationToken = default);

    /// <summary>
    /// Performs OCR on an image from a byte array.
    /// </summary>
    /// <param name="imageData">Byte array containing image data.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>OCR result containing detected text regions.</returns>
    Task<OcrResult> RecognizeAsync(byte[] imageData, CancellationToken cancellationToken = default);

    /// <summary>
    /// Performs text detection only (without recognition).
    /// </summary>
    /// <param name="imagePath">Path to the image file.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>List of detected text regions with bounding boxes.</returns>
    Task<IReadOnlyList<DetectedRegion>> DetectAsync(string imagePath, CancellationToken cancellationToken = default);

    /// <summary>
    /// Performs text detection only from a stream.
    /// </summary>
    /// <param name="imageStream">Stream containing image data.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>List of detected text regions with bounding boxes.</returns>
    Task<IReadOnlyList<DetectedRegion>> DetectAsync(Stream imageStream, CancellationToken cancellationToken = default);
}
