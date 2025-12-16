using LMSupply.Detector.Models;

namespace LMSupply.Detector;

/// <summary>
/// Interface for object detection models.
/// </summary>
public interface IDetectorModel : IAsyncDisposable
{
    /// <summary>
    /// Gets the model identifier.
    /// </summary>
    string ModelId { get; }

    /// <summary>
    /// Detects objects in an image file.
    /// </summary>
    /// <param name="imagePath">Path to the image file.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>List of detected objects sorted by confidence (highest first).</returns>
    Task<IReadOnlyList<DetectionResult>> DetectAsync(
        string imagePath,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Detects objects in an image stream.
    /// </summary>
    /// <param name="imageStream">Stream containing image data.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>List of detected objects sorted by confidence (highest first).</returns>
    Task<IReadOnlyList<DetectionResult>> DetectAsync(
        Stream imageStream,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Detects objects in an image byte array.
    /// </summary>
    /// <param name="imageData">Byte array containing image data.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>List of detected objects sorted by confidence (highest first).</returns>
    Task<IReadOnlyList<DetectionResult>> DetectAsync(
        byte[] imageData,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Detects objects in multiple images.
    /// </summary>
    /// <param name="imagePaths">Paths to image files.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Detection results for each image.</returns>
    Task<IReadOnlyList<IReadOnlyList<DetectionResult>>> DetectBatchAsync(
        IEnumerable<string> imagePaths,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Pre-loads the model to avoid cold start latency on first inference.
    /// </summary>
    /// <param name="cancellationToken">Cancellation token.</param>
    Task WarmupAsync(CancellationToken cancellationToken = default);

    /// <summary>
    /// Gets information about the loaded model.
    /// </summary>
    /// <returns>Model information, or null if not yet loaded.</returns>
    DetectorModelInfo? GetModelInfo();

    /// <summary>
    /// Gets the COCO class labels supported by this model.
    /// </summary>
    IReadOnlyList<string> ClassLabels { get; }
}
