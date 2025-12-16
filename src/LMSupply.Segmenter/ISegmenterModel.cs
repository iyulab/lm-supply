using LMSupply.Segmenter.Models;

namespace LMSupply.Segmenter;

/// <summary>
/// Interface for image segmentation models.
/// </summary>
public interface ISegmenterModel : IAsyncDisposable
{
    /// <summary>
    /// Gets the model identifier.
    /// </summary>
    string ModelId { get; }

    /// <summary>
    /// Gets the class labels supported by this model.
    /// </summary>
    IReadOnlyList<string> ClassLabels { get; }

    /// <summary>
    /// Warms up the model by running a dummy inference.
    /// </summary>
    /// <param name="cancellationToken">Cancellation token.</param>
    Task WarmupAsync(CancellationToken cancellationToken = default);

    /// <summary>
    /// Gets model information if available.
    /// </summary>
    /// <returns>Model information or null.</returns>
    SegmenterModelInfo? GetModelInfo();

    /// <summary>
    /// Performs semantic segmentation on an image file.
    /// </summary>
    /// <param name="imagePath">Path to the image file.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Segmentation result with class map and confidence scores.</returns>
    Task<SegmentationResult> SegmentAsync(
        string imagePath,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Performs semantic segmentation on an image stream.
    /// </summary>
    /// <param name="imageStream">Stream containing the image data.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Segmentation result with class map and confidence scores.</returns>
    Task<SegmentationResult> SegmentAsync(
        Stream imageStream,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Performs semantic segmentation on image bytes.
    /// </summary>
    /// <param name="imageData">Byte array containing the image data.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Segmentation result with class map and confidence scores.</returns>
    Task<SegmentationResult> SegmentAsync(
        byte[] imageData,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Performs batch semantic segmentation on multiple images.
    /// </summary>
    /// <param name="imagePaths">Paths to image files.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>List of segmentation results.</returns>
    Task<IReadOnlyList<SegmentationResult>> SegmentBatchAsync(
        IEnumerable<string> imagePaths,
        CancellationToken cancellationToken = default);
}
