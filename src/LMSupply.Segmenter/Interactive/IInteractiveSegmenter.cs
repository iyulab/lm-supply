using LMSupply.Segmenter.Models;

namespace LMSupply.Segmenter.Interactive;

/// <summary>
/// Interface for interactive segmentation models like SAM, MobileSAM.
/// These models allow prompt-based segmentation with points and boxes.
/// </summary>
public interface IInteractiveSegmenter : IDisposable, IAsyncDisposable
{
    /// <summary>
    /// Gets the model information.
    /// </summary>
    SegmenterModelInfo? GetModelInfo();

    /// <summary>
    /// Warms up the model by running a dummy inference.
    /// </summary>
    /// <param name="cancellationToken">Cancellation token.</param>
    Task WarmupAsync(CancellationToken cancellationToken = default);

    /// <summary>
    /// Creates an interactive session for an image file.
    /// The session caches the image embedding for efficient multiple prompts.
    /// </summary>
    /// <param name="imagePath">Path to the image file.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>An interactive session for the image.</returns>
    Task<IInteractiveSession> CreateSessionAsync(
        string imagePath,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Creates an interactive session for an image stream.
    /// </summary>
    /// <param name="imageStream">Stream containing the image data.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>An interactive session for the image.</returns>
    Task<IInteractiveSession> CreateSessionAsync(
        Stream imageStream,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Creates an interactive session for image bytes.
    /// </summary>
    /// <param name="imageData">Byte array containing the image data.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>An interactive session for the image.</returns>
    Task<IInteractiveSession> CreateSessionAsync(
        byte[] imageData,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Performs one-shot segmentation with point prompts.
    /// For multiple prompts on the same image, use CreateSessionAsync instead.
    /// </summary>
    /// <param name="imagePath">Path to the image file.</param>
    /// <param name="points">Point prompts.</param>
    /// <param name="multimask">If true, returns multiple mask candidates.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Segmentation result with mask(s).</returns>
    Task<InteractiveSegmentationResult> SegmentAsync(
        string imagePath,
        IEnumerable<PointPrompt> points,
        bool multimask = false,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Performs one-shot segmentation with a box prompt.
    /// </summary>
    /// <param name="imagePath">Path to the image file.</param>
    /// <param name="box">Box prompt.</param>
    /// <param name="multimask">If true, returns multiple mask candidates.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Segmentation result with mask(s).</returns>
    Task<InteractiveSegmentationResult> SegmentAsync(
        string imagePath,
        BoxPrompt box,
        bool multimask = false,
        CancellationToken cancellationToken = default);
}
