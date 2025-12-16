using LMSupply.Vision;

namespace LMSupply.Segmenter.Interactive;

/// <summary>
/// Result of interactive segmentation using SAM-like models.
/// </summary>
public sealed class InteractiveSegmentationResult
{
    /// <summary>
    /// Gets the segmentation masks (may contain multiple candidates).
    /// Ordered by quality score (best first).
    /// </summary>
    public required IReadOnlyList<SegmentationMask> Masks { get; init; }

    /// <summary>
    /// Gets the IoU (Intersection over Union) scores for each mask.
    /// Higher scores indicate better mask quality.
    /// </summary>
    public required IReadOnlyList<float> IoUScores { get; init; }

    /// <summary>
    /// Gets the original image width.
    /// </summary>
    public int ImageWidth { get; init; }

    /// <summary>
    /// Gets the original image height.
    /// </summary>
    public int ImageHeight { get; init; }

    /// <summary>
    /// Gets the inference time in milliseconds.
    /// </summary>
    public double InferenceTimeMs { get; init; }

    /// <summary>
    /// Gets the best mask (highest IoU score).
    /// </summary>
    public SegmentationMask BestMask => Masks.Count > 0 ? Masks[0] : throw new InvalidOperationException("No masks available");

    /// <summary>
    /// Gets the best IoU score.
    /// </summary>
    public float BestScore => IoUScores.Count > 0 ? IoUScores[0] : 0;
}

/// <summary>
/// An interactive segmentation session that caches the image embedding.
/// </summary>
public interface IInteractiveSession : IDisposable, IAsyncDisposable
{
    /// <summary>
    /// Gets the original image width.
    /// </summary>
    int ImageWidth { get; }

    /// <summary>
    /// Gets the original image height.
    /// </summary>
    int ImageHeight { get; }

    /// <summary>
    /// Gets whether this session is ready for segmentation.
    /// </summary>
    bool IsReady { get; }

    /// <summary>
    /// Performs segmentation with point prompts.
    /// </summary>
    /// <param name="points">Point prompts.</param>
    /// <param name="multimask">If true, returns multiple mask candidates.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Segmentation result with mask(s).</returns>
    Task<InteractiveSegmentationResult> SegmentAsync(
        IEnumerable<PointPrompt> points,
        bool multimask = false,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Performs segmentation with a box prompt.
    /// </summary>
    /// <param name="box">Box prompt defining the region of interest.</param>
    /// <param name="multimask">If true, returns multiple mask candidates.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Segmentation result with mask(s).</returns>
    Task<InteractiveSegmentationResult> SegmentAsync(
        BoxPrompt box,
        bool multimask = false,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Performs segmentation with combined prompts (points and box).
    /// </summary>
    /// <param name="points">Point prompts (can be empty).</param>
    /// <param name="box">Box prompt (can be null).</param>
    /// <param name="multimask">If true, returns multiple mask candidates.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Segmentation result with mask(s).</returns>
    Task<InteractiveSegmentationResult> SegmentAsync(
        IEnumerable<PointPrompt> points,
        BoxPrompt? box,
        bool multimask = false,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Performs segmentation with previous mask for refinement.
    /// </summary>
    /// <param name="points">Point prompts for refinement.</param>
    /// <param name="previousMask">Previous mask to refine.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Refined segmentation result.</returns>
    Task<InteractiveSegmentationResult> RefineAsync(
        IEnumerable<PointPrompt> points,
        MaskPrompt previousMask,
        CancellationToken cancellationToken = default);
}
