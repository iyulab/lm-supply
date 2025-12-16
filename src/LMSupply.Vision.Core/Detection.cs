namespace LMSupply.Vision;

/// <summary>
/// Represents a single detected object in an image.
/// </summary>
/// <param name="BoundingBox">The bounding box of the detection.</param>
/// <param name="ClassId">The class ID of the detected object.</param>
/// <param name="Confidence">The confidence score (0-1).</param>
/// <param name="Label">The human-readable class label (optional).</param>
public readonly record struct Detection(
    BoundingBox BoundingBox,
    int ClassId,
    float Confidence,
    string? Label = null)
{
    /// <summary>
    /// Gets the bounding box X coordinate.
    /// </summary>
    public float X => BoundingBox.X;

    /// <summary>
    /// Gets the bounding box Y coordinate.
    /// </summary>
    public float Y => BoundingBox.Y;

    /// <summary>
    /// Gets the bounding box width.
    /// </summary>
    public float Width => BoundingBox.Width;

    /// <summary>
    /// Gets the bounding box height.
    /// </summary>
    public float Height => BoundingBox.Height;

    /// <summary>
    /// Gets the area of the bounding box.
    /// </summary>
    public float Area => BoundingBox.Area;

    /// <summary>
    /// Calculates IoU with another detection.
    /// </summary>
    /// <param name="other">The other detection.</param>
    /// <returns>IoU value between 0 and 1.</returns>
    public float IoU(Detection other) => BoundingBox.IoU(other.BoundingBox);

    /// <summary>
    /// Creates a new detection with scaled bounding box.
    /// </summary>
    /// <param name="scaleX">X scale factor.</param>
    /// <param name="scaleY">Y scale factor.</param>
    /// <returns>A new scaled detection.</returns>
    public Detection Scale(float scaleX, float scaleY) =>
        this with { BoundingBox = BoundingBox.Scale(scaleX, scaleY) };

    /// <summary>
    /// Creates a new detection with clipped bounding box.
    /// </summary>
    /// <param name="imageWidth">Image width.</param>
    /// <param name="imageHeight">Image height.</param>
    /// <returns>A new clipped detection.</returns>
    public Detection Clip(float imageWidth, float imageHeight) =>
        this with { BoundingBox = BoundingBox.Clip(imageWidth, imageHeight) };

    /// <inheritdoc />
    public override string ToString() =>
        $"Detection({Label ?? $"Class{ClassId}"}: {Confidence:P1} at {BoundingBox})";
}

/// <summary>
/// Extension methods for working with detection collections.
/// </summary>
public static class DetectionExtensions
{
    /// <summary>
    /// Filters detections by confidence threshold.
    /// </summary>
    /// <param name="detections">The detections to filter.</param>
    /// <param name="threshold">Minimum confidence threshold.</param>
    /// <returns>Filtered detections.</returns>
    public static IEnumerable<Detection> FilterByConfidence(
        this IEnumerable<Detection> detections,
        float threshold)
    {
        return detections.Where(d => d.Confidence >= threshold);
    }

    /// <summary>
    /// Filters detections by class ID.
    /// </summary>
    /// <param name="detections">The detections to filter.</param>
    /// <param name="classIds">Class IDs to include.</param>
    /// <returns>Filtered detections.</returns>
    public static IEnumerable<Detection> FilterByClass(
        this IEnumerable<Detection> detections,
        IEnumerable<int> classIds)
    {
        var classSet = classIds.ToHashSet();
        return detections.Where(d => classSet.Contains(d.ClassId));
    }

    /// <summary>
    /// Filters detections by class label.
    /// </summary>
    /// <param name="detections">The detections to filter.</param>
    /// <param name="labels">Labels to include (case-insensitive).</param>
    /// <returns>Filtered detections.</returns>
    public static IEnumerable<Detection> FilterByLabel(
        this IEnumerable<Detection> detections,
        params string[] labels)
    {
        var labelSet = labels.ToHashSet(StringComparer.OrdinalIgnoreCase);
        return detections.Where(d => d.Label != null && labelSet.Contains(d.Label));
    }

    /// <summary>
    /// Applies Non-Maximum Suppression (NMS) to remove overlapping detections.
    /// </summary>
    /// <param name="detections">The detections to process.</param>
    /// <param name="iouThreshold">IoU threshold for suppression.</param>
    /// <returns>Detections after NMS.</returns>
    public static IReadOnlyList<Detection> ApplyNms(
        this IEnumerable<Detection> detections,
        float iouThreshold = 0.5f)
    {
        var sorted = detections
            .OrderByDescending(d => d.Confidence)
            .ToList();

        var result = new List<Detection>();

        while (sorted.Count > 0)
        {
            var best = sorted[0];
            result.Add(best);
            sorted.RemoveAt(0);

            sorted = sorted
                .Where(d => best.IoU(d) < iouThreshold)
                .ToList();
        }

        return result;
    }

    /// <summary>
    /// Applies class-aware NMS (NMS applied within each class separately).
    /// </summary>
    /// <param name="detections">The detections to process.</param>
    /// <param name="iouThreshold">IoU threshold for suppression.</param>
    /// <returns>Detections after class-aware NMS.</returns>
    public static IReadOnlyList<Detection> ApplyClassAwareNms(
        this IEnumerable<Detection> detections,
        float iouThreshold = 0.5f)
    {
        return detections
            .GroupBy(d => d.ClassId)
            .SelectMany(g => g.ApplyNms(iouThreshold))
            .OrderByDescending(d => d.Confidence)
            .ToList();
    }

    /// <summary>
    /// Scales all detections by the given factors.
    /// </summary>
    /// <param name="detections">The detections to scale.</param>
    /// <param name="scaleX">X scale factor.</param>
    /// <param name="scaleY">Y scale factor.</param>
    /// <returns>Scaled detections.</returns>
    public static IEnumerable<Detection> Scale(
        this IEnumerable<Detection> detections,
        float scaleX,
        float scaleY)
    {
        return detections.Select(d => d.Scale(scaleX, scaleY));
    }

    /// <summary>
    /// Clips all detections to image bounds.
    /// </summary>
    /// <param name="detections">The detections to clip.</param>
    /// <param name="imageWidth">Image width.</param>
    /// <param name="imageHeight">Image height.</param>
    /// <returns>Clipped detections.</returns>
    public static IEnumerable<Detection> Clip(
        this IEnumerable<Detection> detections,
        float imageWidth,
        float imageHeight)
    {
        return detections.Select(d => d.Clip(imageWidth, imageHeight));
    }

    /// <summary>
    /// Gets the top N detections by confidence.
    /// </summary>
    /// <param name="detections">The detections.</param>
    /// <param name="count">Number of detections to return.</param>
    /// <returns>Top N detections.</returns>
    public static IReadOnlyList<Detection> TopN(
        this IEnumerable<Detection> detections,
        int count)
    {
        return detections
            .OrderByDescending(d => d.Confidence)
            .Take(count)
            .ToList();
    }
}
