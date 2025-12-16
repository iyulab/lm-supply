namespace LMSupply.Detector;

/// <summary>
/// Represents a single detected object in an image.
/// </summary>
/// <param name="ClassId">The class ID from COCO or custom dataset (0-indexed).</param>
/// <param name="Label">Human-readable class label (e.g., "person", "car").</param>
/// <param name="Confidence">Detection confidence score (0.0 to 1.0).</param>
/// <param name="Box">Bounding box coordinates.</param>
public readonly record struct DetectionResult(
    int ClassId,
    string Label,
    float Confidence,
    BoundingBox Box)
{
    /// <summary>
    /// Returns a string representation of the detection.
    /// </summary>
    public override string ToString() =>
        $"{Label} ({Confidence:P1}) at [{Box.X1:F0}, {Box.Y1:F0}, {Box.X2:F0}, {Box.Y2:F0}]";
}

/// <summary>
/// Represents a bounding box with pixel coordinates.
/// </summary>
/// <param name="X1">Left edge X coordinate.</param>
/// <param name="Y1">Top edge Y coordinate.</param>
/// <param name="X2">Right edge X coordinate.</param>
/// <param name="Y2">Bottom edge Y coordinate.</param>
public readonly record struct BoundingBox(float X1, float Y1, float X2, float Y2)
{
    /// <summary>
    /// Gets the width of the bounding box.
    /// </summary>
    public float Width => X2 - X1;

    /// <summary>
    /// Gets the height of the bounding box.
    /// </summary>
    public float Height => Y2 - Y1;

    /// <summary>
    /// Gets the center X coordinate.
    /// </summary>
    public float CenterX => (X1 + X2) / 2;

    /// <summary>
    /// Gets the center Y coordinate.
    /// </summary>
    public float CenterY => (Y1 + Y2) / 2;

    /// <summary>
    /// Gets the area of the bounding box.
    /// </summary>
    public float Area => Width * Height;

    /// <summary>
    /// Calculates Intersection over Union (IoU) with another box.
    /// </summary>
    /// <param name="other">The other bounding box.</param>
    /// <returns>IoU value between 0 and 1.</returns>
    public float IoU(BoundingBox other)
    {
        var intersectX1 = Math.Max(X1, other.X1);
        var intersectY1 = Math.Max(Y1, other.Y1);
        var intersectX2 = Math.Min(X2, other.X2);
        var intersectY2 = Math.Min(Y2, other.Y2);

        if (intersectX1 >= intersectX2 || intersectY1 >= intersectY2)
            return 0f;

        var intersectionArea = (intersectX2 - intersectX1) * (intersectY2 - intersectY1);
        var unionArea = Area + other.Area - intersectionArea;

        return unionArea > 0 ? intersectionArea / unionArea : 0f;
    }

    /// <summary>
    /// Creates a bounding box from center coordinates and dimensions.
    /// </summary>
    /// <param name="cx">Center X coordinate.</param>
    /// <param name="cy">Center Y coordinate.</param>
    /// <param name="width">Box width.</param>
    /// <param name="height">Box height.</param>
    /// <returns>A new bounding box.</returns>
    public static BoundingBox FromCenterSize(float cx, float cy, float width, float height) =>
        new(cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2);

    /// <summary>
    /// Scales the bounding box to target image dimensions.
    /// </summary>
    /// <param name="scaleX">X scale factor.</param>
    /// <param name="scaleY">Y scale factor.</param>
    /// <returns>Scaled bounding box.</returns>
    public BoundingBox Scale(float scaleX, float scaleY) =>
        new(X1 * scaleX, Y1 * scaleY, X2 * scaleX, Y2 * scaleY);

    /// <summary>
    /// Clamps the bounding box to image boundaries.
    /// </summary>
    /// <param name="imageWidth">Image width.</param>
    /// <param name="imageHeight">Image height.</param>
    /// <returns>Clamped bounding box.</returns>
    public BoundingBox Clamp(int imageWidth, int imageHeight) =>
        new(
            Math.Max(0, Math.Min(X1, imageWidth)),
            Math.Max(0, Math.Min(Y1, imageHeight)),
            Math.Max(0, Math.Min(X2, imageWidth)),
            Math.Max(0, Math.Min(Y2, imageHeight)));
}
