namespace LMSupply.Vision;

/// <summary>
/// Represents a bounding box for object detection.
/// Coordinates are in pixel values relative to the image.
/// </summary>
/// <param name="X">Left coordinate (x-min).</param>
/// <param name="Y">Top coordinate (y-min).</param>
/// <param name="Width">Box width.</param>
/// <param name="Height">Box height.</param>
public readonly record struct BoundingBox(float X, float Y, float Width, float Height)
{
    /// <summary>
    /// Gets the right coordinate (x-max).
    /// </summary>
    public float Right => X + Width;

    /// <summary>
    /// Gets the bottom coordinate (y-max).
    /// </summary>
    public float Bottom => Y + Height;

    /// <summary>
    /// Gets the center X coordinate.
    /// </summary>
    public float CenterX => X + Width / 2;

    /// <summary>
    /// Gets the center Y coordinate.
    /// </summary>
    public float CenterY => Y + Height / 2;

    /// <summary>
    /// Gets the area of the bounding box.
    /// </summary>
    public float Area => Width * Height;

    /// <summary>
    /// Calculates Intersection over Union (IoU) with another bounding box.
    /// </summary>
    /// <param name="other">The other bounding box.</param>
    /// <returns>IoU value between 0 and 1.</returns>
    public float IoU(BoundingBox other)
    {
        var intersectX = Math.Max(X, other.X);
        var intersectY = Math.Max(Y, other.Y);
        var intersectRight = Math.Min(Right, other.Right);
        var intersectBottom = Math.Min(Bottom, other.Bottom);

        var intersectWidth = Math.Max(0, intersectRight - intersectX);
        var intersectHeight = Math.Max(0, intersectBottom - intersectY);
        var intersectionArea = intersectWidth * intersectHeight;

        var unionArea = Area + other.Area - intersectionArea;

        return unionArea > 0 ? intersectionArea / unionArea : 0;
    }

    /// <summary>
    /// Creates a bounding box from center coordinates.
    /// </summary>
    /// <param name="centerX">Center X coordinate.</param>
    /// <param name="centerY">Center Y coordinate.</param>
    /// <param name="width">Box width.</param>
    /// <param name="height">Box height.</param>
    /// <returns>A new bounding box.</returns>
    public static BoundingBox FromCenter(float centerX, float centerY, float width, float height)
    {
        return new BoundingBox(
            centerX - width / 2,
            centerY - height / 2,
            width,
            height);
    }

    /// <summary>
    /// Creates a bounding box from corner coordinates (x1, y1, x2, y2).
    /// </summary>
    /// <param name="x1">Left coordinate.</param>
    /// <param name="y1">Top coordinate.</param>
    /// <param name="x2">Right coordinate.</param>
    /// <param name="y2">Bottom coordinate.</param>
    /// <returns>A new bounding box.</returns>
    public static BoundingBox FromCorners(float x1, float y1, float x2, float y2)
    {
        return new BoundingBox(x1, y1, x2 - x1, y2 - y1);
    }

    /// <summary>
    /// Scales the bounding box by given factors.
    /// </summary>
    /// <param name="scaleX">X scale factor.</param>
    /// <param name="scaleY">Y scale factor.</param>
    /// <returns>A scaled bounding box.</returns>
    public BoundingBox Scale(float scaleX, float scaleY)
    {
        return new BoundingBox(
            X * scaleX,
            Y * scaleY,
            Width * scaleX,
            Height * scaleY);
    }

    /// <summary>
    /// Clips the bounding box to fit within image bounds.
    /// </summary>
    /// <param name="imageWidth">Image width.</param>
    /// <param name="imageHeight">Image height.</param>
    /// <returns>A clipped bounding box.</returns>
    public BoundingBox Clip(float imageWidth, float imageHeight)
    {
        var clippedX = Math.Max(0, Math.Min(X, imageWidth));
        var clippedY = Math.Max(0, Math.Min(Y, imageHeight));
        var clippedRight = Math.Max(0, Math.Min(Right, imageWidth));
        var clippedBottom = Math.Max(0, Math.Min(Bottom, imageHeight));

        return new BoundingBox(
            clippedX,
            clippedY,
            clippedRight - clippedX,
            clippedBottom - clippedY);
    }

    /// <summary>
    /// Checks if this bounding box contains a point.
    /// </summary>
    /// <param name="x">Point X coordinate.</param>
    /// <param name="y">Point Y coordinate.</param>
    /// <returns>True if the point is inside the box.</returns>
    public bool Contains(float x, float y)
    {
        return x >= X && x <= Right && y >= Y && y <= Bottom;
    }

    /// <summary>
    /// Checks if this bounding box intersects with another.
    /// </summary>
    /// <param name="other">The other bounding box.</param>
    /// <returns>True if the boxes intersect.</returns>
    public bool Intersects(BoundingBox other)
    {
        return X < other.Right && Right > other.X &&
               Y < other.Bottom && Bottom > other.Y;
    }

    /// <inheritdoc />
    public override string ToString() =>
        $"BoundingBox(X={X:F2}, Y={Y:F2}, W={Width:F2}, H={Height:F2})";
}
