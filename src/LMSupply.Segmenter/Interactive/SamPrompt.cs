namespace LMSupply.Segmenter.Interactive;

/// <summary>
/// Base class for SAM (Segment Anything Model) prompts.
/// </summary>
public abstract class SamPrompt
{
    /// <summary>
    /// Gets the prompt type.
    /// </summary>
    public abstract SamPromptType Type { get; }
}

/// <summary>
/// SAM prompt types.
/// </summary>
public enum SamPromptType
{
    /// <summary>
    /// Point prompt with x, y coordinates.
    /// </summary>
    Point,

    /// <summary>
    /// Box prompt with bounding box coordinates.
    /// </summary>
    Box,

    /// <summary>
    /// Mask prompt for refinement.
    /// </summary>
    Mask
}

/// <summary>
/// Point label for SAM point prompts.
/// </summary>
public enum PointLabel
{
    /// <summary>
    /// Background point (negative click).
    /// </summary>
    Background = 0,

    /// <summary>
    /// Foreground point (positive click).
    /// </summary>
    Foreground = 1
}

/// <summary>
/// Point prompt for interactive segmentation.
/// </summary>
public sealed class PointPrompt : SamPrompt
{
    /// <inheritdoc />
    public override SamPromptType Type => SamPromptType.Point;

    /// <summary>
    /// Gets the X coordinate (in image pixels).
    /// </summary>
    public float X { get; }

    /// <summary>
    /// Gets the Y coordinate (in image pixels).
    /// </summary>
    public float Y { get; }

    /// <summary>
    /// Gets the point label (foreground or background).
    /// </summary>
    public PointLabel Label { get; }

    /// <summary>
    /// Creates a new point prompt.
    /// </summary>
    /// <param name="x">X coordinate in image pixels.</param>
    /// <param name="y">Y coordinate in image pixels.</param>
    /// <param name="label">Point label (foreground or background).</param>
    public PointPrompt(float x, float y, PointLabel label = PointLabel.Foreground)
    {
        X = x;
        Y = y;
        Label = label;
    }

    /// <summary>
    /// Creates a foreground point prompt.
    /// </summary>
    public static PointPrompt Foreground(float x, float y) => new(x, y, PointLabel.Foreground);

    /// <summary>
    /// Creates a background point prompt.
    /// </summary>
    public static PointPrompt Background(float x, float y) => new(x, y, PointLabel.Background);
}

/// <summary>
/// Box prompt for interactive segmentation.
/// </summary>
public sealed class BoxPrompt : SamPrompt
{
    /// <inheritdoc />
    public override SamPromptType Type => SamPromptType.Box;

    /// <summary>
    /// Gets the X coordinate of the top-left corner.
    /// </summary>
    public float X { get; }

    /// <summary>
    /// Gets the Y coordinate of the top-left corner.
    /// </summary>
    public float Y { get; }

    /// <summary>
    /// Gets the width of the box.
    /// </summary>
    public float Width { get; }

    /// <summary>
    /// Gets the height of the box.
    /// </summary>
    public float Height { get; }

    /// <summary>
    /// Gets the right edge X coordinate.
    /// </summary>
    public float Right => X + Width;

    /// <summary>
    /// Gets the bottom edge Y coordinate.
    /// </summary>
    public float Bottom => Y + Height;

    /// <summary>
    /// Creates a new box prompt.
    /// </summary>
    /// <param name="x">X coordinate of top-left corner.</param>
    /// <param name="y">Y coordinate of top-left corner.</param>
    /// <param name="width">Box width.</param>
    /// <param name="height">Box height.</param>
    public BoxPrompt(float x, float y, float width, float height)
    {
        X = x;
        Y = y;
        Width = width;
        Height = height;
    }

    /// <summary>
    /// Creates a box prompt from corner coordinates.
    /// </summary>
    public static BoxPrompt FromCorners(float x1, float y1, float x2, float y2)
    {
        var minX = MathF.Min(x1, x2);
        var minY = MathF.Min(y1, y2);
        var maxX = MathF.Max(x1, x2);
        var maxY = MathF.Max(y1, y2);
        return new BoxPrompt(minX, minY, maxX - minX, maxY - minY);
    }
}

/// <summary>
/// Mask prompt for refinement (input mask from previous prediction).
/// </summary>
public sealed class MaskPrompt : SamPrompt
{
    /// <inheritdoc />
    public override SamPromptType Type => SamPromptType.Mask;

    /// <summary>
    /// Gets the mask data (2D array where true = part of object).
    /// </summary>
    public bool[,] Mask { get; }

    /// <summary>
    /// Creates a new mask prompt.
    /// </summary>
    /// <param name="mask">The mask data.</param>
    public MaskPrompt(bool[,] mask)
    {
        ArgumentNullException.ThrowIfNull(mask);
        Mask = mask;
    }
}
