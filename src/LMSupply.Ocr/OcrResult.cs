using LMSupply.Vision;

namespace LMSupply.Ocr;

/// <summary>
/// Result of an OCR operation.
/// </summary>
/// <param name="Regions">List of detected and recognized text regions.</param>
/// <param name="ProcessingTimeMs">Total processing time in milliseconds.</param>
public record OcrResult(
    IReadOnlyList<TextRegion> Regions,
    double ProcessingTimeMs)
{
    /// <summary>
    /// Gets the full text by concatenating all regions with newlines.
    /// </summary>
    public string FullText => string.Join(Environment.NewLine,
        Regions.OrderBy(r => r.BoundingBox.Y).ThenBy(r => r.BoundingBox.X).Select(r => r.Text));

    /// <summary>
    /// Gets the full text with regions on the same line joined with spaces.
    /// </summary>
    public string GetTextWithLayout(int lineTolerancePixels = 10)
    {
        if (Regions.Count == 0) return string.Empty;

        var lines = new List<List<TextRegion>>();
        var sortedRegions = Regions.OrderBy(r => r.BoundingBox.Y).ThenBy(r => r.BoundingBox.X).ToList();

        foreach (var region in sortedRegions)
        {
            var existingLine = lines.FirstOrDefault(line =>
                Math.Abs(line[0].BoundingBox.Y - region.BoundingBox.Y) < lineTolerancePixels);

            if (existingLine != null)
            {
                existingLine.Add(region);
            }
            else
            {
                lines.Add([region]);
            }
        }

        return string.Join(Environment.NewLine,
            lines.Select(line => string.Join(" ",
                line.OrderBy(r => r.BoundingBox.X).Select(r => r.Text))));
    }
}

/// <summary>
/// A recognized text region with its location and content.
/// </summary>
/// <param name="Text">The recognized text content.</param>
/// <param name="Confidence">Recognition confidence score (0-1).</param>
/// <param name="BoundingBox">Rectangular bounding box of the text region.</param>
/// <param name="Polygon">Optional polygon coordinates for more precise boundaries.</param>
public record TextRegion(
    string Text,
    float Confidence,
    BoundingBox BoundingBox,
    IReadOnlyList<Point>? Polygon = null);

/// <summary>
/// A detected text region (before recognition).
/// </summary>
/// <param name="BoundingBox">Rectangular bounding box of the detected region.</param>
/// <param name="Confidence">Detection confidence score (0-1).</param>
/// <param name="Polygon">Optional polygon coordinates for more precise boundaries.</param>
public record DetectedRegion(
    BoundingBox BoundingBox,
    float Confidence,
    IReadOnlyList<Point>? Polygon = null);

/// <summary>
/// A 2D point with integer coordinates.
/// </summary>
/// <param name="X">X coordinate.</param>
/// <param name="Y">Y coordinate.</param>
public record Point(int X, int Y);

/// <summary>Geometry helpers for OCR polygons.</summary>
public static class OcrPolygon
{
    /// <summary>The smallest rectangle enclosing <paramref name="polygon"/> (an empty box for an empty polygon).</summary>
    public static BoundingBox Bounds(IReadOnlyList<Point> polygon)
    {
        ArgumentNullException.ThrowIfNull(polygon);
        if (polygon.Count == 0)
            return new BoundingBox(0, 0, 0, 0);

        return BoundingBox.FromCorners(
            polygon.Min(p => p.X), polygon.Min(p => p.Y), polygon.Max(p => p.X), polygon.Max(p => p.Y));
    }
}
