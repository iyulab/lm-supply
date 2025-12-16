namespace LMSupply.Segmenter;

/// <summary>
/// Represents the result of semantic segmentation.
/// </summary>
public sealed class SegmentationResult
{
    /// <summary>
    /// Gets the original image width.
    /// </summary>
    public int Width { get; init; }

    /// <summary>
    /// Gets the original image height.
    /// </summary>
    public int Height { get; init; }

    /// <summary>
    /// Gets the class index for each pixel.
    /// Array size is Width * Height, row-major order (y * Width + x).
    /// </summary>
    public int[] ClassMap { get; init; } = [];

    /// <summary>
    /// Gets the confidence scores for the predicted class at each pixel.
    /// Array size is Width * Height, row-major order.
    /// </summary>
    public float[] ConfidenceMap { get; init; } = [];

    /// <summary>
    /// Gets the number of unique classes found in this segmentation.
    /// </summary>
    public int UniqueClassCount => ClassMap.Distinct().Count();

    /// <summary>
    /// Gets the class index at a specific pixel location.
    /// </summary>
    /// <param name="x">X coordinate (column).</param>
    /// <param name="y">Y coordinate (row).</param>
    /// <returns>The class index at the specified location.</returns>
    public int GetClassAt(int x, int y)
    {
        ValidateCoordinates(x, y);
        return ClassMap[y * Width + x];
    }

    /// <summary>
    /// Gets the confidence score at a specific pixel location.
    /// </summary>
    /// <param name="x">X coordinate (column).</param>
    /// <param name="y">Y coordinate (row).</param>
    /// <returns>The confidence score at the specified location.</returns>
    public float GetConfidenceAt(int x, int y)
    {
        ValidateCoordinates(x, y);
        return ConfidenceMap[y * Width + x];
    }

    /// <summary>
    /// Gets a binary mask for a specific class.
    /// </summary>
    /// <param name="classId">The class index to extract.</param>
    /// <returns>A boolean array where true indicates the class is present.</returns>
    public bool[] GetClassMask(int classId)
    {
        var mask = new bool[Width * Height];
        for (int i = 0; i < ClassMap.Length; i++)
        {
            mask[i] = ClassMap[i] == classId;
        }
        return mask;
    }

    /// <summary>
    /// Gets the pixel count for each class.
    /// </summary>
    /// <returns>Dictionary mapping class ID to pixel count.</returns>
    public Dictionary<int, int> GetClassPixelCounts()
    {
        var counts = new Dictionary<int, int>();
        foreach (var classId in ClassMap)
        {
            counts.TryGetValue(classId, out int count);
            counts[classId] = count + 1;
        }
        return counts;
    }

    /// <summary>
    /// Gets the percentage of image covered by each class.
    /// </summary>
    /// <returns>Dictionary mapping class ID to coverage percentage (0-100).</returns>
    public Dictionary<int, float> GetClassCoveragePercentages()
    {
        var totalPixels = (float)(Width * Height);
        var counts = GetClassPixelCounts();
        return counts.ToDictionary(
            kvp => kvp.Key,
            kvp => kvp.Value / totalPixels * 100f);
    }

    private void ValidateCoordinates(int x, int y)
    {
        if (x < 0 || x >= Width)
            throw new ArgumentOutOfRangeException(nameof(x), $"X must be between 0 and {Width - 1}");
        if (y < 0 || y >= Height)
            throw new ArgumentOutOfRangeException(nameof(y), $"Y must be between 0 and {Height - 1}");
    }
}
