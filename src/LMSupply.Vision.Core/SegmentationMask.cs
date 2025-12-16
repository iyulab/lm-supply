namespace LMSupply.Vision;

/// <summary>
/// Represents a segmentation mask for a single class or instance.
/// </summary>
public sealed class SegmentationMask
{
    /// <summary>
    /// Gets the width of the mask.
    /// </summary>
    public int Width { get; }

    /// <summary>
    /// Gets the height of the mask.
    /// </summary>
    public int Height { get; }

    /// <summary>
    /// Gets the class ID this mask represents.
    /// </summary>
    public int ClassId { get; }

    /// <summary>
    /// Gets the class label name (if available).
    /// </summary>
    public string? Label { get; init; }

    /// <summary>
    /// Gets the mask data as a 2D boolean array.
    /// True indicates the pixel belongs to this class/instance.
    /// </summary>
    public bool[,] Mask { get; }

    /// <summary>
    /// Gets the confidence scores for each pixel (optional).
    /// Values are in range [0, 1].
    /// </summary>
    public float[,]? Confidence { get; init; }

    /// <summary>
    /// Gets the bounding box that encloses the mask region.
    /// </summary>
    public BoundingBox BoundingBox { get; }

    /// <summary>
    /// Gets the number of pixels in the mask.
    /// </summary>
    public int PixelCount { get; }

    /// <summary>
    /// Initializes a new instance of the <see cref="SegmentationMask"/> class.
    /// </summary>
    /// <param name="mask">The mask data.</param>
    /// <param name="classId">The class ID.</param>
    public SegmentationMask(bool[,] mask, int classId)
    {
        ArgumentNullException.ThrowIfNull(mask);

        Mask = mask;
        Width = mask.GetLength(1);
        Height = mask.GetLength(0);
        ClassId = classId;
        (BoundingBox, PixelCount) = ComputeBoundingBoxAndCount();
    }

    /// <summary>
    /// Initializes a new instance from a class map for a specific class.
    /// </summary>
    /// <param name="classMap">The class map with class IDs for each pixel.</param>
    /// <param name="classId">The class ID to extract.</param>
    public SegmentationMask(int[,] classMap, int classId)
    {
        ArgumentNullException.ThrowIfNull(classMap);

        Height = classMap.GetLength(0);
        Width = classMap.GetLength(1);
        ClassId = classId;
        Mask = new bool[Height, Width];

        for (int y = 0; y < Height; y++)
        {
            for (int x = 0; x < Width; x++)
            {
                Mask[y, x] = classMap[y, x] == classId;
            }
        }

        (BoundingBox, PixelCount) = ComputeBoundingBoxAndCount();
    }

    /// <summary>
    /// Checks if a pixel at the given coordinates is part of this mask.
    /// </summary>
    /// <param name="x">X coordinate.</param>
    /// <param name="y">Y coordinate.</param>
    /// <returns>True if the pixel is part of the mask.</returns>
    public bool Contains(int x, int y)
    {
        if (x < 0 || x >= Width || y < 0 || y >= Height)
            return false;

        return Mask[y, x];
    }

    /// <summary>
    /// Gets the confidence at the given coordinates.
    /// </summary>
    /// <param name="x">X coordinate.</param>
    /// <param name="y">Y coordinate.</param>
    /// <returns>Confidence value or null if not available.</returns>
    public float? GetConfidence(int x, int y)
    {
        if (Confidence == null || x < 0 || x >= Width || y < 0 || y >= Height)
            return null;

        return Confidence[y, x];
    }

    /// <summary>
    /// Gets the percentage of image area covered by this mask.
    /// </summary>
    /// <returns>Coverage percentage (0-100).</returns>
    public float GetCoveragePercent()
    {
        var totalPixels = Width * Height;
        return totalPixels > 0 ? (PixelCount * 100f) / totalPixels : 0;
    }

    /// <summary>
    /// Resizes the mask to new dimensions using nearest neighbor interpolation.
    /// </summary>
    /// <param name="newWidth">New width.</param>
    /// <param name="newHeight">New height.</param>
    /// <returns>A new resized mask.</returns>
    public SegmentationMask Resize(int newWidth, int newHeight)
    {
        if (newWidth <= 0 || newHeight <= 0)
            throw new ArgumentException("Dimensions must be positive");

        var newMask = new bool[newHeight, newWidth];
        var scaleX = (float)Width / newWidth;
        var scaleY = (float)Height / newHeight;

        for (int y = 0; y < newHeight; y++)
        {
            for (int x = 0; x < newWidth; x++)
            {
                var srcX = (int)(x * scaleX);
                var srcY = (int)(y * scaleY);
                srcX = Math.Clamp(srcX, 0, Width - 1);
                srcY = Math.Clamp(srcY, 0, Height - 1);
                newMask[y, x] = Mask[srcY, srcX];
            }
        }

        return new SegmentationMask(newMask, ClassId) { Label = Label };
    }

    /// <summary>
    /// Computes Intersection over Union (IoU) with another mask.
    /// </summary>
    /// <param name="other">The other mask.</param>
    /// <returns>IoU value between 0 and 1.</returns>
    public float IoU(SegmentationMask other)
    {
        ArgumentNullException.ThrowIfNull(other);

        if (Width != other.Width || Height != other.Height)
            throw new ArgumentException("Masks must have the same dimensions");

        int intersection = 0;
        int union = 0;

        for (int y = 0; y < Height; y++)
        {
            for (int x = 0; x < Width; x++)
            {
                var a = Mask[y, x];
                var b = other.Mask[y, x];

                if (a && b) intersection++;
                if (a || b) union++;
            }
        }

        return union > 0 ? (float)intersection / union : 0;
    }

    /// <summary>
    /// Converts the mask to a flat byte array (0 or 255 values).
    /// </summary>
    /// <returns>Byte array representing the mask.</returns>
    public byte[] ToByteArray()
    {
        var result = new byte[Width * Height];
        for (int y = 0; y < Height; y++)
        {
            for (int x = 0; x < Width; x++)
            {
                result[y * Width + x] = Mask[y, x] ? (byte)255 : (byte)0;
            }
        }
        return result;
    }

    /// <summary>
    /// Creates a mask from a byte array.
    /// </summary>
    /// <param name="data">Byte array data.</param>
    /// <param name="width">Mask width.</param>
    /// <param name="height">Mask height.</param>
    /// <param name="classId">Class ID.</param>
    /// <param name="threshold">Threshold for considering a pixel as part of mask.</param>
    /// <returns>A new segmentation mask.</returns>
    public static SegmentationMask FromByteArray(byte[] data, int width, int height, int classId, byte threshold = 128)
    {
        ArgumentNullException.ThrowIfNull(data);

        if (data.Length != width * height)
            throw new ArgumentException($"Data length {data.Length} does not match dimensions {width}x{height}");

        var mask = new bool[height, width];
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                mask[y, x] = data[y * width + x] >= threshold;
            }
        }

        return new SegmentationMask(mask, classId);
    }

    private (BoundingBox, int) ComputeBoundingBoxAndCount()
    {
        int minX = Width, minY = Height, maxX = -1, maxY = -1;
        int count = 0;

        for (int y = 0; y < Height; y++)
        {
            for (int x = 0; x < Width; x++)
            {
                if (Mask[y, x])
                {
                    count++;
                    if (x < minX) minX = x;
                    if (x > maxX) maxX = x;
                    if (y < minY) minY = y;
                    if (y > maxY) maxY = y;
                }
            }
        }

        if (count == 0)
        {
            return (new BoundingBox(0, 0, 0, 0), 0);
        }

        return (new BoundingBox(minX, minY, maxX - minX + 1, maxY - minY + 1), count);
    }
}
