namespace LMSupply.Vision;

/// <summary>
/// An 8-bit RGB image held in memory: <see cref="Width"/> × <see cref="Height"/> pixels, three bytes per pixel
/// (R, G, B), rows top to bottom with no padding. This is the image type the vision models take; load one with
/// <see cref="ImageLoader"/>.
/// </summary>
public sealed class RgbImage
{
    private readonly byte[] _pixels;

    /// <summary>Creates a black image.</summary>
    public RgbImage(int width, int height)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(width);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(height);
        Width = width;
        Height = height;
        _pixels = new byte[checked(width * height * 3)];
    }

    /// <summary>Creates an image from RGB bytes (copied), row-major, <c>width * height * 3</c> long.</summary>
    public RgbImage(int width, int height, ReadOnlySpan<byte> pixels)
        : this(width, height)
    {
        if (pixels.Length != _pixels.Length)
        {
            throw new ArgumentException($"Expected {_pixels.Length} bytes for a {width}x{height} RGB image, got {pixels.Length}.", nameof(pixels));
        }

        pixels.CopyTo(_pixels);
    }

    private RgbImage(int width, int height, byte[] pixels)
    {
        Width = width;
        Height = height;
        _pixels = pixels;
    }

    /// <summary>Width in pixels.</summary>
    public int Width { get; }

    /// <summary>Height in pixels.</summary>
    public int Height { get; }

    /// <summary>All pixels: R, G, B per pixel, row-major.</summary>
    public Span<byte> Pixels => _pixels;

    /// <summary>One row of pixels: <c>Width * 3</c> bytes.</summary>
    public Span<byte> GetRow(int y)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(y);
        ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(y, Height);
        return _pixels.AsSpan(y * Width * 3, Width * 3);
    }

    /// <summary>Gets or sets one pixel.</summary>
    public (byte R, byte G, byte B) this[int x, int y]
    {
        get
        {
            var row = GetRow(y);
            var at = Column(x);
            return (row[at], row[at + 1], row[at + 2]);
        }

        set
        {
            var row = GetRow(y);
            var at = Column(x);
            row[at] = value.R;
            row[at + 1] = value.G;
            row[at + 2] = value.B;
        }
    }

    /// <summary>Creates an image filled with one color.</summary>
    public static RgbImage Filled(int width, int height, byte r, byte g, byte b)
    {
        var image = new RgbImage(width, height);
        var pixels = image._pixels;
        for (var i = 0; i < pixels.Length; i += 3)
        {
            pixels[i] = r;
            pixels[i + 1] = g;
            pixels[i + 2] = b;
        }

        return image;
    }

    /// <summary>A copy of this image.</summary>
    public RgbImage Clone() => new(Width, Height, (byte[])_pixels.Clone());

    /// <summary>
    /// A copy scaled to <paramref name="width"/> × <paramref name="height"/> (aspect ratio not preserved) with bicubic
    /// (Catmull-Rom) resampling. When shrinking, the filter widens with the scale factor, so every source pixel
    /// contributes and fine detail does not alias — the same convolution the common Python image pipelines apply.
    /// </summary>
    public RgbImage Resize(int width, int height)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(width);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(height);
        return width == Width && height == Height
            ? Clone()
            : new RgbImage(width, height, ImageResampler.Resize(_pixels, Width, Height, width, height));
    }

    /// <summary>A copy of the rectangle at (<paramref name="x"/>, <paramref name="y"/>), which must lie inside the image.</summary>
    public RgbImage Crop(int x, int y, int width, int height)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(x);
        ArgumentOutOfRangeException.ThrowIfNegative(y);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(width);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(height);
        if (x + width > Width || y + height > Height)
        {
            throw new ArgumentOutOfRangeException(nameof(width), $"Crop {x},{y} {width}x{height} exceeds the {Width}x{Height} image.");
        }

        var cropped = new byte[width * height * 3];
        for (var row = 0; row < height; row++)
        {
            _pixels.AsSpan(((y + row) * Width + x) * 3, width * 3).CopyTo(cropped.AsSpan(row * width * 3));
        }

        return new RgbImage(width, height, cropped);
    }

    /// <summary>
    /// This image scaled to fit inside <paramref name="width"/> × <paramref name="height"/> with its aspect ratio kept,
    /// centered on black (letterbox).
    /// </summary>
    public RgbImage Letterbox(int width, int height)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(width);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(height);
        var scaleHeight = height / (float)Height;
        var scaleWidth = width / (float)Width;
        int contentWidth = width, contentHeight = height;
        if (scaleHeight < scaleWidth)
        {
            contentWidth = Math.Max(1, (int)MathF.Round(Width * scaleHeight));
        }
        else
        {
            contentHeight = Math.Max(1, (int)MathF.Round(Height * scaleWidth));
        }

        var content = Resize(contentWidth, contentHeight);
        var canvas = new RgbImage(width, height);
        var left = (width - contentWidth) / 2;
        var top = (height - contentHeight) / 2;
        for (var row = 0; row < contentHeight; row++)
        {
            content.GetRow(row).CopyTo(canvas._pixels.AsSpan(((top + row) * width + left) * 3));
        }

        return canvas;
    }

    private int Column(int x)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(x);
        ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(x, Width);
        return x * 3;
    }
}
