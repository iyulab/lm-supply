using SkiaSharp;

namespace LMSupply.Vision;

/// <summary>
/// Default implementation of <see cref="IImageLoader"/>. Decodes JPEG, PNG, WebP, GIF (first frame), BMP and ICO. Pixels are taken as stored: no EXIF rotation and no color profile
/// conversion; transparency is dropped (the color channels are kept as they are, not blended).
/// </summary>
public sealed class ImageLoader : IImageLoader
{
    /// <summary>
    /// Shared singleton instance.
    /// </summary>
    public static ImageLoader Instance { get; } = new();

    /// <inheritdoc />
    public async Task<RgbImage> LoadAsync(string path, CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        if (!File.Exists(path))
        {
            throw new FileNotFoundException($"Image file not found: {path}", path);
        }

        var data = await File.ReadAllBytesAsync(path, cancellationToken).ConfigureAwait(false);
        return Decode(data);
    }

    /// <inheritdoc />
    public async Task<RgbImage> LoadAsync(Stream stream, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(stream);

        using var buffer = new MemoryStream();
        await stream.CopyToAsync(buffer, cancellationToken).ConfigureAwait(false);
        return Decode(buffer.GetBuffer().AsSpan(0, (int)buffer.Length));
    }

    /// <inheritdoc />
    public Task<RgbImage> LoadAsync(byte[] data, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(data);
        cancellationToken.ThrowIfCancellationRequested();
        return Task.FromResult(Decode(data));
    }

    /// <inheritdoc />
    public Task<RgbImage> LoadAsync(ReadOnlyMemory<byte> data, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        return Task.FromResult(Decode(data.Span));
    }

    /// <summary>Decodes an encoded image (see the class remarks for formats).</summary>
    /// <exception cref="InvalidDataException">The data is not an image in a supported format, or is truncated or corrupt.</exception>
    public static RgbImage Decode(ReadOnlySpan<byte> data)
    {
        using var skData = SKData.CreateCopy(data);
        using var codec = SKCodec.Create(skData)
            ?? throw new InvalidDataException("The data is not an image in a supported format (JPEG, PNG, WebP, GIF, BMP, ICO).");

        var width = codec.Info.Width;
        var height = codec.Info.Height;
        var info = new SKImageInfo(width, height, SKColorType.Rgba8888, SKAlphaType.Unpremul);
        var rgba = new byte[info.BytesSize];
        SKCodecResult result;
        unsafe
        {
            fixed (byte* pixels = rgba)
            {
                result = codec.GetPixels(info, (IntPtr)pixels);
            }
        }

        if (result != SKCodecResult.Success)
        {
            throw new InvalidDataException($"The image could not be decoded ({result}).");
        }

        var image = new RgbImage(width, height);
        var rgb = image.Pixels;
        for (int source = 0, target = 0; target < rgb.Length; source += 4, target += 3)
        {
            rgb[target] = rgba[source];
            rgb[target + 1] = rgba[source + 1];
            rgb[target + 2] = rgba[source + 2];
        }

        return image;
    }

    /// <summary>Reads an encoded image's dimensions without decoding its pixels; null when the data is not a supported image.</summary>
    public static (int Width, int Height)? Identify(ReadOnlySpan<byte> data)
    {
        using var skData = SKData.CreateCopy(data);
        using var codec = SKCodec.Create(skData);
        return codec is null ? null : (codec.Info.Width, codec.Info.Height);
    }

    /// <summary>Encodes an image as PNG.</summary>
    public static byte[] EncodePng(RgbImage image)
    {
        ArgumentNullException.ThrowIfNull(image);

        var info = new SKImageInfo(image.Width, image.Height, SKColorType.Rgba8888, SKAlphaType.Opaque);
        using var bitmap = new SKBitmap(info);
        var rgba = bitmap.GetPixelSpan();
        var rgb = image.Pixels;
        for (int source = 0, target = 0; source < rgb.Length; source += 3, target += 4)
        {
            rgba[target] = rgb[source];
            rgba[target + 1] = rgb[source + 1];
            rgba[target + 2] = rgb[source + 2];
            rgba[target + 3] = 255;
        }

        using var encoded = bitmap.Encode(SKEncodedImageFormat.Png, 100)
            ?? throw new InvalidOperationException("PNG encoding failed.");
        return encoded.ToArray();
    }
}
