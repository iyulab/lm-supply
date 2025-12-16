using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace LMSupply.Vision;

/// <summary>
/// Default implementation of <see cref="IImageLoader"/> using ImageSharp.
/// </summary>
public sealed class ImageLoader : IImageLoader
{
    /// <summary>
    /// Shared singleton instance.
    /// </summary>
    public static ImageLoader Instance { get; } = new();

    /// <inheritdoc />
    public async Task<Image<Rgb24>> LoadAsync(string path, CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        if (!File.Exists(path))
        {
            throw new FileNotFoundException($"Image file not found: {path}", path);
        }

        return await Image.LoadAsync<Rgb24>(path, cancellationToken).ConfigureAwait(false);
    }

    /// <inheritdoc />
    public async Task<Image<Rgb24>> LoadAsync(Stream stream, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(stream);

        return await Image.LoadAsync<Rgb24>(stream, cancellationToken).ConfigureAwait(false);
    }

    /// <inheritdoc />
    public async Task<Image<Rgb24>> LoadAsync(byte[] data, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(data);

        using var stream = new MemoryStream(data);
        return await LoadAsync(stream, cancellationToken).ConfigureAwait(false);
    }

    /// <inheritdoc />
    public async Task<Image<Rgb24>> LoadAsync(ReadOnlyMemory<byte> data, CancellationToken cancellationToken = default)
    {
        using var stream = new MemoryStream(data.ToArray());
        return await LoadAsync(stream, cancellationToken).ConfigureAwait(false);
    }
}
