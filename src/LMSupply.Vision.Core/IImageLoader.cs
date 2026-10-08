namespace LMSupply.Vision;

/// <summary>
/// Abstraction for loading images from various sources.
/// </summary>
public interface IImageLoader
{
    /// <summary>
    /// Loads an image from a file path.
    /// </summary>
    /// <param name="path">Path to the image file.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>The decoded image.</returns>
    Task<RgbImage> LoadAsync(string path, CancellationToken cancellationToken = default);

    /// <summary>
    /// Loads an image from a stream.
    /// </summary>
    /// <param name="stream">Stream containing image data.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>The decoded image.</returns>
    Task<RgbImage> LoadAsync(Stream stream, CancellationToken cancellationToken = default);

    /// <summary>
    /// Loads an image from a byte array.
    /// </summary>
    /// <param name="data">Byte array containing image data.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>The decoded image.</returns>
    Task<RgbImage> LoadAsync(byte[] data, CancellationToken cancellationToken = default);

    /// <summary>
    /// Loads an image from a ReadOnlyMemory buffer.
    /// </summary>
    /// <param name="data">Memory buffer containing image data.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>The decoded image.</returns>
    Task<RgbImage> LoadAsync(ReadOnlyMemory<byte> data, CancellationToken cancellationToken = default);
}
