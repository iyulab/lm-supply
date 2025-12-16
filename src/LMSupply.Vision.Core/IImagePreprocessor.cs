using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace LMSupply.Vision;

/// <summary>
/// Interface for preprocessing images according to model-specific requirements.
/// </summary>
public interface IImagePreprocessor
{
    /// <summary>
    /// Preprocesses an image from a file path.
    /// </summary>
    /// <param name="imagePath">Path to the image file.</param>
    /// <param name="profile">Preprocessing profile specifying target size and normalization.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Preprocessed image as a float array in NCHW or NHWC format.</returns>
    Task<float[]> PreprocessAsync(
        string imagePath,
        PreprocessProfile profile,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Preprocesses an image from a stream.
    /// </summary>
    /// <param name="imageStream">Stream containing image data.</param>
    /// <param name="profile">Preprocessing profile specifying target size and normalization.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Preprocessed image as a float array in NCHW or NHWC format.</returns>
    Task<float[]> PreprocessAsync(
        Stream imageStream,
        PreprocessProfile profile,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Preprocesses an image from a byte array.
    /// </summary>
    /// <param name="imageData">Byte array containing image data.</param>
    /// <param name="profile">Preprocessing profile specifying target size and normalization.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Preprocessed image as a float array in NCHW or NHWC format.</returns>
    Task<float[]> PreprocessAsync(
        byte[] imageData,
        PreprocessProfile profile,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Preprocesses an already-loaded image.
    /// </summary>
    /// <param name="image">Loaded image in RGB24 format.</param>
    /// <param name="profile">Preprocessing profile specifying target size and normalization.</param>
    /// <returns>Preprocessed image as a float array in NCHW or NHWC format.</returns>
    float[] Preprocess(Image<Rgb24> image, PreprocessProfile profile);
}
