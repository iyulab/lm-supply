namespace LMSupply.Vision;

/// <summary>
/// Default implementation of <see cref="IImagePreprocessor"/>.
/// Handles resizing, cropping, and normalization according to model requirements.
/// </summary>
public sealed class ImagePreprocessor : IImagePreprocessor
{
    private readonly IImageLoader _imageLoader;

    /// <summary>
    /// Creates a new ImagePreprocessor with default image loader.
    /// </summary>
    public ImagePreprocessor() : this(ImageLoader.Instance)
    {
    }

    /// <summary>
    /// Creates a new ImagePreprocessor with custom image loader.
    /// </summary>
    /// <param name="imageLoader">Image loader to use.</param>
    public ImagePreprocessor(IImageLoader imageLoader)
    {
        _imageLoader = imageLoader ?? throw new ArgumentNullException(nameof(imageLoader));
    }

    /// <summary>
    /// Shared singleton instance.
    /// </summary>
    public static ImagePreprocessor Instance { get; } = new();

    /// <inheritdoc />
    public async Task<float[]> PreprocessAsync(
        string imagePath,
        PreprocessProfile profile,
        CancellationToken cancellationToken = default)
    {
        var image = await _imageLoader.LoadAsync(imagePath, cancellationToken).ConfigureAwait(false);
        return Preprocess(image, profile);
    }

    /// <inheritdoc />
    public async Task<float[]> PreprocessAsync(
        Stream imageStream,
        PreprocessProfile profile,
        CancellationToken cancellationToken = default)
    {
        var image = await _imageLoader.LoadAsync(imageStream, cancellationToken).ConfigureAwait(false);
        return Preprocess(image, profile);
    }

    /// <inheritdoc />
    public async Task<float[]> PreprocessAsync(
        byte[] imageData,
        PreprocessProfile profile,
        CancellationToken cancellationToken = default)
    {
        var image = await _imageLoader.LoadAsync(imageData, cancellationToken).ConfigureAwait(false);
        return Preprocess(image, profile);
    }

    /// <inheritdoc />
    public float[] Preprocess(RgbImage image, PreprocessProfile profile)
    {
        ArgumentNullException.ThrowIfNull(image);
        ArgumentNullException.ThrowIfNull(profile);

        var resized = profile.ResizeMode switch
        {
            ResizeMode.Stretch => image.Resize(profile.Width, profile.Height),
            ResizeMode.Fit => image.Letterbox(profile.Width, profile.Height),
            ResizeMode.CenterCrop or ResizeMode.ShortEdgeCrop => CenterCrop(image, profile.Width, profile.Height),
            _ => throw new ArgumentOutOfRangeException(nameof(profile), $"Unknown resize mode: {profile.ResizeMode}"),
        };

        return ToNormalizedTensor(resized, profile);
    }

    // Scale so the image covers the target (the shorter edge matches), then cut the centered target-sized window.
    private static RgbImage CenterCrop(RgbImage image, int targetWidth, int targetHeight)
    {
        float scale = Math.Max((float)targetWidth / image.Width, (float)targetHeight / image.Height);
        int newWidth = (int)Math.Ceiling(image.Width * scale);
        int newHeight = (int)Math.Ceiling(image.Height * scale);
        var scaled = image.Resize(newWidth, newHeight);
        return scaled.Crop((newWidth - targetWidth) / 2, (newHeight - targetHeight) / 2, targetWidth, targetHeight);
    }

    private static float[] ToNormalizedTensor(RgbImage image, PreprocessProfile profile)
    {
        int width = profile.Width;
        int height = profile.Height;
        int pixelCount = width * height;
        var tensor = new float[3 * pixelCount];

        for (int y = 0; y < height; y++)
        {
            var row = image.GetRow(y);
            for (int x = 0; x < width; x++)
            {
                // Normalize: (pixel/255 - mean) / std
                var r = ((row[x * 3] / 255f) - profile.Mean[0]) / profile.Std[0];
                var g = ((row[(x * 3) + 1] / 255f) - profile.Mean[1]) / profile.Std[1];
                var b = ((row[(x * 3) + 2] / 255f) - profile.Mean[2]) / profile.Std[2];
                if (profile.ChannelFirst)
                {
                    // NCHW format: [1, 3, H, W]
                    int idx = y * width + x;
                    tensor[idx] = r;
                    tensor[pixelCount + idx] = g;
                    tensor[(2 * pixelCount) + idx] = b;
                }
                else
                {
                    // NHWC format: [1, H, W, 3]
                    int idx = (y * width + x) * 3;
                    tensor[idx] = r;
                    tensor[idx + 1] = g;
                    tensor[idx + 2] = b;
                }
            }
        }

        return tensor;
    }
}
