using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace LMSupply.Vision;

/// <summary>
/// Default implementation of <see cref="IImagePreprocessor"/> using ImageSharp.
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
        using var image = await _imageLoader.LoadAsync(imagePath, cancellationToken).ConfigureAwait(false);
        return Preprocess(image, profile);
    }

    /// <inheritdoc />
    public async Task<float[]> PreprocessAsync(
        Stream imageStream,
        PreprocessProfile profile,
        CancellationToken cancellationToken = default)
    {
        using var image = await _imageLoader.LoadAsync(imageStream, cancellationToken).ConfigureAwait(false);
        return Preprocess(image, profile);
    }

    /// <inheritdoc />
    public async Task<float[]> PreprocessAsync(
        byte[] imageData,
        PreprocessProfile profile,
        CancellationToken cancellationToken = default)
    {
        using var image = await _imageLoader.LoadAsync(imageData, cancellationToken).ConfigureAwait(false);
        return Preprocess(image, profile);
    }

    /// <inheritdoc />
    public float[] Preprocess(Image<Rgb24> image, PreprocessProfile profile)
    {
        ArgumentNullException.ThrowIfNull(image);
        ArgumentNullException.ThrowIfNull(profile);

        // Clone to avoid modifying the original
        using var processedImage = image.Clone();

        // Apply resize/crop according to profile
        ApplyResize(processedImage, profile);

        // Convert to normalized float tensor
        return ToNormalizedTensor(processedImage, profile);
    }

    private static void ApplyResize(Image<Rgb24> image, PreprocessProfile profile)
    {
        var targetWidth = profile.Width;
        var targetHeight = profile.Height;

        switch (profile.ResizeMode)
        {
            case ResizeMode.Stretch:
                image.Mutate(x => x.Resize(new ResizeOptions
                {
                    Size = new Size(targetWidth, targetHeight),
                    Mode = SixLabors.ImageSharp.Processing.ResizeMode.Stretch,
                    Sampler = KnownResamplers.Bicubic
                }));
                break;

            case ResizeMode.Fit:
                // Resize to fit within bounds, then pad
                image.Mutate(x => x.Resize(new ResizeOptions
                {
                    Size = new Size(targetWidth, targetHeight),
                    Mode = SixLabors.ImageSharp.Processing.ResizeMode.Pad,
                    Sampler = KnownResamplers.Bicubic,
                    PadColor = Color.Black
                }));
                break;

            case ResizeMode.CenterCrop:
                // Resize so shorter edge matches, then center crop
                ApplyCenterCrop(image, targetWidth, targetHeight);
                break;

            case ResizeMode.ShortEdgeCrop:
                // Same as CenterCrop but explicit naming
                ApplyCenterCrop(image, targetWidth, targetHeight);
                break;

            default:
                throw new ArgumentOutOfRangeException(nameof(profile), $"Unknown resize mode: {profile.ResizeMode}");
        }
    }

    private static void ApplyCenterCrop(Image<Rgb24> image, int targetWidth, int targetHeight)
    {
        // Calculate scale factor based on shorter edge
        float scaleX = (float)targetWidth / image.Width;
        float scaleY = (float)targetHeight / image.Height;
        float scale = Math.Max(scaleX, scaleY);

        int newWidth = (int)Math.Ceiling(image.Width * scale);
        int newHeight = (int)Math.Ceiling(image.Height * scale);

        // Resize
        image.Mutate(x => x.Resize(new ResizeOptions
        {
            Size = new Size(newWidth, newHeight),
            Mode = SixLabors.ImageSharp.Processing.ResizeMode.Stretch,
            Sampler = KnownResamplers.Bicubic
        }));

        // Center crop
        int cropX = (newWidth - targetWidth) / 2;
        int cropY = (newHeight - targetHeight) / 2;

        image.Mutate(x => x.Crop(new Rectangle(cropX, cropY, targetWidth, targetHeight)));
    }

    private static float[] ToNormalizedTensor(Image<Rgb24> image, PreprocessProfile profile)
    {
        int width = profile.Width;
        int height = profile.Height;
        int pixelCount = width * height;

        float[] tensor;

        if (profile.ChannelFirst)
        {
            // NCHW format: [1, 3, H, W]
            tensor = new float[3 * pixelCount];

            image.ProcessPixelRows(accessor =>
            {
                for (int y = 0; y < height; y++)
                {
                    var row = accessor.GetRowSpan(y);
                    for (int x = 0; x < width; x++)
                    {
                        var pixel = row[x];
                        int idx = y * width + x;

                        // Normalize: (pixel/255 - mean) / std
                        tensor[0 * pixelCount + idx] = ((pixel.R / 255f) - profile.Mean[0]) / profile.Std[0];
                        tensor[1 * pixelCount + idx] = ((pixel.G / 255f) - profile.Mean[1]) / profile.Std[1];
                        tensor[2 * pixelCount + idx] = ((pixel.B / 255f) - profile.Mean[2]) / profile.Std[2];
                    }
                }
            });
        }
        else
        {
            // NHWC format: [1, H, W, 3]
            tensor = new float[3 * pixelCount];

            image.ProcessPixelRows(accessor =>
            {
                for (int y = 0; y < height; y++)
                {
                    var row = accessor.GetRowSpan(y);
                    for (int x = 0; x < width; x++)
                    {
                        var pixel = row[x];
                        int idx = (y * width + x) * 3;

                        // Normalize: (pixel/255 - mean) / std
                        tensor[idx + 0] = ((pixel.R / 255f) - profile.Mean[0]) / profile.Std[0];
                        tensor[idx + 1] = ((pixel.G / 255f) - profile.Mean[1]) / profile.Std[1];
                        tensor[idx + 2] = ((pixel.B / 255f) - profile.Mean[2]) / profile.Std[2];
                    }
                }
            });
        }

        return tensor;
    }
}
