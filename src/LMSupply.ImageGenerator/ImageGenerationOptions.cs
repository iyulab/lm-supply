namespace LMSupply.ImageGenerator;

/// <summary>
/// Options for a single image generation request.
/// </summary>
public sealed class ImageGenerationOptions
{
    /// <summary>
    /// Negative prompt to specify what to avoid in the generated image.
    /// </summary>
    public string? NegativePrompt { get; set; }

    /// <summary>
    /// Width of the generated image in pixels.
    /// Must be divisible by 8. Default: 512
    /// </summary>
    public int Width { get; set; } = 512;

    /// <summary>
    /// Height of the generated image in pixels.
    /// Must be divisible by 8. Default: 512
    /// </summary>
    public int Height { get; set; } = 512;

    /// <summary>
    /// Number of inference steps.
    /// LCM models work best with 2-4 steps. Default: 4
    /// </summary>
    public int Steps { get; set; } = 4;

    /// <summary>
    /// Guidance scale for classifier-free guidance.
    /// LCM models work best with 1.0-2.0. Default: 1.0
    /// </summary>
    public float GuidanceScale { get; set; } = 1.0f;

    /// <summary>
    /// Random seed for reproducible generation.
    /// Default: null (random seed)
    /// </summary>
    public int? Seed { get; set; }

    /// <summary>
    /// Whether to generate step-by-step preview images.
    /// Default: false (faster without previews)
    /// </summary>
    public bool GeneratePreviews { get; set; }

    /// <summary>
    /// Validates the options and throws if invalid.
    /// </summary>
    internal void Validate()
    {
        if (Width <= 0 || Width % 8 != 0)
            throw new ArgumentException("Width must be a positive multiple of 8.", nameof(Width));

        if (Height <= 0 || Height % 8 != 0)
            throw new ArgumentException("Height must be a positive multiple of 8.", nameof(Height));

        if (Steps < 1 || Steps > 50)
            throw new ArgumentException("Steps must be between 1 and 50.", nameof(Steps));

        if (GuidanceScale < 0)
            throw new ArgumentException("GuidanceScale must be non-negative.", nameof(GuidanceScale));
    }

    /// <summary>
    /// Creates a copy of these options with a specific seed.
    /// </summary>
    internal ImageGenerationOptions WithSeed(int seed) => new()
    {
        NegativePrompt = NegativePrompt,
        Width = Width,
        Height = Height,
        Steps = Steps,
        GuidanceScale = GuidanceScale,
        Seed = seed,
        GeneratePreviews = GeneratePreviews
    };
}
