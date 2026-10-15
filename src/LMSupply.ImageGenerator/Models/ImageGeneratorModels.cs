using LMSupply.Core;

namespace LMSupply.ImageGenerator.Models;

/// <summary>
/// Provides definitions for built-in supported image generation models.
/// All models use Latent Consistency Model (LCM) for fast inference (2-8 steps).
/// </summary>
public static class ImageGeneratorModels
{
    /// <summary>
    /// Gets the default model (balanced speed and quality).
    /// </summary>
    public static ImageGeneratorModelInfo Default => LcmDreamshaperV7;

    /// <summary>
    /// LCM-Dreamshaper-V7 - Default model, best balance of speed and quality.
    /// 512x512, ~1GB, 4 steps, guidance 1.0.
    /// </summary>
    public static ImageGeneratorModelInfo LcmDreamshaperV7 { get; } = new()
    {
        ModelId = "TheyCallMeHex/LCM-Dreamshaper-V7-ONNX",
        AliasName = "default",
        ModelName = "LCM-Dreamshaper-V7",
        Description = "Default: LCM-Dreamshaper-V7, balanced speed and quality",
        Architecture = "LCM",
        Provider = ExecutionProvider.Auto,
        IsFp16 = false,
        DefaultWidth = 512,
        DefaultHeight = 512,
        RecommendedSteps = 4,
        RecommendedGuidanceScale = 1.0f
    };

    /// <summary>
    /// LCM-Dreamshaper-V7 Fast - Fewer steps for faster generation.
    /// 512x512, ~1GB, 2 steps, guidance 1.0.
    /// </summary>
    public static ImageGeneratorModelInfo LcmDreamshaperV7Fast { get; } = new()
    {
        ModelId = "TheyCallMeHex/LCM-Dreamshaper-V7-ONNX",
        AliasName = "fast",
        ModelName = "LCM-Dreamshaper-V7",
        Description = "Fast: LCM-Dreamshaper-V7 with fewer steps",
        Architecture = "LCM",
        Provider = ExecutionProvider.Auto,
        IsFp16 = false,
        DefaultWidth = 512,
        DefaultHeight = 512,
        RecommendedSteps = 2,
        RecommendedGuidanceScale = 1.0f
    };

    /// <summary>
    /// LCM-Dreamshaper-V7 Quality - More steps and higher guidance for better quality.
    /// 512x512, ~1GB, 8 steps, guidance 1.5.
    /// </summary>
    public static ImageGeneratorModelInfo LcmDreamshaperV7Quality { get; } = new()
    {
        ModelId = "TheyCallMeHex/LCM-Dreamshaper-V7-ONNX",
        AliasName = "quality",
        ModelName = "LCM-Dreamshaper-V7",
        Description = "Quality: LCM-Dreamshaper-V7 with more steps and guidance",
        Architecture = "LCM",
        Provider = ExecutionProvider.Auto,
        IsFp16 = false,
        DefaultWidth = 512,
        DefaultHeight = 512,
        RecommendedSteps = 8,
        RecommendedGuidanceScale = 1.5f
    };

    /// <summary>
    /// LCM-Dreamshaper-V7 - Direct model reference with standard settings.
    /// </summary>
    public static ImageGeneratorModelInfo LcmDreamshaperV7Direct { get; } = new()
    {
        ModelId = "TheyCallMeHex/LCM-Dreamshaper-V7-ONNX",
        AliasName = "lcm-dreamshaper-v7",
        ModelName = "LCM-Dreamshaper-V7",
        Description = "LCM-Dreamshaper-V7 (direct reference)",
        Architecture = "LCM",
        Provider = ExecutionProvider.Auto,
        IsFp16 = false,
        DefaultWidth = 512,
        DefaultHeight = 512,
        RecommendedSteps = 4,
        RecommendedGuidanceScale = 1.0f
    };

    /// <summary>
    /// LCM-SSD-1B - Smaller, faster alternative (1B params).
    /// </summary>
    public static ImageGeneratorModelInfo LcmSsd1B { get; } = new()
    {
        ModelId = "segmind/LCM-SSD-1B-onnx",
        AliasName = "lcm-ssd-1b",
        ModelName = "LCM-SSD-1B",
        Description = "LCM-SSD-1B: Smaller, faster alternative (1B params)",
        Architecture = "LCM",
        Provider = ExecutionProvider.Auto,
        IsFp16 = false,
        DefaultWidth = 512,
        DefaultHeight = 512,
        RecommendedSteps = 4,
        RecommendedGuidanceScale = 1.0f
    };

    /// <summary>
    /// Gets all built-in models.
    /// </summary>
    public static IReadOnlyList<ImageGeneratorModelInfo> All { get; } =
    [
        LcmDreamshaperV7,       // default
        LcmDreamshaperV7Fast,   // fast
        LcmDreamshaperV7Quality,// quality
        LcmDreamshaperV7Direct, // lcm-dreamshaper-v7
        LcmSsd1B                // lcm-ssd-1b
    ];
}
