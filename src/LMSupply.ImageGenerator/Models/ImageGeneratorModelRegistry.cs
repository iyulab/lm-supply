using System.Diagnostics;
using LMSupply.Hardware;

namespace LMSupply.ImageGenerator.Models;

/// <summary>
/// Model registry for the ImageGenerator domain.
/// </summary>
public sealed class ImageGeneratorModelRegistry : ModelRegistryBase<ImageGeneratorModelInfo>
{
    /// <summary>
    /// Gets the default registry instance with built-in models.
    /// </summary>
    public static ImageGeneratorModelRegistry Default { get; } = CreateDefault();

    /// <summary>
    /// Builds the default registry: built-in models plus the user's
    /// ~/.lmsupply/aliases.json "imagegenerator" section (fail-soft).
    /// </summary>
    internal static ImageGeneratorModelRegistry CreateDefault()
        => AliasConfiguration.ApplyDomain(new ImageGeneratorModelRegistry(ImageGeneratorModels.All), AliasConfiguration.Domains.ImageGenerator);

    /// <summary>
    /// Initializes a new registry with the specified system models.
    /// </summary>
    /// <param name="systemModels">Models to register as system defaults.</param>
    public ImageGeneratorModelRegistry(IEnumerable<ImageGeneratorModelInfo> systemModels)
        : base(systemModels) { }

    /// <summary>
    /// Gets the optimal model based on current hardware profile.
    /// Uses PerformanceTier to select appropriate model configuration.
    /// </summary>
    /// <remarks>
    /// Tier mapping:
    /// - Low:    LCM-Dreamshaper-V7 Fast (2 steps)
    /// - Medium: LCM-Dreamshaper-V7 Default (4 steps)
    /// - High:   LCM-Dreamshaper-V7 Quality (8 steps)
    /// - Ultra:  LCM-Dreamshaper-V7 Quality (8 steps)
    /// </remarks>
    protected override ImageGeneratorModelInfo GetAutoModel()
    {
        var tier = HardwareProfile.Current.Tier;
        Trace.TraceInformation($"[ImageGeneratorModelRegistry] Auto-selecting model for tier: {tier}");

        var model = tier switch
        {
            PerformanceTier.Ultra or PerformanceTier.High => ImageGeneratorModels.LcmDreamshaperV7Quality,
            PerformanceTier.Medium => ImageGeneratorModels.LcmDreamshaperV7,
            _ => ImageGeneratorModels.LcmDreamshaperV7Fast
        };

        return new ImageGeneratorModelInfo
        {
            ModelId = model.ModelId,
            AliasName = "auto",
            ModelName = model.ModelName,
            Description = model.Description,
            Architecture = model.Architecture,
            Provider = model.Provider,
            IsFp16 = model.IsFp16,
            DefaultWidth = model.DefaultWidth,
            DefaultHeight = model.DefaultHeight,
            RecommendedSteps = model.RecommendedSteps,
            RecommendedGuidanceScale = model.RecommendedGuidanceScale
        };
    }

    /// <summary>
    /// Creates a fallback model info for unknown model IDs (HuggingFace repos or local paths).
    /// </summary>
    protected override ImageGeneratorModelInfo CreateFallbackModelInfo(string modelId)
    {
        Trace.TraceInformation($"[ImageGeneratorModelRegistry] Creating fallback model info for: {modelId}");

        var parts = modelId.Split('/');
        var name = parts.Length > 1 ? parts[1] : modelId;

        return new ImageGeneratorModelInfo
        {
            ModelId = modelId,
            AliasName = modelId,
            ModelName = name,
            Description = $"Custom model: {modelId}",
            Architecture = "LCM",
            Provider = ExecutionProvider.Auto,
            IsFp16 = false,
            DefaultWidth = 512,
            DefaultHeight = 512,
            RecommendedSteps = 4,
            RecommendedGuidanceScale = 1.0f
        };
    }

    /// <summary>
    /// Converts an <see cref="ImageGeneratorModelInfo"/> to a <see cref="ModelDefinition"/>
    /// for use with the internal inference pipeline.
    /// </summary>
    internal static ModelDefinition ToModelDefinition(ImageGeneratorModelInfo info)
    {
        return new ModelDefinition(
            info.ModelId,
            info.ModelName,
            info.RecommendedSteps,
            info.RecommendedGuidanceScale);
    }
}
