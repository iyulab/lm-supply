using System.Diagnostics;

namespace LMSupply.Generator;

/// <summary>
/// Model registry for the Generator domain.
/// </summary>
public sealed class GeneratorModelRegistry : ModelRegistryBase<ModelInfo>
{
    /// <summary>
    /// Auto-selection candidates sorted by size descending (largest first).
    /// </summary>
    private static readonly ModelInfo[] AutoCandidates =
    [
        DefaultGeneratorModels.Phi4,      // 14B params
        DefaultGeneratorModels.Phi4Mini,  // 3.8B params
    ];

    /// <summary>
    /// Gets the default registry instance with built-in models.
    /// </summary>
    public static GeneratorModelRegistry Default { get; } = CreateDefault();

    /// <summary>
    /// Builds the default registry: built-in models plus the user's
    /// ~/.lmsupply/aliases.json "generator" section (fail-soft).
    /// </summary>
    internal static GeneratorModelRegistry CreateDefault()
        => AliasConfiguration.ApplyDomain(new GeneratorModelRegistry(DefaultGeneratorModels.All), AliasConfiguration.Domains.Generator);

    /// <summary>
    /// Initializes a new registry with the specified system models.
    /// Also registers "fast" as a system alias for the default model (Phi4Mini),
    /// since it is the smallest FC-capable ONNX model.
    /// </summary>
    /// <param name="systemModels">Models to register as system defaults.</param>
    public GeneratorModelRegistry(IEnumerable<ModelInfo> systemModels)
        : base(AppendFastAlias(systemModels)) { }

    private static IEnumerable<ModelInfo> AppendFastAlias(IEnumerable<ModelInfo> models)
    {
        foreach (var model in models)
            yield return model;

        // Register "fast" as an alias pointing to Phi4Mini
        yield return DefaultGeneratorModels.Phi4Mini with { AliasName = "fast" };
    }

    /// <summary>
    /// Gets the optimal ONNX model based on available VRAM.
    /// Selects the largest model that fits in available GPU memory,
    /// falling back to the smallest model if none fit.
    /// All models are MIT-licensed Phi-4 series with function calling support.
    /// </summary>
    protected override ModelInfo GetAutoModel()
    {
        return SelectLargestFitting(AutoCandidates) with { AliasName = "auto" };
    }

    /// <summary>
    /// Creates a fallback model info for unknown model IDs (HuggingFace repos or local paths).
    /// </summary>
    protected override ModelInfo CreateFallbackModelInfo(string modelId)
    {
        Trace.TraceInformation($"[GeneratorModelRegistry] Creating fallback model info for: {modelId}");

        var parts = modelId.Split('/');
        var name = parts.Length > 1 ? parts[1] : modelId;

        return new ModelInfo
        {
            ModelId = modelId,
            AliasName = modelId,
            DisplayName = name,
            Description = $"Custom model: {modelId}",
            ParameterCount = 0,
            License = LicenseTier.Conditional,
            LicenseName = "Unknown",
            ChatFormat = "chatml", // Safe default
            DefaultQuantization = Quantization.Quant4,
            RecommendedContextLength = 4096,
            NumLayers = 32,
            HiddenSize = 3072
        };
    }
}
