using System.Diagnostics;
using LMSupply.Core.Download;

namespace LMSupply.Generator;

/// <summary>
/// Model registry for the Generator domain.
/// </summary>
public sealed class GeneratorModelRegistry : ModelRegistryBase<GeneratorModelEntry>
{
    // COMPAT(remove-after: 0.150.0) — names retired in 0.144.0. They load nothing; the hint tells the
    // person who typed one what replaced it.
    private static readonly Dictionary<string, string> _retiredAliases = new(StringComparer.OrdinalIgnoreCase)
    {
        ["fast"] = "it is now 'entry'",
        ["quality"] = "it is now 'advanced' (the model is also 'phi-4')",
    };

    /// <inheritdoc />
    protected override IReadOnlyDictionary<string, string> RetiredAliases => _retiredAliases;

    /// <summary>
    /// Auto-selection candidates sorted by size descending (largest first).
    /// </summary>
    private static readonly GeneratorModelEntry[] AutoCandidates =
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
    /// Phi4Mini, the smallest FC-capable ONNX model, holds the <c>entry</c> rung and Phi-4 the <c>advanced</c> one.
    /// </summary>
    /// <param name="systemModels">Models to register as system defaults.</param>
    public GeneratorModelRegistry(IEnumerable<GeneratorModelEntry> systemModels)
        : base(systemModels) { }

    /// <summary>
    /// Gets the optimal ONNX model based on available VRAM.
    /// Selects the largest model that fits in available GPU memory,
    /// falling back to the smallest model if none fit.
    /// All models are MIT-licensed Phi-4 series with function calling support.
    /// </summary>
    protected override GeneratorModelEntry GetAutoModel()
    {
        return SelectLargestFitting(AutoCandidates) with { AliasName = "auto" };
    }

    /// <summary>
    /// Creates a fallback model info for unknown model IDs (HuggingFace repos or local paths).
    /// </summary>
    protected override GeneratorModelEntry CreateFallbackModelInfo(string modelId)
    {
        Trace.TraceInformation($"[GeneratorModelRegistry] Creating fallback model info for: {modelId}");

        var parts = modelId.Split('/');
        var name = parts.Length > 1 ? parts[1] : modelId;

        return new GeneratorModelEntry
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
