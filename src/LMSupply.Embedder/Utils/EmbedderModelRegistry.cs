using System.Diagnostics;

namespace LMSupply.Embedder.Utils;

/// <summary>
/// Model registry for the Embedder domain.
/// </summary>
public sealed class EmbedderModelRegistry : ModelRegistryBase<ModelInfo>
{
    /// <summary>
    /// Auto-selection candidates sorted by size descending (largest first).
    /// </summary>
    internal static readonly ModelInfo[] AutoCandidates =
    [
        DefaultModels.BgeM3DefaultAlias,         // 568M params, 1024 dims, 8K context, 100+ langs
        DefaultModels.MultilingualE5LargeAlias,  // 560M params, 1024 dims
        DefaultModels.NomicEmbedTextV15,         // 137M params, 768 dims, 8K context, Matryoshka
        DefaultModels.MultilingualE5SmallAlias,  // 118M params, 384 dims
    ];

    /// <summary>
    /// Gets the default registry instance with built-in models.
    /// </summary>
    public static EmbedderModelRegistry Default { get; } = CreateDefault();

    /// <summary>
    /// Builds the default registry: built-in models plus the user's
    /// ~/.lmsupply/aliases.json "embedder" section (fail-soft).
    /// </summary>
    internal static EmbedderModelRegistry CreateDefault()
        => AliasConfiguration.ApplyDomain(new EmbedderModelRegistry(DefaultModels.All), AliasConfiguration.Domains.Embedder);

    /// <summary>
    /// Initializes a new registry with the specified system models.
    /// </summary>
    /// <param name="systemModels">Models to register as system defaults.</param>
    public EmbedderModelRegistry(IEnumerable<ModelInfo> systemModels)
        : base(systemModels) { }

    /// <summary>
    /// Gets the optimal model based on available VRAM.
    /// Selects the largest model that fits in available GPU memory,
    /// falling back to the smallest model if none fit.
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
        Trace.TraceInformation($"[EmbedderModelRegistry] Creating fallback model info for: {modelId}");

        return new ModelInfo
        {
            RepoId = modelId,
            AliasName = modelId,
            Dimensions = 384,
            MaxSequenceLength = 512,
            PoolingMode = PoolingMode.Mean,
            DoLowerCase = false,
            Description = $"Custom model: {modelId}",
            Subfolder = "onnx"
        };
    }
}
