using System.Diagnostics;

namespace LMSupply.Synthesizer.Models;

/// <summary>
/// Model registry for the Synthesizer domain.
/// </summary>
public sealed class SynthesizerModelRegistry : ModelRegistryBase<SynthesizerModelInfo>
{
    // COMPAT(remove-after: 0.150.0) — names retired in 0.144.0. They load nothing; the hint tells the
    // person who typed one what replaced it.
    private static readonly Dictionary<string, string> _retiredAliases = new(StringComparer.OrdinalIgnoreCase)
    {
        ["fast"] = "it is now the voice 'ryan'",
        ["quality"] = "it is now the voice 'amy'",
    };

    /// <inheritdoc />
    protected override IReadOnlyDictionary<string, string> RetiredAliases => _retiredAliases;

    /// <summary>
    /// Auto-selection candidates sorted by size descending (largest first).
    /// TTS models are small (~16-64MB), so VRAM is rarely a constraint;
    /// this ordering naturally selects highest quality when possible.
    /// </summary>
    private static readonly SynthesizerModelInfo[] AutoCandidates =
    [
        SynthesizerModels.EnUsAmy,     // 64MB - amy
        SynthesizerModels.EnUsLessac,  // 63MB - default
        SynthesizerModels.EnUsRyan,    // 16MB - ryan
    ];

    /// <summary>
    /// Gets the default registry instance with built-in models.
    /// </summary>
    public static SynthesizerModelRegistry Default { get; } = CreateDefault();

    /// <summary>
    /// Builds the default registry: built-in models plus the user's
    /// ~/.lmsupply/aliases.json "synthesizer" section (fail-soft).
    /// </summary>
    internal static SynthesizerModelRegistry CreateDefault()
        => AliasConfiguration.ApplyDomain(new SynthesizerModelRegistry(SynthesizerModels.All), AliasConfiguration.Domains.Synthesizer);

    /// <summary>
    /// Initializes a new registry with the specified system models.
    /// </summary>
    /// <param name="systemModels">Models to register as system defaults.</param>
    public SynthesizerModelRegistry(IEnumerable<SynthesizerModelInfo> systemModels)
        : base(systemModels) { }

    /// <summary>
    /// Gets the optimal model based on available VRAM.
    /// Selects the largest model that fits in available GPU memory,
    /// falling back to the smallest model if none fit.
    /// </summary>
    protected override SynthesizerModelInfo GetAutoModel()
    {
        return SelectLargestFitting(AutoCandidates) with { AliasName = "auto" };
    }

    /// <summary>
    /// Creates a fallback model info for unknown model IDs (HuggingFace repos or local paths).
    /// </summary>
    protected override SynthesizerModelInfo CreateFallbackModelInfo(string modelId)
    {
        Trace.TraceInformation($"[SynthesizerModelRegistry] Creating fallback model info for: {modelId}");

        var parts = modelId.Split('/');
        var name = parts.Length > 1 ? parts[1] : modelId;

        return new SynthesizerModelInfo
        {
            Id = modelId,
            AliasName = modelId,
            DisplayName = name,
            Architecture = "VITS"
        };
    }
}
