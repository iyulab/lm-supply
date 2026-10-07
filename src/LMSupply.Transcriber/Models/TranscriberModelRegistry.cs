using System.Diagnostics;

namespace LMSupply.Transcriber.Models;

/// <summary>
/// Model registry for the Transcriber domain.
/// </summary>
public sealed class TranscriberModelRegistry : ModelRegistryBase<TranscriberModelInfo>
{
    /// <summary>
    /// Auto-selection candidates sorted by size descending (largest first).
    /// </summary>
    private static readonly TranscriberModelInfo[] AutoCandidates =
    [
        DefaultModels.WhisperLargeV3Turbo,  // 809M params, ~3.2GB
        DefaultModels.WhisperSmall,          // 244M params, ~970MB
        DefaultModels.WhisperBase,           // 74M params, ~290MB
        DefaultModels.WhisperTiny,           // 39M params, ~150MB
    ];

    /// <summary>
    /// Gets the default registry instance with built-in models.
    /// </summary>
    public static TranscriberModelRegistry Default { get; } = CreateDefault();

    /// <summary>
    /// Builds the default registry: built-in models plus the user's
    /// ~/.lmsupply/aliases.json "transcriber" section (fail-soft).
    /// </summary>
    internal static TranscriberModelRegistry CreateDefault()
        => AliasConfiguration.ApplyDomain(new TranscriberModelRegistry(DefaultModels.All), AliasConfiguration.Domains.Transcriber);

    /// <summary>
    /// Initializes a new registry with the specified system models.
    /// </summary>
    /// <param name="systemModels">Models to register as system defaults.</param>
    public TranscriberModelRegistry(IEnumerable<TranscriberModelInfo> systemModels)
        : base(systemModels) { }

    /// <summary>
    /// Gets the optimal model based on available VRAM.
    /// Selects the largest model that fits in available GPU memory,
    /// falling back to the smallest model if none fit.
    /// </summary>
    protected override TranscriberModelInfo GetAutoModel()
    {
        var selected = SelectLargestFitting(AutoCandidates);

        return new TranscriberModelInfo
        {
            Id = selected.Id,
            AliasName = "auto",
            DisplayName = selected.DisplayName,
            Architecture = selected.Architecture,
            ParametersM = selected.ParametersM,
            SizeBytes = selected.SizeBytes,
            WerLibriSpeech = selected.WerLibriSpeech,
            MaxDurationSeconds = selected.MaxDurationSeconds,
            SampleRate = selected.SampleRate,
            NumMelBins = selected.NumMelBins,
            HiddenSize = selected.HiddenSize,
            EncoderFile = selected.EncoderFile,
            DecoderFile = selected.DecoderFile,
            SupportedLanguages = selected.SupportedLanguages,
            IsMultilingual = selected.IsMultilingual,
            Description = selected.Description,
            License = selected.License
        };
    }

    /// <summary>
    /// Creates a fallback model info for unknown model IDs (HuggingFace repos or local paths).
    /// </summary>
    protected override TranscriberModelInfo CreateFallbackModelInfo(string modelId)
    {
        Trace.TraceInformation($"[TranscriberModelRegistry] Creating fallback model info for: {modelId}");

        var parts = modelId.Split('/');
        var name = parts.Length > 1 ? parts[1] : modelId;

        return new TranscriberModelInfo
        {
            Id = modelId,
            AliasName = modelId,
            DisplayName = name,
            Architecture = "Whisper"
        };
    }
}
