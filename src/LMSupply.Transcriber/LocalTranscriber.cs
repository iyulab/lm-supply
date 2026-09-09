using LMSupply.Transcriber.Core;
using LMSupply.Transcriber.Models;

namespace LMSupply.Transcriber;

/// <summary>
/// Main entry point for loading and using speech-to-text transcription models.
/// </summary>
public static class LocalTranscriber
{
    /// <summary>
    /// Gets the model registry for the Transcriber domain.
    /// Provides access to model resolution, alias management, and model enumeration.
    /// </summary>
    public static IModelRegistry<TranscriberModelInfo> Registry => TranscriberModelRegistry.Default;

    /// <summary>
    /// Shared pool for named model management. Supports GetOrLoadAsync / UnloadAsync by model ID.
    /// </summary>
    public static LMSupply.Pool.ModelPool<ITranscriberModel, TranscriberOptions> Pool { get; }
        = new(new Pool.TranscriberLoader());

    /// <summary>
    /// Loads a transcription model by name or path.
    /// </summary>
    /// <param name="modelIdOrPath">
    /// Either a model alias (e.g., "default", "fast", "quality", "large"),
    /// a HuggingFace model ID (e.g., "openai/whisper-base"),
    /// or a local path to ONNX model files.
    /// </param>
    /// <param name="options">Optional configuration options.</param>
    /// <param name="progress">Optional progress reporting for downloads.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>A loaded transcriber ready for inference.</returns>
    public static async Task<ITranscriberModel> LoadAsync(
        string modelIdOrPath,
        TranscriberOptions? options = null,
        IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        options ??= new TranscriberOptions();

        // Parse variant qualifier (e.g., "large:fp16" → modelId="large", hint="fp16")
        var (baseId, qualifier) = LMSupplyOptionsBase.SplitQualifier(modelIdOrPath);
        options.ModelId = baseId;
        options.QuantizationHint ??= qualifier;

        var transcriber = CreateModel(options);

        // Eagerly initialize and warm up the model
        await transcriber.WarmupAsync(cancellationToken);

        return transcriber;
    }

    /// <summary>
    /// Picks the model family from the registry entry's <see cref="TranscriberModelInfo.Architecture"/> (or, for a local
    /// directory, from its <c>config.json</c>). Everything that is not a known Parakeet TDT export takes the Whisper path —
    /// which is what every previously supported id did.
    /// </summary>
    private static ITranscriberModel CreateModel(TranscriberOptions options)
    {
        if (Registry.TryResolve(options.ModelId, out var info) && info is not null
            && TranscriberArchitectures.IsParakeetTdt(info.Architecture))
        {
            return new ParakeetTdtTranscriberModel(options, info);
        }

        if (Directory.Exists(options.ModelId)
            && Internal.NemoConfigReader.ReadConfig(options.ModelId) is { IsTdt: true } config)
        {
            var template = DefaultModels.ParakeetTdt06BV3;
            return new ParakeetTdtTranscriberModel(options, new TranscriberModelInfo
            {
                Id = options.ModelId,
                AliasName = Path.GetFileName(options.ModelId.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar)),
                DisplayName = "Parakeet TDT (local)",
                Architecture = TranscriberArchitectures.ParakeetTdt,
                NumMelBins = config.FeaturesSize ?? template.NumMelBins,
                HiddenSize = template.HiddenSize,
                MaxDurationSeconds = template.MaxDurationSeconds,
                EncoderFile = template.EncoderFile,
                DecoderFile = template.DecoderFile,
                IsMultilingual = true,
                SupportedLanguages = template.SupportedLanguages,
                License = template.License
            });
        }

        return new OnnxTranscriberModel(options);
    }

    /// <summary>
    /// Loads the default transcription model (Whisper Base).
    /// </summary>
    /// <param name="options">Optional configuration options.</param>
    /// <param name="progress">Optional progress reporting for downloads.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>A loaded transcriber ready for inference.</returns>
    public static Task<ITranscriberModel> LoadAsync(
        TranscriberOptions? options = null,
        IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        return LoadAsync(options?.ModelId ?? "default", options, progress, cancellationToken);
    }

    /// <summary>
    /// Gets a list of pre-configured model aliases available for use.
    /// </summary>
    /// <returns>Available model aliases.</returns>
    public static IEnumerable<string> GetAvailableModels()
    {
        return TranscriberModelRegistry.Default.GetAliases().Select(a => a.Name);
    }

    /// <summary>
    /// Gets all registered model information.
    /// </summary>
    /// <returns>Collection of model information.</returns>
    public static IEnumerable<TranscriberModelInfo> GetAllModels()
    {
        return TranscriberModelRegistry.Default.GetAvailableModels();
    }
}
