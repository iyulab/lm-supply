using LMSupply.Synthesizer.Core;
using LMSupply.Synthesizer.Models;
using LMSupply.Download;

namespace LMSupply.Synthesizer;

/// <summary>
/// Main entry point for loading and using text-to-speech synthesis models.
/// </summary>
/// <remarks>
/// A voice is run by the contract its config declares. Text for an English or Korean voice is turned into its phonemes by
/// the package's built-in phonemizer for that language; for a voice in another language it needs a caller's
/// <see cref="SynthesizerOptions.Phonemizer"/>, and throws <see cref="NotSupportedException"/> without one. Phonemes
/// (<see cref="ISynthesizerModel.SynthesizePhonemesAsync"/>) work for every voice. See the package README.
/// </remarks>
public static class LocalSynthesizer
{
    /// <summary>
    /// Gets the model registry for the Synthesizer domain.
    /// Provides access to model resolution, alias management, and model enumeration.
    /// </summary>
    public static IModelRegistry<SynthesizerModelInfo> Registry => SynthesizerModelRegistry.Default;

    /// <summary>
    /// Shared pool for named model management. Supports GetOrLoadAsync / UnloadAsync by model ID.
    /// </summary>
    public static LMSupply.Pool.ModelPool<ISynthesizerModel, SynthesizerOptions> Pool { get; }
        = new(new Pool.SynthesizerLoader());

    /// <summary>
    /// Loads a TTS model by name or path.
    /// </summary>
    /// <param name="modelIdOrPath">
    /// Either a model alias (e.g., "default", "fast", "quality"), a catalog voice's
    /// <see cref="SynthesizerModelInfo.CatalogKey"/> (e.g., "rhasspy/piper-voices:en/en_US/ljspeech/medium"),
    /// a HuggingFace repository ID, or a local path to ONNX model files. The repository ID of the catalog's
    /// voices names no single voice and loads the first of them, the default one.
    /// </param>
    /// <param name="options">Optional configuration options.</param>
    /// <param name="progress">Optional progress reporting for downloads.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>A loaded synthesizer ready for inference.</returns>
    public static async Task<ISynthesizerModel> LoadAsync(
        string modelIdOrPath,
        SynthesizerOptions? options = null,
        IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        progress = LoadProgress.Begin(progress);
        options = options?.Clone() ?? new SynthesizerOptions();
        // An unsupported provider is refused here, before any model resolution or download (0.67.1).
        ExecutionProviderSupport.ThrowIfUnsupported(options.Provider);

        // Parse variant qualifier (e.g., "default:fp16" → modelId="default", hint="fp16")
        var (baseId, qualifier) = LMSupplyOptionsBase.SplitQualifier(modelIdOrPath);
        options.ModelId = baseId;
        options.QuantizationHint ??= qualifier;

        var synthesizer = new OnnxSynthesizerModel(options, progress);

        // Eagerly initialize and warm up the model
        await synthesizer.WarmupAsync(cancellationToken);

        return synthesizer;
    }

    /// <summary>
    /// Loads the default TTS model (English US LJSpeech).
    /// </summary>
    /// <param name="options">Optional configuration options.</param>
    /// <param name="progress">Optional progress reporting for downloads.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>A loaded synthesizer ready for inference.</returns>
    public static Task<ISynthesizerModel> LoadAsync(
        SynthesizerOptions? options = null,
        IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        progress = LoadProgress.Begin(progress);
        return LoadAsync(options?.ModelId ?? "default", options, progress, cancellationToken);
    }

    /// <summary>
    /// Gets a list of pre-configured model aliases available for use.
    /// </summary>
    /// <returns>Available model aliases.</returns>
    public static IEnumerable<string> GetAvailableModels()
    {
        return SynthesizerModelRegistry.Default.GetAliases().Select(a => a.Name);
    }

    /// <summary>
    /// Gets every voice of the catalog, once each.
    /// </summary>
    /// <returns>Collection of model information.</returns>
    public static IEnumerable<SynthesizerModelInfo> GetAllModels()
    {
        return SynthesizerModelRegistry.Default.GetAvailableModels();
    }
}
