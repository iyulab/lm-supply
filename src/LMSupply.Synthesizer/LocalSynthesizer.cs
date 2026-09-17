using LMSupply.Synthesizer.Core;
using LMSupply.Synthesizer.Models;

namespace LMSupply.Synthesizer;

/// <summary>
/// Main entry point for loading and using text-to-speech synthesis models.
/// </summary>
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
    /// Either a model alias (e.g., "default", "fast", "quality"),
    /// a HuggingFace model ID, or a local path to ONNX model files.
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
        options ??= new SynthesizerOptions();
        // An unsupported provider is refused here, before any model resolution or download (0.67.1).
        ExecutionProviderSupport.ThrowIfUnsupported(options.Provider);

        // Parse variant qualifier (e.g., "default:fp16" → modelId="default", hint="fp16")
        var (baseId, qualifier) = LMSupplyOptionsBase.SplitQualifier(modelIdOrPath);
        options.ModelId = baseId;
        options.QuantizationHint ??= qualifier;

        var synthesizer = new OnnxSynthesizerModel(options);

        // Eagerly initialize and warm up the model
        await synthesizer.WarmupAsync(cancellationToken);

        return synthesizer;
    }

    /// <summary>
    /// Loads the default TTS model (English US Lessac).
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
    /// Gets all registered model information.
    /// </summary>
    /// <returns>Collection of model information.</returns>
    public static IEnumerable<SynthesizerModelInfo> GetAllModels()
    {
        return SynthesizerModelRegistry.Default.GetAvailableModels();
    }
}
