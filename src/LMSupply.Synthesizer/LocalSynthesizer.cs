using LMSupply.Synthesizer.Core;
using LMSupply.Synthesizer.Models;

namespace LMSupply.Synthesizer;

/// <summary>
/// Main entry point for loading and using text-to-speech synthesis models.
/// </summary>
public static class LocalSynthesizer
{
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
        options.ModelId = modelIdOrPath;

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
        return LoadAsync("default", options, progress, cancellationToken);
    }

    /// <summary>
    /// Gets a list of pre-configured model aliases available for use.
    /// </summary>
    /// <returns>Available model aliases.</returns>
    public static IEnumerable<string> GetAvailableModels()
    {
        return SynthesizerModelRegistry.Default.GetAliases();
    }

    /// <summary>
    /// Gets all registered model information.
    /// </summary>
    /// <returns>Collection of model information.</returns>
    public static IEnumerable<SynthesizerModelInfo> GetAllModels()
    {
        return SynthesizerModelRegistry.Default.GetAll();
    }
}
