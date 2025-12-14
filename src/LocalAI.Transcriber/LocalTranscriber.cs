using LocalAI.Transcriber.Core;
using LocalAI.Transcriber.Models;

namespace LocalAI.Transcriber;

/// <summary>
/// Main entry point for loading and using speech-to-text transcription models.
/// </summary>
public static class LocalTranscriber
{
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
        options.ModelId = modelIdOrPath;

        var transcriber = new OnnxTranscriberModel(options);

        // Eagerly initialize and warm up the model
        await transcriber.WarmupAsync(cancellationToken);

        return transcriber;
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
        return LoadAsync("default", options, progress, cancellationToken);
    }

    /// <summary>
    /// Gets a list of pre-configured model aliases available for use.
    /// </summary>
    /// <returns>Available model aliases.</returns>
    public static IEnumerable<string> GetAvailableModels()
    {
        return TranscriberModelRegistry.Default.GetAliases();
    }

    /// <summary>
    /// Gets all registered model information.
    /// </summary>
    /// <returns>Collection of model information.</returns>
    public static IEnumerable<TranscriberModelInfo> GetAllModels()
    {
        return TranscriberModelRegistry.Default.GetAll();
    }
}
