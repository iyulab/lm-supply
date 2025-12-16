using LMSupply.Translator.Core;
using LMSupply.Translator.Models;

namespace LMSupply.Translator;

/// <summary>
/// Main entry point for loading and using translation models.
/// </summary>
public static class LocalTranslator
{
    /// <summary>
    /// Loads a translation model by name or path.
    /// </summary>
    /// <param name="modelIdOrPath">
    /// Either a model alias (e.g., "default", "ko-en", "en-ko"),
    /// a HuggingFace model ID (e.g., "Helsinki-NLP/opus-mt-ko-en"),
    /// or a local path to ONNX model files.
    /// </param>
    /// <param name="options">Optional configuration options.</param>
    /// <param name="progress">Optional progress reporting for downloads.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>A loaded translator ready for inference.</returns>
    public static async Task<ITranslatorModel> LoadAsync(
        string modelIdOrPath,
        TranslatorOptions? options = null,
        IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        options ??= new TranslatorOptions();
        options.ModelId = modelIdOrPath;

        var translator = new OnnxTranslatorModel(options);

        // Eagerly initialize and warm up the model
        await translator.WarmupAsync(cancellationToken);

        return translator;
    }

    /// <summary>
    /// Loads the default translation model (Korean to English).
    /// </summary>
    /// <param name="options">Optional configuration options.</param>
    /// <param name="progress">Optional progress reporting for downloads.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>A loaded translator ready for inference.</returns>
    public static Task<ITranslatorModel> LoadAsync(
        TranslatorOptions? options = null,
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
        return TranslatorModelRegistry.Default.GetAliases();
    }

    /// <summary>
    /// Gets all registered model information.
    /// </summary>
    /// <returns>Collection of model information.</returns>
    public static IEnumerable<TranslatorModelInfo> GetAllModels()
    {
        return TranslatorModelRegistry.Default.GetAll();
    }
}
