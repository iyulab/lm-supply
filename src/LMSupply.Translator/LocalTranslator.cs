using LMSupply.Translator.Core;
using LMSupply.Translator.Models;

namespace LMSupply.Translator;

/// <summary>
/// Main entry point for loading and using translation models.
/// </summary>
public static class LocalTranslator
{
    /// <summary>
    /// Gets the model registry for the Translator domain.
    /// Provides access to model resolution, alias management, and model enumeration.
    /// </summary>
    public static IModelRegistry<TranslatorModelInfo> Registry => TranslatorModelRegistry.Default;

    /// <summary>
    /// Shared pool for named model management. Supports GetOrLoadAsync / UnloadAsync by model ID.
    /// </summary>
    public static LMSupply.Pool.ModelPool<ITranslatorModel, TranslatorOptions> Pool { get; }
        = new(new Pool.TranslatorLoader());

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
        // An unsupported provider is refused here, before any model resolution or download (0.67.1).
        ExecutionProviderSupport.ThrowIfUnsupported(options.Provider);

        // Parse variant qualifier (e.g., "default:fp16" → modelId="default", hint="fp16")
        var (baseId, qualifier) = LMSupplyOptionsBase.SplitQualifier(modelIdOrPath);
        options.ModelId = baseId;
        options.QuantizationHint ??= qualifier;

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
        return LoadAsync(options?.ModelId ?? "default", options, progress, cancellationToken);
    }

    /// <summary>
    /// Gets a list of pre-configured model aliases available for use.
    /// </summary>
    /// <returns>Available model aliases.</returns>
    public static IEnumerable<string> GetAvailableModels()
    {
        return TranslatorModelRegistry.Default.GetAliases().Select(a => a.Name);
    }

    /// <summary>
    /// Gets all registered model information.
    /// </summary>
    /// <returns>Collection of model information.</returns>
    public static IEnumerable<TranslatorModelInfo> GetAllModels()
    {
        return TranslatorModelRegistry.Default.GetAvailableModels();
    }
}
