using LMSupply.Translator.Core;
using LMSupply.Translator.Models;
using LMSupply.Download;
using LMSupply.Exceptions;

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
    /// Either a model alias (e.g., "default", "ko-en", "ja-en"),
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
        progress = LoadProgress.Begin(progress);
        options = options?.Clone() ?? new TranslatorOptions();
        // An unsupported provider is refused here, before any model resolution or download (0.67.1).
        ExecutionProviderSupport.ThrowIfUnsupported(options.Provider);

        // Parse variant qualifier (e.g., "default:fp16" → modelId="default", hint="fp16")
        var (baseId, qualifier) = LMSupplyOptionsBase.SplitQualifier(modelIdOrPath);
        options.ModelId = baseId;
        options.QuantizationHint ??= qualifier;

        var translator = new OnnxTranslatorModel(options, progress);

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
        progress = LoadProgress.Begin(progress);
        return LoadAsync(options?.ModelId ?? "default", options, progress, cancellationToken);
    }

    /// <summary>
    /// Bytes <see cref="LoadAsync(string, TranslatorOptions?, IProgress{DownloadProgress}?, CancellationToken)"/> would
    /// download for the same id and options into an empty cache: the encoder, decoder and tokenizer files that load picks
    /// (for a model published in quantized variants, the variant <see cref="LMSupplyOptionsBase.QuantizationHint"/> or the
    /// hardware tier selects, as the load decides), at the lengths the repository lists. 0 for a model on local disk.
    /// </summary>
    /// <remarks>
    /// Reads the repository listing (cached for the download that follows); downloads and loads nothing. The native ONNX
    /// Runtime a first load also provisions is not counted — <see cref="TranslatorArtifacts"/> counts it. With
    /// <see cref="TranslatorOptions.DisableAutoDownload"/> the listing comes from the cache only, as the load's does.
    /// </remarks>
    /// <param name="modelIdOrPath">Anything <see cref="LoadAsync(string, TranslatorOptions?, IProgress{DownloadProgress}?, CancellationToken)"/> accepts; a <c>:variant</c> qualifier is honoured.</param>
    /// <param name="options">The options the load will use; not modified.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <exception cref="ModelNotFoundException">The repository does not exist, or (downloads disabled) it was never listed into this cache.</exception>
    /// <exception cref="ModelDownloadException">A file the load needs is not in the repository.</exception>
    public static async Task<long> GetDownloadSizeBytesAsync(
        string modelIdOrPath, TranslatorOptions? options = null, CancellationToken cancellationToken = default)
    {
        var (info, resolved) = Prepare(modelIdOrPath, options);
        return (await OnnxTranslatorModel.PlanAsync(info, resolved, cancellationToken))?.TotalBytes ?? 0;
    }

    /// <summary>
    /// Bytes a load with the same id and options would still download now: the files of
    /// <see cref="GetDownloadSizeBytesAsync"/> the cache does not hold at the length the repository lists; 0 when cached or
    /// on local disk. A partly downloaded file counts in full. Runtimes are not counted.
    /// </summary>
    public static async Task<long> GetRemainingDownloadBytesAsync(
        string modelIdOrPath, TranslatorOptions? options = null, CancellationToken cancellationToken = default)
    {
        var (info, resolved) = Prepare(modelIdOrPath, options);
        return await OnnxTranslatorModel.PlanAsync(info, resolved, cancellationToken) is { } plan
            ? plan.GetRemainingBytes(OnnxTranslatorModel.CacheDirectory(resolved))
            : 0;
    }

    /// <summary>
    /// Whether a load with the same id and options would open cached files only: every file it picks is in the cache at the
    /// length the repository lists, or the model is on local disk. Makes no network request; never errs toward
    /// <see langword="true"/>.
    /// </summary>
    public static async Task<bool> IsModelDownloadedAsync(
        string modelIdOrPath, TranslatorOptions? options = null, CancellationToken cancellationToken = default)
    {
        var (info, resolved) = Prepare(modelIdOrPath, options);
        // Plan from the cached listing only: the check must not reach the network.
        resolved.DisableAutoDownload = true;
        try
        {
            return await OnnxTranslatorModel.PlanAsync(info, resolved, cancellationToken) is not { } plan
                   || plan.GetRemainingBytes(OnnxTranslatorModel.CacheDirectory(resolved)) == 0;
        }
        catch (Exception ex) when (ex is ModelNotFoundException or ModelDownloadException)
        {
            return false;
        }
    }

    /// <summary>
    /// Downloads the files a load with the same id and options opens, without loading anything, so a load with downloads
    /// disabled opens cached files only. Cached files are not fetched again; a model on local disk downloads nothing. The
    /// load downloads through the same path.
    /// </summary>
    /// <exception cref="ModelNotFoundException">The model is unknown, or (downloads disabled) not cached.</exception>
    public static async Task DownloadModelAsync(
        string modelIdOrPath, TranslatorOptions? options = null, IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        var (info, resolved) = Prepare(modelIdOrPath, options);
        await OnnxTranslatorModel.DownloadAsync(info, resolved, progress, cancellationToken);
    }

    /// <summary>
    /// Deletes what LMSupply downloaded of the model's repository from the cache, so the next load downloads it again —
    /// every variant it downloaded, and nothing another tool put in the shared cache. A model on local disk is kept.
    /// </summary>
    /// <returns>Whether anything was deleted.</returns>
    /// <exception cref="IOException">A file is open, for example by a loaded translator (Windows).</exception>
    public static bool DeleteModel(string modelIdOrPath, TranslatorOptions? options = null)
    {
        var (info, resolved) = Prepare(modelIdOrPath, options);
        if (OnnxTranslatorModel.LocalDirectory(info) is not null || File.Exists(info.Id) || !info.Id.Contains('/'))
            return false;

        return CacheManager.DeleteModel(OnnxTranslatorModel.CacheDirectory(resolved), info.Id);
    }

    // The registry entry a load resolves, with the options it runs under (the qualifier as its quantization hint)
    private static (TranslatorModelInfo Info, TranslatorOptions Options) Prepare(string modelIdOrPath, TranslatorOptions? options)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(modelIdOrPath);
        options = options?.Clone() ?? new TranslatorOptions();
        ExecutionProviderSupport.ThrowIfUnsupported(options.Provider);
        var (baseId, qualifier) = LMSupplyOptionsBase.SplitQualifier(modelIdOrPath);
        options.ModelId = baseId;
        options.QuantizationHint ??= qualifier;
        return (TranslatorModelRegistry.Default.Resolve(baseId), options);
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
