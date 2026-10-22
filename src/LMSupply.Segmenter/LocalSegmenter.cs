using LMSupply.Segmenter.Core;
using LMSupply.Segmenter.Interactive;
using LMSupply.Segmenter.Models;
using LMSupply.Download;
using LMSupply.Exceptions;

namespace LMSupply.Segmenter;

/// <summary>
/// Main entry point for loading and using image segmentation models.
/// </summary>
public static class LocalSegmenter
{
    /// <summary>
    /// Gets the model registry for the Segmenter domain.
    /// Provides access to model resolution, alias management, and model enumeration.
    /// </summary>
    public static IModelRegistry<SegmenterModelInfo> Registry => SegmenterModelRegistry.Default;

    /// <summary>
    /// Shared pool for named model management. Supports GetOrLoadAsync / UnloadAsync by model ID.
    /// </summary>
    public static LMSupply.Pool.ModelPool<ISegmenterModel, SegmenterOptions> Pool { get; }
        = new(new Pool.SegmenterLoader());

    /// <summary>
    /// Loads an image segmentation model by name or path.
    /// </summary>
    /// <param name="modelIdOrPath">
    /// Either a model alias (e.g., "default", "quality", "fast"),
    /// a HuggingFace model ID (e.g., "nvidia/segformer-b0-finetuned-ade-512-512"),
    /// or a local path to an ONNX model file.
    /// </param>
    /// <param name="options">Optional configuration options.</param>
    /// <param name="progress">Optional progress reporting for downloads.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>A loaded segmenter ready for inference.</returns>
    public static async Task<ISegmenterModel> LoadAsync(
        string modelIdOrPath,
        SegmenterOptions? options = null,
        IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        progress = LoadProgress.Begin(progress);
        options = options?.Clone() ?? new SegmenterOptions();
        // An unsupported provider is refused here, before any model resolution or download (0.67.1).
        ExecutionProviderSupport.ThrowIfUnsupported(options.Provider);

        // Parse variant qualifier (e.g., "default:fp16" → modelId="default", hint="fp16")
        var (baseId, qualifier) = LMSupplyOptionsBase.SplitQualifier(modelIdOrPath);
        options.ModelId = baseId;
        options.QuantizationHint ??= qualifier;

        if (SegmenterModelRegistry.Default.TryResolve(options.ModelId, out var resolved) && resolved is { IsInteractive: true })
        {
            throw new ArgumentException(
                $"'{modelIdOrPath}' is a prompt-based (interactive) model; load it with LocalSegmenter.LoadInteractiveAsync.",
                nameof(modelIdOrPath));
        }

        var segmenter = new OnnxSegmenterModel(options, progress);

        // Eagerly initialize and warm up the model
        await segmenter.WarmupAsync(cancellationToken);

        return segmenter;
    }

    /// <summary>
    /// Loads a prompt-based (interactive) segmentation model — segment what a point or box points at — and warms it up.
    /// </summary>
    /// <param name="modelIdOrAlias">An interactive model alias; <c>"interactive"</c> (MobileSAM) is the one registered.</param>
    /// <param name="options">Optional configuration options (provider, cache directory, downloads).</param>
    /// <param name="progress">Optional progress reporting for the model and runtime downloads a first load makes.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>A segmenter whose <see cref="IInteractiveSegmenter.CreateSessionAsync(string, CancellationToken)"/> encodes an image once for many prompts.</returns>
    /// <exception cref="ArgumentException">The id does not name an interactive model.</exception>
    public static async Task<IInteractiveSegmenter> LoadInteractiveAsync(
        string modelIdOrAlias = "interactive",
        SegmenterOptions? options = null,
        IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        progress = LoadProgress.Begin(progress);
        options = options?.Clone() ?? new SegmenterOptions();
        ExecutionProviderSupport.ThrowIfUnsupported(options.Provider);

        var modelInfo = SegmenterModelRegistry.Default.Resolve(modelIdOrAlias);
        if (!modelInfo.IsInteractive)
        {
            throw new ArgumentException(
                $"'{modelIdOrAlias}' is not an interactive model; load it with LocalSegmenter.LoadAsync.", nameof(modelIdOrAlias));
        }

        options.ModelId = modelInfo.Id;
        var segmenter = new MobileSamModel(options, modelInfo, progress);
        try
        {
            await segmenter.WarmupAsync(cancellationToken);
        }
        catch
        {
            await segmenter.DisposeAsync();
            throw;
        }

        return segmenter;
    }

    /// <summary>
    /// Loads the default image segmentation model.
    /// </summary>
    /// <param name="options">Optional configuration options.</param>
    /// <param name="progress">Optional progress reporting for downloads.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>A loaded segmenter ready for inference.</returns>
    public static Task<ISegmenterModel> LoadAsync(
        SegmenterOptions? options = null,
        IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        progress = LoadProgress.Begin(progress);
        return LoadAsync(options?.ModelId ?? "default", options, progress, cancellationToken);
    }

    /// <summary>
    /// Bytes a load of <paramref name="modelIdOrPath"/> with the same options would download into an empty cache: the files
    /// that load takes — for a segmentation model the file it picks from the repository, for an interactive one its encoder,
    /// decoder and configuration — at the lengths the repository lists. 0 for a model on local disk. Reads the repository
    /// listing (cached for the download that follows); downloads nothing. The native ONNX Runtime a first load also
    /// provisions is not counted — <see cref="SegmenterArtifacts"/> counts it.
    /// </summary>
    /// <exception cref="ModelNotFoundException">The repository does not exist, or (downloads disabled) it was never listed into this cache.</exception>
    public static async Task<long> GetDownloadSizeBytesAsync(string modelIdOrPath, SegmenterOptions? options = null, CancellationToken cancellationToken = default)
        => (await PlanAsync(modelIdOrPath, options, cancellationToken)).Plan?.TotalBytes ?? 0;

    /// <summary>
    /// Bytes a load with the same id and options would still download now: the files of <see cref="GetDownloadSizeBytesAsync"/>
    /// the cache does not hold at the length the repository lists; 0 when cached or on local disk. A partly downloaded file
    /// counts in full. Runtimes are not counted.
    /// </summary>
    public static async Task<long> GetRemainingDownloadBytesAsync(string modelIdOrPath, SegmenterOptions? options = null, CancellationToken cancellationToken = default)
        => await PlanAsync(modelIdOrPath, options, cancellationToken) is { Plan: { } plan } planned ? plan.GetRemainingBytes(planned.CacheDirectory) : 0;

    /// <summary>
    /// Whether a load with the same id and options would open cached files only: the files it takes are in the cache at the
    /// lengths the repository lists, or the model is on local disk. Makes no network request; never errs toward true.
    /// </summary>
    public static async Task<bool> IsModelDownloadedAsync(string modelIdOrPath, SegmenterOptions? options = null, CancellationToken cancellationToken = default)
    {
        options = options?.Clone() ?? new SegmenterOptions();
        options.DisableAutoDownload = true;
        try
        {
            return await PlanAsync(modelIdOrPath, options, cancellationToken) is not { Plan: { } plan } planned
                   || plan.GetRemainingBytes(planned.CacheDirectory) == 0;
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
        string modelIdOrPath, SegmenterOptions? options = null, IProgress<DownloadProgress>? progress = null, CancellationToken cancellationToken = default)
    {
        var (info, resolved) = Prepare(modelIdOrPath, options);
        if (info.IsInteractive)
        {
            if (OnLocalDisk(info.Id))
                return;
            using var downloader = new HuggingFaceDownloader(resolved.CacheDirectory, localFilesOnly: resolved.DisableAutoDownload);
            await downloader.DownloadModelAsync(info.Id, files: InteractiveFiles(info), progress: progress, cancellationToken: cancellationToken);
            return;
        }

        using var resolver = new LMSupply.Core.Download.ModelPathResolver(resolved.CacheDirectory, localFilesOnly: resolved.DisableAutoDownload);
        await resolver.ResolveModelAsync(info.Id, expectedOnnxFile: info.OnnxFile, progress: progress, cancellationToken: cancellationToken);
    }

    /// <summary>
    /// Deletes what LMSupply downloaded of the model's repository from the cache, so the next load downloads it again. A model
    /// on local disk, and files another tool put in the shared cache, are kept.
    /// </summary>
    /// <returns>Whether anything was deleted.</returns>
    /// <exception cref="IOException">A file is open, for example by a loaded segmenter (Windows).</exception>
    public static bool DeleteModel(string modelIdOrPath, SegmenterOptions? options = null)
    {
        var (info, resolved) = Prepare(modelIdOrPath, options);
        var repoId = LMSupply.Core.Download.ModelPathResolver.StripVariantSuffix(info.Id);
        if (OnLocalDisk(repoId) || !repoId.Contains('/'))
            return false;

        return CacheManager.DeleteModel(resolved.CacheDirectory ?? CacheManager.GetDefaultCacheDirectory(), repoId);
    }

    // The registry entry a load resolves, with the options it runs under
    private static (SegmenterModelInfo Info, SegmenterOptions Options) Prepare(string modelIdOrPath, SegmenterOptions? options)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(modelIdOrPath);
        options = options?.Clone() ?? new SegmenterOptions();
        ExecutionProviderSupport.ThrowIfUnsupported(options.Provider);
        var (baseId, _) = LMSupplyOptionsBase.SplitQualifier(modelIdOrPath);
        return (SegmenterModelRegistry.Default.Resolve(baseId), options);
    }

    // An interactive model's id names a local directory, or a file in one, when it is on disk — as its load reads it
    private static bool OnLocalDisk(string id) =>
        File.Exists(id) || Directory.Exists(id) || (Path.GetDirectoryName(id) is { Length: > 0 } parent && Directory.Exists(parent));

    private static string[] InteractiveFiles(SegmenterModelInfo info) => [info.EncoderFile!, info.DecoderFile!, "config.json"];

    private static async Task<(DownloadPlan? Plan, string CacheDirectory)> PlanAsync(string modelIdOrPath, SegmenterOptions? options, CancellationToken cancellationToken)
    {
        var (info, resolved) = Prepare(modelIdOrPath, options);
        var cacheDirectory = resolved.CacheDirectory ?? CacheManager.GetDefaultCacheDirectory();
        if (info.IsInteractive)
        {
            if (OnLocalDisk(info.Id))
                return (null, cacheDirectory);
            using var downloader = new HuggingFaceDownloader(resolved.CacheDirectory, localFilesOnly: resolved.DisableAutoDownload);
            return (await downloader.PlanModelAsync(info.Id, InteractiveFiles(info), cancellationToken: cancellationToken), cacheDirectory);
        }

        using var resolver = new LMSupply.Core.Download.ModelPathResolver(resolved.CacheDirectory, localFilesOnly: resolved.DisableAutoDownload);
        return (await resolver.PlanModelAsync(info.Id, info.OnnxFile, cancellationToken: cancellationToken), cacheDirectory);
    }

    /// <summary>
    /// Gets a list of pre-configured model aliases available for use.
    /// </summary>
    /// <returns>Available model aliases.</returns>
    public static IEnumerable<string> GetAvailableModels()
    {
        return SegmenterModelRegistry.Default.GetAliases().Select(a => a.Name);
    }

    /// <summary>
    /// Gets all registered model information.
    /// </summary>
    /// <returns>Collection of model information.</returns>
    public static IEnumerable<SegmenterModelInfo> GetAllModels()
    {
        return SegmenterModelRegistry.Default.GetAvailableModels();
    }

    /// <summary>
    /// Gets the ADE20K class labels.
    /// </summary>
    public static IReadOnlyList<string> Ade20kClassLabels => Ade20kLabels.Labels;
}
