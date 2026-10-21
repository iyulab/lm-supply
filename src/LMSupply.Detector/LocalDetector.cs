using LMSupply.Detector.Core;
using LMSupply.Detector.Models;
using LMSupply.Download;
using LMSupply.Exceptions;

namespace LMSupply.Detector;

/// <summary>
/// Main entry point for loading and using object detection models.
/// </summary>
public static class LocalDetector
{
    /// <summary>
    /// Gets the model registry for the Detector domain.
    /// Provides access to model resolution, alias management, and model enumeration.
    /// </summary>
    public static IModelRegistry<DetectorModelInfo> Registry => DetectorModelRegistry.Default;

    /// <summary>
    /// Shared pool for named model management. Supports GetOrLoadAsync / UnloadAsync by model ID.
    /// </summary>
    public static LMSupply.Pool.ModelPool<IDetectorModel, DetectorOptions> Pool { get; }
        = new(new Pool.DetectorLoader());

    /// <summary>
    /// Loads an object detection model by name or path.
    /// </summary>
    /// <param name="modelIdOrPath">
    /// Either a model alias (e.g., "default", "quality", "fast", "face", "pose"),
    /// a HuggingFace model ID (e.g., "PekingU/rtdetr_r18vd"),
    /// or a local path to an ONNX model file.
    /// </param>
    /// <param name="options">Optional configuration options.</param>
    /// <param name="progress">Optional progress reporting for downloads.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>A loaded detector ready for inference.</returns>
    public static async Task<IDetectorModel> LoadAsync(
        string modelIdOrPath,
        DetectorOptions? options = null,
        IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        progress = LoadProgress.Begin(progress);
        options = options?.Clone() ?? new DetectorOptions();
        // An unsupported provider is refused here, before any model resolution or download (0.67.1).
        ExecutionProviderSupport.ThrowIfUnsupported(options.Provider);

        // Parse variant qualifier (e.g., "default:fp16" → modelId="default", hint="fp16")
        var (baseId, qualifier) = LMSupplyOptionsBase.SplitQualifier(modelIdOrPath);
        options.ModelId = baseId;
        options.QuantizationHint ??= qualifier;

        var detector = new OnnxDetectorModel(options, progress);

        // Eagerly initialize and warm up the model
        await detector.WarmupAsync(cancellationToken);

        return detector;
    }

    /// <summary>
    /// Loads the default object detection model.
    /// </summary>
    /// <param name="options">Optional configuration options.</param>
    /// <param name="progress">Optional progress reporting for downloads.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>A loaded detector ready for inference.</returns>
    public static Task<IDetectorModel> LoadAsync(
        DetectorOptions? options = null,
        IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        progress = LoadProgress.Begin(progress);
        return LoadAsync(options?.ModelId ?? "default", options, progress, cancellationToken);
    }

    /// <summary>
    /// Bytes <see cref="LoadAsync(string, DetectorOptions?, IProgress{DownloadProgress}?, CancellationToken)"/> would download
    /// for the same id and options into an empty cache: the model file that load picks, at the length the repository lists.
    /// 0 for a model on local disk. Reads the repository listing (cached for the download that follows); downloads nothing.
    /// The native ONNX Runtime a first load also provisions is not counted — <see cref="DetectorArtifacts"/> counts it.
    /// </summary>
    /// <exception cref="ModelNotFoundException">The repository does not exist, or (downloads disabled) it was never listed into this cache.</exception>
    public static async Task<long> GetDownloadSizeBytesAsync(string modelIdOrPath, DetectorOptions? options = null, CancellationToken cancellationToken = default)
        => (await PlanAsync(modelIdOrPath, options, cancellationToken))?.Plan?.TotalBytes ?? 0;

    /// <summary>
    /// Bytes a load with the same id and options would still download now: the files of <see cref="GetDownloadSizeBytesAsync"/>
    /// the cache does not hold at the length the repository lists; 0 when cached or on local disk. A partly downloaded file
    /// counts in full. Runtimes are not counted.
    /// </summary>
    public static async Task<long> GetRemainingDownloadBytesAsync(string modelIdOrPath, DetectorOptions? options = null, CancellationToken cancellationToken = default)
        => await PlanAsync(modelIdOrPath, options, cancellationToken) is { Plan: { } plan } planned ? plan.GetRemainingBytes(planned.CacheDirectory) : 0;

    /// <summary>
    /// Whether a load with the same id and options would open cached files only: the file it picks is in the cache at the
    /// length the repository lists, or the model is on local disk. Makes no network request; never errs toward true.
    /// </summary>
    public static async Task<bool> IsModelDownloadedAsync(string modelIdOrPath, DetectorOptions? options = null, CancellationToken cancellationToken = default)
    {
        options = options?.Clone() ?? new DetectorOptions();
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
    /// Downloads the file a load with the same id and options opens, without loading anything, so a load with downloads
    /// disabled opens cached files only. Cached files are not fetched again; a model on local disk downloads nothing. The
    /// load downloads through the same path.
    /// </summary>
    /// <exception cref="ModelNotFoundException">The model is unknown, or (downloads disabled) not cached.</exception>
    public static async Task DownloadModelAsync(
        string modelIdOrPath, DetectorOptions? options = null, IProgress<DownloadProgress>? progress = null, CancellationToken cancellationToken = default)
    {
        var (info, resolved) = Prepare(modelIdOrPath, options);
        using var resolver = new LMSupply.Core.Download.ModelPathResolver(resolved.CacheDirectory, localFilesOnly: resolved.DisableAutoDownload);
        await resolver.ResolveModelAsync(info.Id, expectedOnnxFile: info.OnnxFile, progress: progress, cancellationToken: cancellationToken);
    }

    /// <summary>
    /// Deletes what LMSupply downloaded of the model's repository from the cache, so the next load downloads it again. A model
    /// on local disk, and files another tool put in the shared cache, are kept.
    /// </summary>
    /// <returns>Whether anything was deleted.</returns>
    /// <exception cref="IOException">A file is open, for example by a loaded detector (Windows).</exception>
    public static bool DeleteModel(string modelIdOrPath, DetectorOptions? options = null)
    {
        var (info, resolved) = Prepare(modelIdOrPath, options);
        var repoId = LMSupply.Core.Download.ModelPathResolver.StripVariantSuffix(info.Id);
        if (File.Exists(repoId) || Directory.Exists(repoId) || !repoId.Contains('/'))
            return false;

        return CacheManager.DeleteModel(resolved.CacheDirectory ?? CacheManager.GetDefaultCacheDirectory(), repoId);
    }

    // The registry entry a load resolves, with the options it runs under
    private static (DetectorModelInfo Info, DetectorOptions Options) Prepare(string modelIdOrPath, DetectorOptions? options)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(modelIdOrPath);
        options = options?.Clone() ?? new DetectorOptions();
        ExecutionProviderSupport.ThrowIfUnsupported(options.Provider);
        var (baseId, _) = LMSupplyOptionsBase.SplitQualifier(modelIdOrPath);
        return (DetectorModelRegistry.Default.Resolve(baseId), options);
    }

    private static async Task<(DownloadPlan? Plan, string CacheDirectory)?> PlanAsync(string modelIdOrPath, DetectorOptions? options, CancellationToken cancellationToken)
    {
        var (info, resolved) = Prepare(modelIdOrPath, options);
        using var resolver = new LMSupply.Core.Download.ModelPathResolver(resolved.CacheDirectory, localFilesOnly: resolved.DisableAutoDownload);
        var plan = await resolver.PlanModelAsync(info.Id, info.OnnxFile, cancellationToken: cancellationToken);
        return (plan, resolved.CacheDirectory ?? CacheManager.GetDefaultCacheDirectory());
    }

    /// <summary>
    /// Gets a list of pre-configured model aliases available for use.
    /// </summary>
    /// <returns>Available model aliases.</returns>
    public static IEnumerable<string> GetAvailableModels()
    {
        return DetectorModelRegistry.Default.GetAliases().Select(a => a.Name);
    }

    /// <summary>
    /// Gets all registered model information.
    /// </summary>
    /// <returns>Collection of model information.</returns>
    public static IEnumerable<DetectorModelInfo> GetAllModels()
    {
        return DetectorModelRegistry.Default.GetAvailableModels();
    }

    /// <summary>
    /// Gets the COCO class labels.
    /// </summary>
    public static IReadOnlyList<string> CocoClassLabels => CocoLabels.Labels;
}
