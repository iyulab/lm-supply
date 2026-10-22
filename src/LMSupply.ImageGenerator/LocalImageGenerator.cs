using LMSupply.Core.Download;
using LMSupply.Download;
using LMSupply.Exceptions;
using LMSupply.ImageGenerator.Models;

namespace LMSupply.ImageGenerator;

/// <summary>
/// Entry point for loading and creating local image generator models.
/// </summary>
public static class LocalImageGenerator
{
    /// <summary>
    /// Gets the model registry for the ImageGenerator domain.
    /// Provides access to model resolution, alias management, and model enumeration.
    /// </summary>
    public static IModelRegistry<ImageGeneratorModelInfo> Registry => ImageGeneratorModelRegistry.Default;

    /// <summary>
    /// Shared pool for named model management. Supports GetOrLoadAsync / UnloadAsync by model ID.
    /// </summary>
    public static LMSupply.Pool.ModelPool<IImageGeneratorModel, ImageGeneratorOptions> Pool { get; }
        = new(new Pool.ImageGeneratorLoader());

    /// <summary>
    /// Loads an image generator model.
    /// </summary>
    /// <param name="modelIdOrPath">
    /// Model identifier. Can be:
    /// - An alias: "default", "fast", "quality"
    /// - A HuggingFace repo ID: "TheyCallMeHex/LCM-Dreamshaper-V7-ONNX"
    /// - A local directory path containing ONNX model files
    /// </param>
    /// <param name="options">Optional model loading options.</param>
    /// <param name="progress">
    /// Optional progress reporting for the model and runtime downloads a first load makes — the same
    /// <see cref="DownloadProgress"/> reports every other domain's load gives.
    /// </param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Loaded image generator model.</returns>
    /// <example>
    /// <code>
    /// // Load default model
    /// await using var generator = await LocalImageGenerator.LoadAsync("default");
    ///
    /// // Generate an image
    /// var result = await generator.GenerateAsync("A sunset over mountains");
    /// await result.SaveAsync("output.png");
    /// </code>
    /// </example>
    public static async Task<IImageGeneratorModel> LoadAsync(
        string modelIdOrPath,
        ImageGeneratorOptions? options = null,
        IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        progress = LoadProgress.Begin(progress);
        ArgumentException.ThrowIfNullOrWhiteSpace(modelIdOrPath);

        options = options?.Clone() ?? new ImageGeneratorOptions();
        // An unsupported provider is refused here, before any model resolution or download (0.67.1).
        ExecutionProviderSupport.ThrowIfUnsupported(options.Provider);

        // Parse variant qualifier (e.g., "default:fp16" → modelId="default", hint="fp16")
        var (baseId, qualifier) = LMSupplyOptionsBase.SplitQualifier(modelIdOrPath);
        modelIdOrPath = baseId;
        // Note: ImageGeneratorOptions doesn't extend LMSupplyOptionsBase,
        // so qualifier is parsed but not stored (reserved for future use)
        _ = qualifier;

        // Resolve model to ImageGeneratorModelInfo via registry
        var modelInfo = ImageGeneratorModelRegistry.Default.Resolve(modelIdOrPath);
        var modelDefinition = ImageGeneratorModelRegistry.ToModelDefinition(modelInfo);

        // Determine if this is a local path or needs download
        string modelPath;

        if (IsLocalPath(modelIdOrPath))
        {
            // Use local path directly
            modelPath = modelIdOrPath;
            if (!Directory.Exists(modelPath))
            {
                throw new DirectoryNotFoundException($"Model directory not found: {modelPath}");
            }
        }
        else
        {
            // Download from HuggingFace
            modelPath = await DownloadRepositoryAsync(
                modelInfo.ModelId,
                options,
                progress,
                cancellationToken);
        }

        // Load the model
        return await OnnxImageGeneratorModel.LoadAsync(
            modelDefinition,
            modelPath,
            options,
            progress,
            cancellationToken);
    }

    /// <summary>
    /// Gets all available model aliases.
    /// </summary>
    /// <returns>Collection of valid model aliases.</returns>
    public static IReadOnlyCollection<string> GetAvailableAliases() =>
        ImageGeneratorModelRegistry.Default.GetAliases()
            .Select(a => a.Name)
            .ToList()
            .AsReadOnly();

    /// <summary>
    /// Checks if the given identifier is a local file path.
    /// </summary>
    private static bool IsLocalPath(string modelIdOrPath)
    {
        // Check for absolute path or relative path indicators
        return Path.IsPathRooted(modelIdOrPath) ||
               modelIdOrPath.StartsWith("./", StringComparison.Ordinal) ||
               modelIdOrPath.StartsWith(".\\", StringComparison.Ordinal) ||
               modelIdOrPath.StartsWith("../", StringComparison.Ordinal) ||
               modelIdOrPath.StartsWith("..\\", StringComparison.Ordinal) ||
               Directory.Exists(modelIdOrPath);
    }

    /// <summary>
    /// Bytes <see cref="LoadAsync"/> would download for the same id and options into an empty cache: the ONNX models, their
    /// external data and the config and tokenizer files discovery picks from the repository, at the lengths it lists. 0 for
    /// a model on local disk.
    /// </summary>
    /// <remarks>
    /// Reads the repository listing (cached for the download that follows); downloads and loads nothing. The native ONNX
    /// Runtime a first load also provisions is not counted — <see cref="ImageGeneratorArtifacts"/> counts it. With
    /// <see cref="ImageGeneratorOptions.DisableAutoDownload"/> the listing comes from the cache only, as the load's does.
    /// </remarks>
    /// <exception cref="ModelNotFoundException">The repository does not exist, or (downloads disabled) it was never listed into this cache.</exception>
    public static async Task<long> GetDownloadSizeBytesAsync(
        string modelIdOrPath, ImageGeneratorOptions? options = null, CancellationToken cancellationToken = default)
    {
        var (repoId, resolved) = Prepare(modelIdOrPath, options);
        return repoId is null ? 0 : (await PlanAsync(repoId, resolved, cancellationToken)).TotalBytes;
    }

    /// <summary>
    /// Bytes a load with the same id and options would still download now: the files of <see cref="GetDownloadSizeBytesAsync"/>
    /// the cache does not hold at the length the repository lists; 0 when cached or on local disk. A partly downloaded file
    /// counts in full. Runtimes are not counted.
    /// </summary>
    public static async Task<long> GetRemainingDownloadBytesAsync(
        string modelIdOrPath, ImageGeneratorOptions? options = null, CancellationToken cancellationToken = default)
    {
        var (repoId, resolved) = Prepare(modelIdOrPath, options);
        return repoId is null ? 0 : (await PlanAsync(repoId, resolved, cancellationToken)).GetRemainingBytes(CacheDirectory(resolved));
    }

    /// <summary>
    /// Whether a load with the same id and options would open cached files only: every file it picks is in the cache at the
    /// length the repository lists, or the model is on local disk. Makes no network request; never errs toward
    /// <see langword="true"/>.
    /// </summary>
    public static async Task<bool> IsModelDownloadedAsync(
        string modelIdOrPath, ImageGeneratorOptions? options = null, CancellationToken cancellationToken = default)
    {
        var (repoId, resolved) = Prepare(modelIdOrPath, options);
        if (repoId is null)
            return true;

        // Plan from the cached listing only: the check must not reach the network.
        resolved.DisableAutoDownload = true;
        try
        {
            return (await PlanAsync(repoId, resolved, cancellationToken)).GetRemainingBytes(CacheDirectory(resolved)) == 0;
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
    /// <exception cref="ModelNotFoundException">The repository does not exist, or (downloads disabled) is not cached.</exception>
    public static async Task DownloadModelAsync(
        string modelIdOrPath, ImageGeneratorOptions? options = null, IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        var (repoId, resolved) = Prepare(modelIdOrPath, options);
        if (repoId is not null)
            await DownloadRepositoryAsync(repoId, resolved, progress, cancellationToken);
    }

    /// <summary>
    /// Deletes what LMSupply downloaded of the model's repository from the cache, so the next load downloads it again — and
    /// nothing another tool put in the shared cache. A model on local disk is kept.
    /// </summary>
    /// <returns>Whether anything was deleted.</returns>
    /// <exception cref="IOException">A file is open, for example by a loaded generator (Windows).</exception>
    public static bool DeleteModel(string modelIdOrPath, ImageGeneratorOptions? options = null)
    {
        var (repoId, resolved) = Prepare(modelIdOrPath, options);
        return repoId is not null && repoId.Contains('/') && CacheManager.DeleteModel(CacheDirectory(resolved), repoId);
    }

    // The repository a load downloads (null for a model on local disk), with the options it runs under
    private static (string? RepoId, ImageGeneratorOptions Options) Prepare(string modelIdOrPath, ImageGeneratorOptions? options)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(modelIdOrPath);
        options = options?.Clone() ?? new ImageGeneratorOptions();
        ExecutionProviderSupport.ThrowIfUnsupported(options.Provider);
        var (baseId, _) = LMSupplyOptionsBase.SplitQualifier(modelIdOrPath);
        return (IsLocalPath(baseId) ? null : ImageGeneratorModelRegistry.Default.Resolve(baseId).ModelId, options);
    }

    // The files DownloadRepositoryAsync fetches, chosen by the same discovery. Downloads nothing.
    private static async Task<DownloadPlan> PlanAsync(string repoId, ImageGeneratorOptions options, CancellationToken cancellationToken)
    {
        using var downloader = new HuggingFaceDownloader(CacheDirectory(options), localFilesOnly: options.DisableAutoDownload);
        return await downloader.PlanWithDiscoveryAsync(repoId, preferences: DiscoveryPreferences, cancellationToken: cancellationToken);
    }

    private static string CacheDirectory(ImageGeneratorOptions options) =>
        options.CacheDirectory ?? CacheManager.GetDefaultCacheDirectory();

    // Discovery finds every file of the pipeline: its ONNX models, their external data, and the config and tokenizer files
    private static ModelPreferences DiscoveryPreferences => new()
    {
        QuantizationPriority = [Quantization.Default],
        PreferredProvider = ExecutionProvider.Cpu
    };

    // The download of a repository by discovery. Shared by the load and DownloadModelAsync.
    private static async Task<string> DownloadRepositoryAsync(
        string repoId,
        ImageGeneratorOptions options,
        IProgress<DownloadProgress>? progress,
        CancellationToken cancellationToken)
    {
        using var downloader = new HuggingFaceDownloader(CacheDirectory(options), localFilesOnly: options.DisableAutoDownload);
        var (modelPath, _) = await downloader.DownloadWithDiscoveryAsync(
            repoId,
            preferences: DiscoveryPreferences,
            revision: "main",
            progress: progress,
            cancellationToken: cancellationToken);

        return modelPath;
    }
}
