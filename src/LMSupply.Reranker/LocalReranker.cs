using LMSupply.Core.Download;
using LMSupply.Download;
using LMSupply.Inference;
using LMSupply.Reranker.Infrastructure;
using LMSupply.Reranker.Inference;
using LMSupply.Reranker.Models;
using LMSupply.Reranker.Utils;

namespace LMSupply.Reranker;

/// <summary>
/// Main entry point for loading and using reranker models.
/// </summary>
public static class LocalReranker
{
    /// <summary>
    /// Default model to use when no model is specified.
    /// MS-MARCO MiniLM L-6 v2, 22M params, high quality cross-encoder.
    /// </summary>
    public const string DefaultModel = "default";

    /// <summary>
    /// The quantization a GGUF repository is loaded at when it offers several. One constant, because the
    /// loader, <see cref="DownloadModelAsync"/> and <see cref="IsModelDownloaded"/> have to pick the same file.
    /// </summary>
    internal const string DefaultGgufQuantization = "Q4_K_M";

    /// <summary>
    /// The quantization a GGUF load asks for: the caller's <see cref="LMSupplyOptionsBase.QuantizationHint"/>
    /// (e.g. <c>"Q8_0"</c>) when there is one, <see cref="DefaultGgufQuantization"/> otherwise. When the
    /// repository has no such file, or it does not fit in memory, selection falls back to the largest
    /// file that fits.
    /// </summary>
    internal static string GgufQuantizationFor(RerankerOptions options)
        => string.IsNullOrWhiteSpace(options.QuantizationHint) ? DefaultGgufQuantization : options.QuantizationHint;

    /// <summary>
    /// Built-in aliases that resolve to the GGUF (llama-server) route. They live here rather than in the
    /// registry because a registry entry describes an ONNX model (graph file, tokenizer file); a GGUF
    /// model is one file served by llama-server and has neither.
    /// </summary>
    /// <remarks>
    /// <c>multilingual-fast</c> is the quantized build of the model behind <c>multilingual</c>
    /// (bge-reranker-v2-m3): the same ranking at about a fifth of the download and a fraction of the CPU
    /// latency, at the price of running a llama-server process. <c>multilingual</c> and <c>auto</c> keep
    /// resolving to ONNX, so a caller that did not ask for llama-server never gets one.
    /// </remarks>
    internal static readonly IReadOnlyDictionary<string, string> GgufAliases =
        new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
        {
            ["multilingual-fast"] = "gguf:gpustack/bge-reranker-v2-m3-GGUF",
        };

    /// <summary>
    /// Gets the model registry for the Reranker domain.
    /// Provides access to model resolution, alias management, and model enumeration.
    /// </summary>
    public static IModelRegistry<ModelInfo> Registry => RerankerModelRegistry.Default;

    /// <summary>
    /// Shared pool for named model management. Supports GetOrLoadAsync / UnloadAsync by model ID.
    /// </summary>
    public static LMSupply.Pool.ModelPool<IRerankerModel, RerankerOptions> Pool { get; }
        = new(new Pool.RerankerLoader());

    /// <summary>
    /// Loads the default reranker model.
    /// </summary>
    /// <param name="options">Optional configuration options.</param>
    /// <param name="progress">Optional progress reporting for downloads.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>A loaded reranker ready for inference.</returns>
    public static Task<IRerankerModel> LoadDefaultAsync(
        RerankerOptions? options = null,
        IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        return LoadAsync(DefaultModel, options, progress, cancellationToken);
    }

    /// <summary>
    /// Loads a reranker model by name or path.
    /// </summary>
    /// <param name="modelIdOrPath">
    /// Either a model alias (e.g., "default", "quality", "fast", or "multilingual-fast" for the GGUF route),
    /// a HuggingFace model ID (e.g., "cross-encoder/ms-marco-MiniLM-L-6-v2"),
    /// a local path to an ONNX model file,
    /// or a GGUF model (prefix with "gguf:" or use repo ending in "-GGUF").
    /// <para>
    /// <b>GGUF compatibility:</b> Only traditional cross-encoder models are supported
    /// (e.g., gpustack/bge-reranker-v2-m3-GGUF, gpustack/jina-reranker-v1-turbo-en-GGUF).
    /// Generative rerankers (e.g., Qwen3-Reranker) that require prompt-based "yes/no"
    /// scoring are NOT compatible with llama-server's --pooling rank mode and will
    /// produce near-zero garbage scores.
    /// </para>
    /// </param>
    /// <param name="options">Optional configuration options.</param>
    /// <param name="progress">Optional progress reporting for downloads.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>A loaded reranker ready for inference.</returns>
    public static async Task<IRerankerModel> LoadAsync(
        string modelIdOrPath,
        RerankerOptions? options = null,
        IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        options ??= new RerankerOptions();
        // An unsupported provider is refused here, before any model resolution or download (0.67.1).
        ExecutionProviderSupport.ThrowIfUnsupported(options.Provider);

        // Parse variant qualifier (e.g., "default:fp16" → modelId="default", hint="fp16")
        var (baseId, qualifier) = LMSupplyOptionsBase.SplitQualifier(modelIdOrPath);
        options.ModelId = baseId;
        options.QuantizationHint ??= qualifier;

        // Alias translation precedes format detection: the gguf check below must see the TARGET
        // (e.g. "my-rerank" -> "gguf:..." or "multilingual-fast" must enter the GGUF path).
        if (TryFollowAlias(baseId, out var aliasTarget))
        {
            modelIdOrPath = aliasTarget;
            options.ModelId = aliasTarget;
        }

        // Check for GGUF format
        if (IsGgufModel(modelIdOrPath))
        {
            return await LoadGgufAsync(modelIdOrPath, options, progress, cancellationToken);
        }

        var reranker = new Reranker(options);

        // Eagerly initialize and warm up the model
        await reranker.WarmupAsync(cancellationToken);

        return reranker;
    }

    /// <summary>
    /// Checks if the model identifier refers to a GGUF model.
    /// </summary>
    private static bool IsGgufModel(string modelIdOrPath)
    {
        // Check for "gguf:" prefix
        if (modelIdOrPath.StartsWith("gguf:", StringComparison.OrdinalIgnoreCase))
            return true;

        // Check for .gguf extension
        if (modelIdOrPath.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase))
            return true;

        // Check if local file exists and has .gguf extension
        if (File.Exists(modelIdOrPath) &&
            modelIdOrPath.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase))
            return true;

        // Check for GGUF indicators in HuggingFace repo name
        var lowerPath = modelIdOrPath.ToLowerInvariant();
        if (lowerPath.Contains("-gguf") || lowerPath.Contains("_gguf"))
            return true;

        return false;
    }

    private static string StripGgufPrefix(string modelIdOrPath)
        => modelIdOrPath.StartsWith("gguf:", StringComparison.OrdinalIgnoreCase)
            ? modelIdOrPath[5..]
            : modelIdOrPath;

    /// <summary>
    /// What <see cref="LoadAsync"/> does to an id before it chooses a backend, for the entry points that
    /// have to choose the same one: a user alias is followed to its target first, because the target is
    /// what says whether this is a GGUF model.
    /// </summary>
    private static string FollowUserAlias(string modelId)
    {
        var (baseId, _) = LMSupplyOptionsBase.SplitQualifier(modelId);
        return TryFollowAlias(baseId, out var target) ? target : modelId;
    }

    /// <summary>
    /// A user alias first — so a caller can point a built-in GGUF alias name somewhere else — then the
    /// built-in GGUF aliases, applied to whatever the user alias gave.
    /// </summary>
    private static bool TryFollowAlias(string baseId, out string target)
    {
        var followed = RerankerModelRegistry.Default.TryGetUserAliasTarget(baseId, out var userTarget);
        target = followed ? userTarget! : baseId;

        if (GgufAliases.TryGetValue(target, out var ggufTarget))
        {
            target = ggufTarget;
            followed = true;
        }

        return followed;
    }

    /// <summary>
    /// Loads a GGUF reranker model.
    /// </summary>
    private static async Task<IRerankerModel> LoadGgufAsync(
        string modelIdOrPath,
        RerankerOptions options,
        IProgress<DownloadProgress>? progress,
        CancellationToken cancellationToken)
    {
        string modelPath;
        string modelId;

        var cleanPath = StripGgufPrefix(modelIdOrPath);

        // Check if it's a local file
        if (File.Exists(cleanPath))
        {
            modelPath = cleanPath;
            modelId = Path.GetFileNameWithoutExtension(modelPath);
        }
        // Check if it's a HuggingFace repo ID
        else if (cleanPath.Contains('/'))
        {
            var cacheDir = options.CacheDirectory ?? CacheManager.GetDefaultCacheDirectory();

            // Download the GGUF file from HuggingFace
            using var downloader = new GgufDownloader(cacheDir, localFilesOnly: options.DisableAutoDownload);
            modelPath = await downloader.DownloadAsync(
                cleanPath,
                preferredQuantization: GgufQuantizationFor(options),
                progress: progress,
                cancellationToken: cancellationToken);

            modelId = cleanPath.Split('/').Last();
        }
        else
        {
            throw new ModelNotFoundException(
                $"GGUF reranker model not found: '{modelIdOrPath}'. " +
                "Provide a local path to a .gguf file or a HuggingFace repo ID " +
                "(e.g., 'gguf:gpustack/bge-reranker-v2-m3-GGUF').",
                modelIdOrPath);
        }

        return await LlamaServerRerankerModel.LoadAsync(
            modelId,
            modelPath,
            options,
            progress,
            cancellationToken);
    }

    /// <summary>
    /// Checks whether the ONNX Runtime native library can be loaded in the current environment.
    /// Call this before <see cref="LoadAsync"/> to verify that the host has the required
    /// shared libraries (e.g., libstdc++, libgomp on Linux; VC++ Redistributable on Windows).
    /// </summary>
    /// <returns>
    /// A tuple where <c>Available</c> is <see langword="true"/> when the runtime can be loaded,
    /// and <c>ErrorMessage</c> contains a diagnostic string when it cannot.
    /// </returns>
    public static (bool Available, string? ErrorMessage) CheckRuntimeAvailability()
        => OnnxSessionFactory.CheckOnnxRuntimeAvailability();

    /// <summary>
    /// Checks whether a model is already downloaded and available in the local cache.
    /// This does NOT load the model into memory or initialize the ONNX Runtime.
    /// </summary>
    /// <remarks>
    /// The answer is the one a load with <see cref="RerankerOptions.DisableAutoDownload"/> would act on:
    /// <see langword="true"/> exactly when that load finds its file. That holds for GGUF ids too — the
    /// probe looks where the GGUF loader looks and picks the quantization the loader picks.
    /// </remarks>
    /// <param name="modelId">
    /// A model alias (e.g., "default"), a known model ID, a HuggingFace repo ID, or a GGUF model in any
    /// form <see cref="LoadAsync"/> accepts (<c>gguf:org/repo</c>, a <c>*-GGUF</c> repo, a local <c>.gguf</c> path).
    /// </param>
    /// <param name="cacheDirectory">Custom cache directory, or <see langword="null"/> for default.</param>
    /// <returns><see langword="true"/> if the model files exist in cache and are not LFS pointers.</returns>
    public static bool IsModelDownloaded(string modelId, string? cacheDirectory = null)
    {
        var cacheDir = cacheDirectory ?? CacheManager.GetDefaultCacheDirectory();

        modelId = FollowUserAlias(modelId);
        if (IsGgufModel(modelId))
        {
            var ggufTarget = StripGgufPrefix(modelId);
            if (File.Exists(ggufTarget))
                return CacheManager.IsCachedFile(ggufTarget);

            return ggufTarget.Contains('/')
                && GgufDownloader.TrySelectFromLocalCache(cacheDir, ggufTarget, DefaultGgufQuantization) != null;
        }

        // Resolve alias to model info
        var registry = RerankerModelRegistry.Default;
        ModelInfo modelInfo;
        try
        {
            modelInfo = registry.Resolve(modelId);
        }
        catch (ModelNotFoundException)
        {
            // Unknown model — check raw repo ID
            var snapshotDir = CacheManager.GetModelDirectory(cacheDir, modelId);
            var onnxPath = Path.Combine(snapshotDir, "model.onnx");
            return File.Exists(onnxPath) && !CacheManager.IsLfsPointerFile(onnxPath);
        }

        using var manager = new ModelManager(cacheDir, autoDownload: false);
        return manager.GetCachedModel(modelInfo) != null;
    }

    /// <summary>
    /// Downloads a model without loading it into memory.
    /// If the model is already cached, this is a no-op.
    /// Use this to pre-fetch models (e.g., during container build or CI) without requiring
    /// the ONNX Runtime to be available at download time.
    /// </summary>
    /// <param name="modelId">
    /// A model alias (e.g., "default"), a known model ID, or a HuggingFace repo ID.
    /// </param>
    /// <param name="options">Optional configuration (cache directory).</param>
    /// <param name="progress">Optional progress reporting for downloads.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    public static async Task DownloadModelAsync(
        string modelId,
        RerankerOptions? options = null,
        IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        options ??= new RerankerOptions();
        // An unsupported provider is refused here, before any model resolution or download (0.67.1).
        ExecutionProviderSupport.ThrowIfUnsupported(options.Provider);

        // Parse variant qualifier
        var (baseId, qualifier) = LMSupplyOptionsBase.SplitQualifier(modelId);
        options.ModelId = baseId;
        options.QuantizationHint ??= qualifier;

        var cacheDir = options.CacheDirectory ?? CacheManager.GetDefaultCacheDirectory();
        var registry = RerankerModelRegistry.Default;

        // A GGUF id is fetched the way the GGUF loader fetches it - one file, at the loader's quantization,
        // into the loader's cache layout. The raw-repository branch below would pull every quantization of
        // the repository into a layout the loader never reads.
        var ggufCandidate = FollowUserAlias(modelId);
        if (IsGgufModel(ggufCandidate))
        {
            var ggufTarget = StripGgufPrefix(ggufCandidate);
            if (File.Exists(ggufTarget))
                return;

            if (!ggufTarget.Contains('/'))
            {
                throw new ModelNotFoundException(
                    $"GGUF reranker model not found: '{modelId}'. " +
                    "Provide a local path to a .gguf file or a HuggingFace repo ID " +
                    "(e.g., 'gguf:gpustack/bge-reranker-v2-m3-GGUF').",
                    modelId);
            }

            using var ggufDownloader = new GgufDownloader(cacheDir, localFilesOnly: options.DisableAutoDownload);
            await ggufDownloader.DownloadAsync(ggufTarget, GgufQuantizationFor(options), progress, cancellationToken);
            return;
        }

        ModelInfo modelInfo;
        try
        {
            modelInfo = registry.Resolve(options.ModelId);
        }
        catch (ModelNotFoundException)
        {
            // Unknown alias — try as raw HuggingFace repo ID
            if (options.ModelId.Contains('/'))
            {
                using var downloader = new HuggingFaceDownloader(cacheDir);
                await downloader.DownloadModelAsync(
                    options.ModelId,
                    progress: progress,
                    cancellationToken: cancellationToken);
                return;
            }
            throw;
        }

        using var manager = new ModelManager(cacheDir, autoDownload: true);
        await manager.EnsureModelAsync(modelInfo, progress, cancellationToken);
    }

    /// <summary>
    /// Gets a list of pre-configured model aliases available for use.
    /// </summary>
    /// <returns>Available model aliases.</returns>
    public static IEnumerable<string> GetAvailableModels()
    {
        return RerankerModelRegistry.Default.GetAliases().Select(a => a.Name)
            .Concat(GgufAliases.Keys)
            .Distinct(StringComparer.OrdinalIgnoreCase);
    }

    /// <summary>
    /// Gets all registered model information.
    /// </summary>
    /// <returns>Collection of model information.</returns>
    public static IEnumerable<ModelInfo> GetAllModels()
    {
        return RerankerModelRegistry.Default.GetAvailableModels();
    }
}
