using LMSupply.Core.Download;
using LMSupply.Download;
using LMSupply.Embedder.Inference;
using LMSupply.Embedder.Pooling;
using LMSupply.Embedder.Utils;
using LMSupply.Hardware;
using LMSupply.Inference;
using LMSupply.Text;

namespace LMSupply.Embedder;

/// <summary>
/// Main entry point for loading and using embedding models.
/// </summary>
public static class LocalEmbedder
{
    /// <summary>
    /// Default model to use when no model is specified.
    /// BGE Small English v1.5, 33M params, MTEB top performer.
    /// </summary>
    public const string DefaultModel = "default";

    /// <summary>
    /// Gets the model registry for embedding models.
    /// Supports system aliases, user aliases, and model resolution.
    /// </summary>
    public static IModelRegistry<ModelInfo> Registry => EmbedderModelRegistry.Default;

    /// <summary>
    /// Shared pool for named model management. Supports GetOrLoadAsync / UnloadAsync by model ID.
    /// </summary>
    public static LMSupply.Pool.ModelPool<IEmbeddingModel, EmbedderOptions> Pool { get; }
        = new(new Pool.EmbedderLoader());

    /// <summary>
    /// Loads the default embedding model.
    /// </summary>
    /// <param name="options">Optional configuration options.</param>
    /// <param name="progress">Optional progress reporting for downloads.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>A loaded embedding model ready for inference.</returns>
    public static Task<IEmbeddingModel> LoadDefaultAsync(
        EmbedderOptions? options = null,
        IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        return LoadAsync(DefaultModel, options, progress, cancellationToken);
    }

    /// <summary>
    /// Loads an embedding model by name or path.
    /// </summary>
    /// <param name="modelIdOrPath">
    /// Either a catalog alias (e.g., "default", "fast", "large") or model ID for auto-download,
    /// or a local path to an ONNX/GGUF model file.
    /// GGUF models are auto-detected by "-GGUF"/"_gguf" in repo name or ".gguf" extension.
    /// </param>
    /// <param name="options">Optional configuration options.</param>
    /// <param name="progress">Optional progress reporting for downloads.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>A loaded embedding model ready for inference.</returns>
    public static async Task<IEmbeddingModel> LoadAsync(
        string modelIdOrPath,
        EmbedderOptions? options = null,
        IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        options ??= new EmbedderOptions();

        // Parse variant qualifier (e.g., "default:fp16" → modelId="default", hint="fp16")
        var (baseId, qualifier) = LMSupplyOptionsBase.SplitQualifier(modelIdOrPath);
        modelIdOrPath = baseId;
        options.QuantizationHint ??= qualifier;

        // User alias translation precedes format detection: the gguf/path checks below
        // must see the TARGET (e.g. "my-embed" -> "gguf:..." must enter the GGUF path).
        if (EmbedderModelRegistry.Default.TryGetUserAliasTarget(modelIdOrPath, out var userAliasTarget))
        {
            modelIdOrPath = userAliasTarget!;
        }

        // Check for GGUF format
        if (IsGgufModel(modelIdOrPath))
        {
            return await LoadGgufAsync(modelIdOrPath, options, progress, cancellationToken);
        }

        string modelPath;
        // Directories to probe for tokenizer files (model dir first, then repo root if different).
        // The first entry must contain model.onnx; subsequent entries are fall-back search roots
        // for tokenizer assets that HuggingFace repos commonly place at the repo root rather than
        // inside the onnx/ subfolder (e.g. tokenizer.json, sentencepiece.bpe.model).
        string tokenizerPrimaryDir;
        string? tokenizerFallbackDir;
        string modelId;

        ModelInfo? loadedModelInfo = null;

        // Check if it's a local path
        if (File.Exists(modelIdOrPath) || modelIdOrPath.EndsWith(".onnx", StringComparison.OrdinalIgnoreCase))
        {
            modelPath = modelIdOrPath;
            modelId = Path.GetFileNameWithoutExtension(modelPath);

            tokenizerPrimaryDir = Path.GetDirectoryName(modelPath) ?? ".";
            tokenizerFallbackDir = null;
        }
        // Check if it's a known model ID
        else if (EmbedderModelRegistry.Default.TryResolve(modelIdOrPath, out var modelInfo))
        {
            loadedModelInfo = modelInfo;

            // Apply model-specific defaults
            if (options.MaxSequenceLength == 512) // default value
            {
                options.MaxSequenceLength = modelInfo!.MaxSequenceLength;
            }
            options.PoolingMode = modelInfo!.PoolingMode;
            options.DoLowerCase = modelInfo.DoLowerCase;

            // Download model
            var cacheDir = options.CacheDirectory ?? CacheManager.GetDefaultCacheDirectory();
            using var downloader = new HuggingFaceDownloader(cacheDir, localFilesOnly: options.DisableAutoDownload);

            var modelDir = await downloader.DownloadModelAsync(
                modelInfo.RepoId,
                subfolder: modelInfo.Subfolder,
                progress: progress,
                cancellationToken: cancellationToken);

            modelPath = Path.Combine(modelDir, "model.onnx");
            // modelDir is the subfolder's own directory; DownloadModelAsync puts tokenizer assets it had
            // to fetch from the repository root there too, so they live alongside model.onnx.
            tokenizerPrimaryDir = modelDir;
            tokenizerFallbackDir = null;
            modelId = modelIdOrPath;
        }
        // Assume it's a HuggingFace repo ID (e.g., "sentence-transformers/all-MiniLM-L6-v2")
        else if (modelIdOrPath.Contains('/'))
        {
            var cacheDir = options.CacheDirectory ?? CacheManager.GetDefaultCacheDirectory();
            using var downloader = new HuggingFaceDownloader(cacheDir, localFilesOnly: options.DisableAutoDownload);

            // Use auto-discovery to find ONNX files and config
            var hwPrefs = ModelPreferences.ForCurrentHardware();
            var preferences = options.QuantizationHint is { } hint
                ? new ModelPreferences
                {
                    PreferLowMemory = hwPrefs.PreferLowMemory,
                    QuantizationPriority = ModelPreferences.ForQuantizationHint(hint).QuantizationPriority,
                    PreferredProvider = options.Provider != ExecutionProvider.Auto
                        ? options.Provider : hwPrefs.PreferredProvider
                }
                : hwPrefs;
            var (downloadedDir, discovery) = await downloader.DownloadWithDiscoveryAsync(
                modelIdOrPath,
                preferences: preferences,
                progress: progress,
                cancellationToken: cancellationToken);

            // Find the main model ONNX file
            var mainOnnxFile = discovery.OnnxFiles.FirstOrDefault(f =>
                f.EndsWith("model.onnx", StringComparison.OrdinalIgnoreCase)) ??
                (discovery.OnnxFiles.Count > 0 ? discovery.OnnxFiles[0] : null);

            if (mainOnnxFile is null)
            {
                throw new ModelNotFoundException(
                    $"No ONNX model file found in repository '{modelIdOrPath}'.",
                    modelIdOrPath);
            }

            // Preserve full path including subfolder (e.g., "onnx/model.onnx")
            modelPath = Path.Combine(downloadedDir, mainOnnxFile);

            // DownloadWithDiscoveryAsync preserves directory structure, so the model may live in
            // a subfolder (e.g. onnx/) while tokenizer assets sit at the repo root.
            tokenizerPrimaryDir = Path.GetDirectoryName(modelPath)!;
            tokenizerFallbackDir = string.Equals(tokenizerPrimaryDir, downloadedDir, StringComparison.OrdinalIgnoreCase)
                ? null
                : downloadedDir;

            modelId = modelIdOrPath.Split('/').Last();
        }
        else
        {
            throw new ModelNotFoundException(
                $"Unknown model '{modelIdOrPath}'. Use a known catalog alias (e.g., 'default', 'fast'), " +
                "a HuggingFace repo ID (e.g., 'sentence-transformers/all-MiniLM-L6-v2'), " +
                "or a local path to an ONNX model file.",
                modelIdOrPath);
        }

        // Validate model file exists
        if (!File.Exists(modelPath))
            throw new ModelNotFoundException("Model file not found", modelPath);

        // Resolve tokenizer directory: prefer the directory that actually contains tokenizer
        // assets. Probe primary first, then fall back to the repo root.
        var tokenizerDir = ResolveTokenizerDir(tokenizerPrimaryDir, tokenizerFallbackDir, modelId);

        // Load tokenizer using Text.Core (auto-detects WordPiece vs SentencePiece)
        var tokenizer = await TokenizerFactory.CreateAutoSequenceAsync(tokenizerDir, options.MaxSequenceLength);

        // Load inference engine (use async to ensure RuntimeManager initializes native binaries)
        var engine = await OnnxInferenceEngine.CreateAsync(modelPath, options.Provider, cancellationToken: cancellationToken);

        LogProviderSelection(modelId, options.Provider, engine);

        // Create pooling strategy
        var poolingStrategy = PoolingFactory.Create(options.PoolingMode);

        return new EmbeddingModel(modelId, engine, tokenizer, poolingStrategy, options, loadedModelInfo, modelPath);
    }

    private static void LogProviderSelection(string modelId, ExecutionProvider requested, OnnxInferenceEngine engine)
    {
        var profile = HardwareProfile.Current;
        var gpu = profile.GpuInfo;
        var active = engine.ActiveProviders.Count > 0
            ? string.Join("+", engine.ActiveProviders)
            : "(none)";
        System.Diagnostics.Trace.TraceInformation(
            $"[LocalEmbedder.auto] Requested={requested}, Active={active}, " +
            $"GPU={gpu.Vendor} {gpu.DeviceName ?? "n/a"}, " +
            $"Recommended={gpu.RecommendedProvider}, model={modelId}");
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
    /// <param name="modelId">
    /// A model alias (e.g., "default"), a known model ID, or a HuggingFace repo ID.
    /// </param>
    /// <param name="cacheDirectory">Custom cache directory, or <see langword="null"/> for default.</param>
    /// <returns><see langword="true"/> if the model files exist in cache and are not LFS pointers.</returns>
    public static bool IsModelDownloaded(string modelId, string? cacheDirectory = null)
    {
        var cacheDir = cacheDirectory ?? CacheManager.GetDefaultCacheDirectory();

        // Resolve alias to repo ID
        string repoId;
        string? subfolder = null;
        if (EmbedderModelRegistry.Default.TryResolve(modelId, out var modelInfo))
        {
            repoId = modelInfo!.RepoId;
            subfolder = modelInfo.Subfolder;
        }
        else
        {
            repoId = modelId;
        }

        var snapshotDir = CacheManager.GetModelDirectory(cacheDir, repoId);

        // When a subfolder is specified, model.onnx lives inside it
        var modelDir = subfolder != null ? Path.Combine(snapshotDir, subfolder) : snapshotDir;

        // Check for the essential model file
        var modelOnnx = Path.Combine(modelDir, "model.onnx");
        if (!File.Exists(modelOnnx) || CacheManager.IsLfsPointerFile(modelOnnx))
            return false;

        // Check for at least one tokenizer file (model dir first, then snapshot root)
        return DirectoryHasTokenizer(modelDir)
            || (subfolder != null && DirectoryHasTokenizer(snapshotDir));
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
    /// <param name="options">Optional configuration (cache directory, quantization hint).</param>
    /// <param name="progress">Optional progress reporting for downloads.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>The local directory path containing the downloaded model files.</returns>
    public static async Task<string> DownloadModelAsync(
        string modelId,
        EmbedderOptions? options = null,
        IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        options ??= new EmbedderOptions();

        // Parse variant qualifier
        var (baseId, qualifier) = LMSupplyOptionsBase.SplitQualifier(modelId);
        modelId = baseId;
        options.QuantizationHint ??= qualifier;

        var cacheDir = options.CacheDirectory ?? CacheManager.GetDefaultCacheDirectory();
        using var downloader = new HuggingFaceDownloader(cacheDir, localFilesOnly: options.DisableAutoDownload);

        // Known model alias / registered model
        if (EmbedderModelRegistry.Default.TryResolve(modelId, out var modelInfo))
        {
            return await downloader.DownloadModelAsync(
                modelInfo!.RepoId,
                subfolder: modelInfo.Subfolder,
                progress: progress,
                cancellationToken: cancellationToken);
        }

        // HuggingFace repo ID with auto-discovery
        if (modelId.Contains('/'))
        {
            var hwPrefs = ModelPreferences.ForCurrentHardware();
            var preferences = options.QuantizationHint is { } hint
                ? new ModelPreferences
                {
                    PreferLowMemory = hwPrefs.PreferLowMemory,
                    QuantizationPriority = ModelPreferences.ForQuantizationHint(hint).QuantizationPriority,
                    PreferredProvider = options.Provider != ExecutionProvider.Auto
                        ? options.Provider : hwPrefs.PreferredProvider
                }
                : hwPrefs;

            var (downloadedDir, _) = await downloader.DownloadWithDiscoveryAsync(
                modelId,
                preferences: preferences,
                progress: progress,
                cancellationToken: cancellationToken);
            return downloadedDir;
        }

        throw new ModelNotFoundException(
            $"Unknown model '{modelId}'. Use a known model alias, a HuggingFace repo ID, " +
            "or a local path to an ONNX model file.",
            modelId);
    }

    /// <summary>
    /// Gets a list of pre-configured model IDs available for download.
    /// </summary>
    public static IEnumerable<string> GetAvailableModels()
        => EmbedderModelRegistry.Default.GetAliases().Select(a => a.Name);

    /// <summary>
    /// Gets all registered model information (deduplicated by RepoId).
    /// </summary>
    public static IEnumerable<ModelInfo> GetAllModels()
        => EmbedderModelRegistry.Default.GetAvailableModels();

    /// <summary>
    /// Computes cosine similarity between two embedding vectors.
    /// </summary>
    public static float CosineSimilarity(ReadOnlySpan<float> embedding1, ReadOnlySpan<float> embedding2)
    {
        return VectorOperations.CosineSimilarity(embedding1, embedding2);
    }

    /// <summary>
    /// Computes Euclidean distance between two embedding vectors.
    /// </summary>
    public static float EuclideanDistance(ReadOnlySpan<float> embedding1, ReadOnlySpan<float> embedding2)
    {
        return VectorOperations.EuclideanDistance(embedding1, embedding2);
    }

    /// <summary>
    /// Computes dot product of two embedding vectors.
    /// </summary>
    public static float DotProduct(ReadOnlySpan<float> embedding1, ReadOnlySpan<float> embedding2)
    {
        return VectorOperations.DotProduct(embedding1, embedding2);
    }

    /// <summary>
    /// Tokenizer file names recognised by <see cref="TokenizerFactory.CreateAutoSequenceAsync"/>.
    /// Used to detect which directory actually contains tokenizer assets.
    /// </summary>
    private static readonly string[] _tokenizerProbeFiles =
    [
        "vocab.txt",
        "tokenizer.json",
        "sentencepiece.bpe.model",
    ];

    /// <summary>
    /// Picks the directory that contains tokenizer assets, preferring the model directory and
    /// falling back to the repo root when present. Throws a descriptive
    /// <see cref="ModelNotFoundException"/> when nothing usable is found.
    /// </summary>
    private static string ResolveTokenizerDir(string primaryDir, string? fallbackDir, string modelId)
    {
        if (DirectoryHasTokenizer(primaryDir))
            return primaryDir;

        if (fallbackDir is not null && DirectoryHasTokenizer(fallbackDir))
            return fallbackDir;

        // Build a helpful error listing every path we tried.
        var triedDirs = fallbackDir is null
            ? new[] { primaryDir }
            : new[] { primaryDir, fallbackDir };
        var triedFiles = string.Join(", ", triedDirs.SelectMany(d => _tokenizerProbeFiles.Select(f => Path.Combine(d, f))));

        throw new ModelNotFoundException(
            $"No tokenizer file found for '{modelId}'. " +
            $"Tried: {triedFiles}. Expected one of: vocab.txt (WordPiece), " +
            "tokenizer.json (WordPiece/Unigram/BPE), sentencepiece.bpe.model (SentencePiece).",
            modelId);
    }

    private static bool DirectoryHasTokenizer(string dir)
    {
        if (!Directory.Exists(dir))
            return false;

        foreach (var name in _tokenizerProbeFiles)
        {
            if (File.Exists(Path.Combine(dir, name)))
                return true;
        }

        // Also accept any *.spm file (used by translation/multilingual models)
        return Directory.EnumerateFiles(dir, "*.spm").Any();
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

    /// <summary>
    /// Loads a GGUF embedding model.
    /// </summary>
    private static async Task<IEmbeddingModel> LoadGgufAsync(
        string modelIdOrPath,
        EmbedderOptions options,
        IProgress<DownloadProgress>? progress,
        CancellationToken cancellationToken)
    {
        string modelPath;
        string modelId;

        // Remove gguf: prefix if present
        var cleanPath = modelIdOrPath.StartsWith("gguf:", StringComparison.OrdinalIgnoreCase)
            ? modelIdOrPath[5..]
            : modelIdOrPath;

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
                preferredQuantization: "Q4_K_M",
                progress: progress,
                cancellationToken: cancellationToken);

            modelId = cleanPath.Split('/').Last();
        }
        else
        {
            throw new ModelNotFoundException(
                $"GGUF model not found: '{modelIdOrPath}'. " +
                "Provide a local path to a .gguf file or a HuggingFace repo ID " +
                "(e.g., 'gguf:nomic-ai/nomic-embed-text-v1.5-GGUF').",
                modelIdOrPath);
        }

        return await LlamaServerEmbeddingModel.LoadAsync(
            modelId,
            modelPath,
            options,
            progress,
            cancellationToken);
    }
}
