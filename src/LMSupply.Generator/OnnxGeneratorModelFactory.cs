using LMSupply.Download;
using LMSupply.Generator.Abstractions;
using LMSupply.Generator.ChatFormatters;

namespace LMSupply.Generator;

/// <summary>
/// Factory for creating ONNX-based generator models.
/// </summary>
public sealed class OnnxGeneratorModelFactory : IGeneratorModelFactory, IDisposable
{
    private readonly string _cacheDirectory;
    private readonly ExecutionProvider _defaultProvider;
    private readonly HuggingFaceDownloader _downloader;
    private bool _disposed;

    /// <summary>
    /// Files required for ONNX GenAI models.
    /// </summary>
    private static readonly string[] GenAiModelFiles =
    [
        "genai_config.json",
        "model.onnx",
        "model.onnx.data",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json"
    ];

    /// <summary>
    /// Creates a new factory with default settings.
    /// </summary>
    public OnnxGeneratorModelFactory()
        : this(GetDefaultCacheDirectory(), ExecutionProvider.Auto)
    {
    }

    /// <summary>
    /// Creates a new factory with specified settings.
    /// </summary>
    /// <param name="cacheDirectory">Directory for model cache.</param>
    /// <param name="defaultProvider">Default execution provider.</param>
    public OnnxGeneratorModelFactory(string cacheDirectory, ExecutionProvider defaultProvider)
    {
        _cacheDirectory = cacheDirectory ?? throw new ArgumentNullException(nameof(cacheDirectory));
        _defaultProvider = defaultProvider;
        _downloader = new HuggingFaceDownloader(cacheDirectory);
    }

    /// <inheritdoc />
    public async Task<IGeneratorModel> LoadAsync(
        string modelId,
        GeneratorOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        options ??= new GeneratorOptions();

        var modelPath = await ResolveModelPathAsync(modelId, cancellationToken);
        var chatFormatter = ResolveChatFormatter(modelId, options.ChatFormat);

        // Merge default provider if not specified
        if (options.Provider == ExecutionProvider.Auto && _defaultProvider != ExecutionProvider.Auto)
        {
            options = new GeneratorOptions
            {
                CacheDirectory = options.CacheDirectory ?? _cacheDirectory,
                Provider = _defaultProvider,
                ChatFormat = options.ChatFormat,
                Verbose = options.Verbose,
                MaxContextLength = options.MaxContextLength,
                MaxConcurrentRequests = options.MaxConcurrentRequests
            };
        }

        return new Internal.OnnxGeneratorModel(modelId, modelPath, chatFormatter, options);
    }

    /// <inheritdoc />
    public bool IsModelAvailable(string modelId)
    {
        var modelPath = GetModelCachePath(modelId);
        if (!Directory.Exists(modelPath))
            return false;

        // Check for required ONNX GenAI files
        return File.Exists(Path.Combine(modelPath, "genai_config.json"))
            || File.Exists(Path.Combine(modelPath, "model.onnx"));
    }

    /// <inheritdoc />
    public async Task DownloadModelAsync(
        string modelId,
        IProgress<ModelDownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        if (IsModelAvailable(modelId))
        {
            progress?.Report(new ModelDownloadProgress(100, 100, null));
            return;
        }

        // Determine the best variant subfolder based on provider
        var subfolder = GetVariantSubfolder(modelId);

        // Create progress adapter
        IProgress<DownloadProgress>? downloadProgress = null;
        if (progress != null)
        {
            downloadProgress = new Progress<DownloadProgress>(p =>
            {
                progress.Report(new ModelDownloadProgress(
                    p.BytesDownloaded,
                    p.TotalBytes,
                    p.FileName));
            });
        }

        // Download using HuggingFace downloader
        await _downloader.DownloadModelAsync(
            modelId,
            files: GenAiModelFiles,
            revision: "main",
            subfolder: subfolder,
            progress: downloadProgress,
            cancellationToken: cancellationToken);
    }

    /// <summary>
    /// Determines the variant subfolder based on the model and provider.
    /// </summary>
    private string? GetVariantSubfolder(string modelId)
    {
        // Check if model has variant subfolders (common for ONNX GenAI models)
        var modelInfo = ModelRegistry.GetModel(modelId);

        // For Microsoft Phi models, they typically have variant subfolders
        if (modelId.Contains("phi", StringComparison.OrdinalIgnoreCase) ||
            modelId.Contains("onnx", StringComparison.OrdinalIgnoreCase))
        {
            return _defaultProvider switch
            {
                ExecutionProvider.Cuda => "cuda-int4-rtn-block-32",
                ExecutionProvider.DirectML => "directml-int4-awq-block-128",
                _ => "cpu-int4-rtn-block-32-acc-level-4"
            };
        }

        return null;
    }

    /// <summary>
    /// Gets the cache path for a model.
    /// </summary>
    public string GetModelCachePath(string modelId)
    {
        // HuggingFace cache format: models--org--name
        var safeName = modelId.Replace("/", "--");
        return Path.Combine(_cacheDirectory, $"models--{safeName}");
    }

    /// <summary>
    /// Lists all locally available models.
    /// </summary>
    public IReadOnlyList<string> GetAvailableModels()
    {
        if (!Directory.Exists(_cacheDirectory))
            return [];

        var models = new List<string>();
        foreach (var dir in Directory.GetDirectories(_cacheDirectory, "models--*"))
        {
            var dirName = Path.GetFileName(dir);
            if (dirName.StartsWith("models--"))
            {
                var modelId = dirName["models--".Length..].Replace("--", "/");

                // Verify it's a valid ONNX GenAI model
                if (File.Exists(Path.Combine(dir, "genai_config.json"))
                    || HasOnnxSubdirectory(dir))
                {
                    models.Add(modelId);
                }
            }
        }

        return models;
    }

    private async Task<string> ResolveModelPathAsync(string modelId, CancellationToken cancellationToken)
    {
        var cachePath = GetModelCachePath(modelId);

        // Check if model exists directly
        if (IsValidModelDirectory(cachePath))
            return cachePath;

        // Check for snapshot subdirectory (HuggingFace cache format)
        var snapshotsDir = Path.Combine(cachePath, "snapshots");
        if (Directory.Exists(snapshotsDir))
        {
            var snapshots = Directory.GetDirectories(snapshotsDir);
            if (snapshots.Length > 0)
            {
                // Use most recent snapshot
                var latestSnapshot = snapshots.OrderByDescending(Directory.GetLastWriteTimeUtc).First();
                if (IsValidModelDirectory(latestSnapshot))
                    return latestSnapshot;
            }
        }

        // Check for variant subdirectories (cpu-int4, cuda-int4, etc.)
        var variants = new[] { "cpu-int4-rtn-block-32-acc-level-4", "cpu-int4", "cuda-int4", "directml-int4" };
        foreach (var variant in variants)
        {
            var variantPath = Path.Combine(cachePath, variant);
            if (IsValidModelDirectory(variantPath))
                return variantPath;
        }

        // Model not found - attempt download
        await DownloadModelAsync(modelId, null, cancellationToken);

        // After download, try again
        if (IsValidModelDirectory(cachePath))
            return cachePath;

        throw new FileNotFoundException($"Model '{modelId}' not found at {cachePath}");
    }

    private static bool IsValidModelDirectory(string path)
    {
        if (!Directory.Exists(path))
            return false;

        return File.Exists(Path.Combine(path, "genai_config.json"))
            || File.Exists(Path.Combine(path, "model.onnx"))
            || File.Exists(Path.Combine(path, "model.onnx.data"));
    }

    private static bool HasOnnxSubdirectory(string path)
    {
        if (!Directory.Exists(path))
            return false;

        foreach (var subdir in Directory.GetDirectories(path))
        {
            if (IsValidModelDirectory(subdir))
                return true;
        }

        return false;
    }

    private static IChatFormatter ResolveChatFormatter(string modelId, string? explicitFormat)
    {
        if (!string.IsNullOrEmpty(explicitFormat))
            return ChatFormatterFactory.Create(explicitFormat);

        // Try to get from registry
        var modelInfo = ModelRegistry.GetModel(modelId);
        if (modelInfo != null)
            return ChatFormatterFactory.Create(modelInfo.ChatFormat);

        // Fall back to auto-detection from model name
        return ChatFormatterFactory.Create(modelId);
    }

    private static string GetDefaultCacheDirectory()
    {
        return Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            ".cache", "huggingface", "hub");
    }

    /// <summary>
    /// Releases resources used by the factory.
    /// </summary>
    public void Dispose()
    {
        if (_disposed) return;
        _downloader.Dispose();
        _disposed = true;
    }
}
