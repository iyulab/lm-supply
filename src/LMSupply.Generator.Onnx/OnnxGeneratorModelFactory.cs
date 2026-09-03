using System.Diagnostics;
using LMSupply.Core.Download;
using LMSupply.Download;
using LMSupply.Generator.Abstractions;
using LMSupply.Generator.ChatFormatters;

namespace LMSupply.Generator;

/// <summary>
/// Factory for creating ONNX-based generator models.
/// </summary>
public sealed class OnnxGeneratorModelFactory : IOnnxGeneratorModelFactory
{
    private readonly string _cacheDirectory;
    private readonly ExecutionProvider _defaultProvider;
    private readonly HuggingFaceDownloader _downloader;
    private bool _disposed;

    private const string DefaultRevision = "main";

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
        : this(CacheManager.GetDefaultCacheDirectory(), ExecutionProvider.Auto)
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

        var (modelPath, configBasePath) = await ResolveModelPathWithBaseAsync(modelId, cancellationToken);
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

        return new Internal.OnnxGeneratorModel(modelId, modelPath, chatFormatter, options, configBasePath);
    }

    /// <inheritdoc />
    public bool IsModelAvailable(string modelId)
    {
        var snapshotPath = GetModelCachePath(modelId);

        // Check if model exists directly in snapshot
        if (IsValidModelDirectory(snapshotPath))
            return true;

        // Check for variant subdirectories
        return FindVariantSubfolder(snapshotPath) != null;
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
    /// Determines the variant subfolder based on model registry info and provider.
    /// </summary>
    private string? GetVariantSubfolder(string modelId)
    {
        // First, check registry for explicit subfolder configuration
        GeneratorModelRegistry.Default.TryResolve(modelId, out var modelInfo);
        if (modelInfo?.Subfolder != null)
        {
            // Registry may have provider-neutral subfolder (e.g., "cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4")
            // Adapt based on provider if needed
            return AdaptSubfolderForProvider(modelInfo.Subfolder, _defaultProvider);
        }

        // Fallback for unregistered models: infer from model ID patterns
        if (modelId.Contains("phi", StringComparison.OrdinalIgnoreCase) && modelId.Contains("onnx", StringComparison.OrdinalIgnoreCase))
        {
            return _defaultProvider switch
            {
                ExecutionProvider.Cuda => "cuda/cuda-int4-rtn-block-32",
                ExecutionProvider.DirectML => "directml/directml-int4-awq-block-128",
                _ => "cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4"
            };
        }

        return null;
    }

    /// <summary>
    /// Adapts a registry subfolder for the target execution provider.
    /// </summary>
    private static string AdaptSubfolderForProvider(string subfolder, ExecutionProvider provider)
    {
        // If the subfolder is already provider-specific, use as-is
        if (subfolder.Contains("cuda", StringComparison.OrdinalIgnoreCase) ||
            subfolder.Contains("directml", StringComparison.OrdinalIgnoreCase) ||
            subfolder.Contains("cpu", StringComparison.OrdinalIgnoreCase))
        {
            return provider switch
            {
                // For CUDA, try to find cuda variant
                ExecutionProvider.Cuda when subfolder.Contains("cpu", StringComparison.OrdinalIgnoreCase)
                    => subfolder.Replace("cpu_and_mobile", "cuda").Replace("cpu-", "cuda-"),
                // For DirectML, try to find directml variant
                ExecutionProvider.DirectML when subfolder.Contains("cpu", StringComparison.OrdinalIgnoreCase)
                    => subfolder.Replace("cpu_and_mobile", "directml").Replace("cpu-", "directml-"),
                _ => subfolder
            };
        }

        return subfolder;
    }

    /// <summary>
    /// Gets the cache path for a model, following HuggingFace cache structure.
    /// </summary>
    public string GetModelCachePath(string modelId)
    {
        // Use CacheManager for consistent path resolution
        // Structure: models--{org}--{name}/snapshots/{revision}
        return CacheManager.GetModelDirectory(_cacheDirectory, modelId, DefaultRevision);
    }

    /// <summary>
    /// Lists all locally available generator models.
    /// </summary>
    public IReadOnlyList<string> GetAvailableModels()
    {
        // Use CacheManager's type detection for consistent discovery
        return CacheManager.GetCachedModelsByType(_cacheDirectory, ModelType.Generator)
            .Select(m => m.RepoId)
            .ToList();
    }

    /// <summary>
    /// Resolves model path and config base path.
    /// Returns (modelPath, configBasePath) where configBasePath is the snapshot root
    /// when the model is in a subfolder, or null when model is at the root.
    /// </summary>
    private async Task<(string modelPath, string? configBasePath)> ResolveModelPathWithBaseAsync(
        string modelId, CancellationToken cancellationToken)
    {
        var snapshotPath = GetModelCachePath(modelId);
        var triedPaths = new List<string> { snapshotPath };

        // Branch (A): snapshot root
        if (IsValidModelDirectory(snapshotPath))
        {
            Trace.TraceInformation(
                $"[OnnxGenerator] Path resolution: snapshot-root hit for {modelId} at {snapshotPath}");
            return (snapshotPath, null);
        }

        // Branch (B): registry-driven subfolder
        GeneratorModelRegistry.Default.TryResolve(modelId, out var registryInfo);
        if (registryInfo?.Subfolder != null)
        {
            var registrySubfolderPath = Path.Combine(snapshotPath, registryInfo.Subfolder);
            triedPaths.Add(registrySubfolderPath);
            if (IsValidModelDirectory(registrySubfolderPath))
            {
                Trace.TraceInformation(
                    $"[OnnxGenerator] Path resolution: registry-subfolder hit for {modelId} — {registryInfo.Subfolder}");
                return (registrySubfolderPath, snapshotPath);
            }
            Trace.TraceInformation(
                $"[OnnxGenerator] Path resolution: registry-subfolder MISS for {modelId} — tried {registrySubfolderPath}");
        }
        else
        {
            Trace.TraceInformation(
                $"[OnnxGenerator] Path resolution: no registry subfolder for {modelId} " +
                $"(registryInfo={(registryInfo == null ? "null" : "no-subfolder")})");
        }

        // Branch (C): variant subfolder search (1-level + 2-level)
        var foundPath = FindVariantSubfolder(snapshotPath);
        if (foundPath != null)
        {
            Trace.TraceInformation(
                $"[OnnxGenerator] Path resolution: variant-search hit for {modelId} — {Path.GetRelativePath(snapshotPath, foundPath)}");
            return (foundPath, snapshotPath);
        }

        // Model not found — attempt download and retry
        Trace.TraceInformation($"[OnnxGenerator] Path resolution: no cached layout found, attempting download for {modelId}");
        await DownloadModelAsync(modelId, null, cancellationToken);

        if (IsValidModelDirectory(snapshotPath))
            return (snapshotPath, null);

        foundPath = FindVariantSubfolder(snapshotPath);
        if (foundPath != null)
            return (foundPath, snapshotPath);

        throw new FileNotFoundException(
            $"Model '{modelId}' not found. Tried:\n  " +
            string.Join("\n  ", triedPaths) +
            "\nSearched variant subfolders (2 levels deep) — no directory containing 'genai_config.json' was found.");
    }

    /// <summary>
    /// Finds a valid variant subfolder within the model directory.
    /// </summary>
    private string? FindVariantSubfolder(string basePath)
    {
        if (!Directory.Exists(basePath))
            return null;

        // Provider-specific variant prefixes in priority order
        var variantPatterns = _defaultProvider switch
        {
            ExecutionProvider.Cuda => new[] { "cuda", "gpu", "cpu" },
            ExecutionProvider.DirectML => new[] { "directml", "gpu", "cpu" },
            _ => new[] { "cpu", "gpu" }
        };

        // Level 1: direct prefix match on an immediate subdirectory
        foreach (var pattern in variantPatterns)
        {
            foreach (var subdir in Directory.GetDirectories(basePath))
            {
                var dirName = Path.GetFileName(subdir);
                if (dirName.StartsWith(pattern, StringComparison.OrdinalIgnoreCase)
                    && IsValidModelDirectory(subdir))
                    return subdir;
            }
        }

        // Level 2: nested variant layout (e.g., cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4)
        // Walk into any level-1 directory whose name contains the provider pattern,
        // then check each of its immediate children for a valid model directory.
        foreach (var pattern in variantPatterns)
        {
            foreach (var subdir in Directory.GetDirectories(basePath))
            {
                var dirName = Path.GetFileName(subdir);
                if (!dirName.Contains(pattern, StringComparison.OrdinalIgnoreCase))
                    continue;

                foreach (var nested in Directory.GetDirectories(subdir))
                {
                    if (IsValidModelDirectory(nested))
                        return nested;
                }
            }
        }

        // Fallback: any valid model at level 1
        foreach (var subdir in Directory.GetDirectories(basePath))
        {
            if (IsValidModelDirectory(subdir))
                return subdir;
        }

        // Fallback: any valid model at level 2
        foreach (var subdir in Directory.GetDirectories(basePath))
        {
            foreach (var nested in Directory.GetDirectories(subdir))
            {
                if (IsValidModelDirectory(nested))
                    return nested;
            }
        }

        return null;
    }

    private static bool IsValidModelDirectory(string path)
    {
        var result = ModelDirectoryValidator.Validate(path);
        return result.IsValid;
    }

    private static IChatFormatter ResolveChatFormatter(string modelId, string? explicitFormat)
    {
        if (!string.IsNullOrEmpty(explicitFormat))
            return ChatFormatterFactory.Create(explicitFormat);

        // Try to get from registry
        if (GeneratorModelRegistry.Default.TryResolve(modelId, out var chatModelInfo) &&
            chatModelInfo?.ChatFormat != null)
            return ChatFormatterFactory.Create(chatModelInfo.ChatFormat);

        // Fall back to auto-detection from model name
        return ChatFormatterFactory.Create(modelId);
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
