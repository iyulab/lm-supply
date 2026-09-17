using LMSupply.Reranker.Models;

namespace LMSupply.Reranker.Infrastructure;

/// <summary>
/// Manages model lifecycle including download, caching, and verification.
/// Uses the shared LMSupply.Core download infrastructure.
/// </summary>
internal sealed class ModelManager : IDisposable
{
    private readonly HuggingFaceDownloader _downloader;
    private readonly string _cacheDir;
    private bool _disposed;

    /// <summary>
    /// Initializes a new ModelManager instance.
    /// </summary>
    /// <param name="cacheDirectory">Custom cache directory, or null for default.</param>
    /// <param name="autoDownload">
    /// Whether to download missing models. When false the downloader reads the local cache only
    /// (<see cref="HuggingFaceDownloader.LocalFilesOnly"/>) and a missing model throws
    /// <see cref="ModelNotFoundException"/>.
    /// </param>
    public ModelManager(
        string? cacheDirectory = null,
        bool autoDownload = true)
    {
        _cacheDir = cacheDirectory ?? CacheManager.GetDefaultCacheDirectory();
        _downloader = new HuggingFaceDownloader(_cacheDir, localFilesOnly: !autoDownload);
    }

    /// <summary>
    /// Gets the cache directory path.
    /// </summary>
    public string CacheDirectory => _cacheDir;

    /// <summary>
    /// Ensures a model is available locally, downloading if necessary.
    /// </summary>
    /// <param name="modelInfo">Model information.</param>
    /// <param name="progress">Optional progress reporter.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Paths to the model and tokenizer files.</returns>
    public async Task<ModelPaths> EnsureModelAsync(
        ModelInfo modelInfo,
        IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(modelInfo);

        // Handle local path
        if (IsLocalPath(modelInfo.Id))
        {
            return GetLocalModelPaths(modelInfo);
        }

        // Get expected file paths. The tokenizer can come from another repository than the weights (an ONNX export that
        // did not carry the SentencePiece model); its files sit together, because the tokenizer is built from their directory.
        var tokenizerRepo = modelInfo.TokenizerRepoId ?? modelInfo.Id;
        var modelPath = CacheManager.GetModelFilePath(_cacheDir, modelInfo.Id, modelInfo.OnnxFile);
        var tokenizerPath = CacheManager.GetModelFilePath(_cacheDir, tokenizerRepo, modelInfo.TokenizerFile);
        var dataPath = modelInfo.OnnxDataFile is null ? null : CacheManager.GetModelFilePath(_cacheDir, modelInfo.Id, modelInfo.OnnxDataFile);

        // Check if already cached and not LFS pointers. A graph whose weights live in an external file is only cached
        // with that file: a cache holding the graph shell alone (left by a release that did not fetch it) is completed here.
        var modelExists = File.Exists(modelPath) && !CacheManager.IsLfsPointerFile(modelPath);
        var tokenizerExists = File.Exists(tokenizerPath) && !CacheManager.IsLfsPointerFile(tokenizerPath);
        var dataExists = dataPath is null || (File.Exists(dataPath) && !CacheManager.IsLfsPointerFile(dataPath));

        if (modelExists && tokenizerExists && dataExists)
        {
            return new ModelPaths(modelPath, tokenizerPath);
        }

        // Download required files — with auto-download disabled the downloader reads the cache only, so
        // a missing graph throws ModelNotFoundException there and a missing tokenizer is caught below.
        var weightFiles = new List<string>();
        if (!modelExists) weightFiles.Add(modelInfo.OnnxFile);
        if (!dataExists) weightFiles.Add(modelInfo.OnnxDataFile!);

        var tokenizerFiles = new List<string>();
        if (!tokenizerExists) tokenizerFiles.Add(modelInfo.TokenizerFile);

        // Also try to download tokenizer-specific files based on model architecture
        // BERT models use vocab.txt, XLM-RoBERTa models use sentencepiece.bpe.model
        var vocabPath = CacheManager.GetModelFilePath(_cacheDir, tokenizerRepo, "vocab.txt");
        if (!File.Exists(vocabPath) || CacheManager.IsLfsPointerFile(vocabPath))
        {
            tokenizerFiles.Add("vocab.txt");
        }

        var sentencepiecePath = CacheManager.GetModelFilePath(_cacheDir, tokenizerRepo, "sentencepiece.bpe.model");
        if (!File.Exists(sentencepiecePath) || CacheManager.IsLfsPointerFile(sentencepiecePath))
        {
            tokenizerFiles.Add("sentencepiece.bpe.model");
        }

        if (tokenizerRepo == modelInfo.Id)
        {
            await _downloader.DownloadModelAsync(
                modelInfo.Id,
                [.. weightFiles, .. tokenizerFiles],
                progress: progress,
                cancellationToken: cancellationToken);
        }
        else
        {
            if (weightFiles.Count > 0)
            {
                await _downloader.DownloadModelAsync(modelInfo.Id, weightFiles, progress: progress, cancellationToken: cancellationToken);
            }

            await _downloader.DownloadModelAsync(tokenizerRepo, tokenizerFiles, progress: progress, cancellationToken: cancellationToken);
        }

        // Verify downloads
        modelPath = CacheManager.GetModelFilePath(_cacheDir, modelInfo.Id, modelInfo.OnnxFile);
        tokenizerPath = CacheManager.GetModelFilePath(_cacheDir, tokenizerRepo, modelInfo.TokenizerFile);

        if (!File.Exists(modelPath))
        {
            throw new ModelDownloadException($"Model file was not downloaded successfully.", modelInfo.Id);
        }

        if (dataPath is not null && !File.Exists(dataPath))
        {
            // The downloader treats only .onnx graphs as required; this graph cannot initialize without its weights.
            throw _downloader.LocalFilesOnly
                ? new ModelNotFoundException(
                    $"External weights '{modelInfo.OnnxDataFile}' of model '{modelInfo.Id}' are not in the local cache and downloads are disabled.",
                    modelInfo.Id)
                : new ModelDownloadException($"External weights file '{modelInfo.OnnxDataFile}' was not downloaded successfully.", modelInfo.Id);
        }

        if (!File.Exists(tokenizerPath))
        {
            // The downloader treats a tokenizer as optional (only graphs are required), but this model
            // cannot run without one.
            throw _downloader.LocalFilesOnly
                ? new ModelNotFoundException(
                    $"Tokenizer '{modelInfo.TokenizerFile}' of model '{modelInfo.Id}' (from '{tokenizerRepo}') is not in the local cache and downloads are disabled.",
                    modelInfo.Id)
                : new ModelDownloadException($"Tokenizer file was not downloaded successfully.", modelInfo.Id);
        }

        return new ModelPaths(modelPath, tokenizerPath);
    }

    /// <summary>
    /// Gets the local path for a model if it exists in cache.
    /// </summary>
    /// <param name="modelInfo">Model information.</param>
    /// <returns>Model paths if cached, null otherwise.</returns>
    public ModelPaths? GetCachedModel(ModelInfo modelInfo)
    {
        ArgumentNullException.ThrowIfNull(modelInfo);

        if (IsLocalPath(modelInfo.Id))
        {
            return GetLocalModelPaths(modelInfo);
        }

        var modelPath = CacheManager.GetModelFilePath(_cacheDir, modelInfo.Id, modelInfo.OnnxFile);
        var tokenizerPath = CacheManager.GetModelFilePath(_cacheDir, modelInfo.TokenizerRepoId ?? modelInfo.Id, modelInfo.TokenizerFile);

        if (File.Exists(modelPath) && File.Exists(tokenizerPath) &&
            !CacheManager.IsLfsPointerFile(modelPath) &&
            (modelInfo.OnnxDataFile is null || File.Exists(CacheManager.GetModelFilePath(_cacheDir, modelInfo.Id, modelInfo.OnnxDataFile))))
        {
            return new ModelPaths(modelPath, tokenizerPath);
        }

        return null;
    }

    /// <summary>
    /// Deletes a model from the cache.
    /// </summary>
    /// <param name="modelId">Model identifier.</param>
    public void DeleteModel(string modelId)
    {
        CacheManager.DeleteModel(_cacheDir, modelId);
    }

    /// <summary>
    /// Gets all cached models.
    /// </summary>
    public IEnumerable<(string ModelId, string Revision)> GetCachedModels()
    {
        return CacheManager.GetCachedModels(_cacheDir);
    }

    private static bool IsLocalPath(string path)
    {
        return path.EndsWith(".onnx", StringComparison.OrdinalIgnoreCase) ||
               Path.IsPathRooted(path) ||
               path.StartsWith("./", StringComparison.Ordinal) ||
               path.StartsWith("../", StringComparison.Ordinal) ||
               path.StartsWith(".\\", StringComparison.Ordinal) ||
               path.StartsWith("..\\", StringComparison.Ordinal);
    }

    private static ModelPaths GetLocalModelPaths(ModelInfo modelInfo)
    {
        var modelPath = Path.GetFullPath(modelInfo.Id);
        var directory = Path.GetDirectoryName(modelPath) ?? ".";
        var tokenizerPath = Path.Combine(directory, modelInfo.TokenizerFile);

        if (!File.Exists(modelPath))
        {
            throw new ModelNotFoundException($"Local model file not found: {modelPath}", modelInfo.Id);
        }

        if (!File.Exists(tokenizerPath))
        {
            // Try vocab.txt as fallback
            var vocabPath = Path.Combine(directory, "vocab.txt");
            if (File.Exists(vocabPath))
            {
                // Create a minimal tokenizer.json reference
                tokenizerPath = vocabPath;
            }
            else
            {
                throw new ModelNotFoundException(
                    $"Tokenizer file not found in local model directory: {directory}",
                    modelInfo.Id);
            }
        }

        return new ModelPaths(modelPath, tokenizerPath);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _downloader.Dispose();
        _disposed = true;
    }
}

/// <summary>
/// Paths to model files.
/// </summary>
/// <param name="ModelPath">Path to the ONNX model file.</param>
/// <param name="TokenizerPath">Path to the tokenizer configuration file.</param>
public readonly record struct ModelPaths(string ModelPath, string TokenizerPath);
