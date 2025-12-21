using LMSupply.Core.Download;
using LMSupply.Embedder.Inference;
using LMSupply.Embedder.Pooling;
using LMSupply.Embedder.Utils;
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
    /// Either a model ID (e.g., "all-MiniLM-L6-v2") for auto-download,
    /// or a local path to an ONNX model file.
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

        string modelPath;
        string vocabPath;
        string modelId;

        ModelInfo? loadedModelInfo = null;

        // Check if it's a local path
        if (File.Exists(modelIdOrPath) || modelIdOrPath.EndsWith(".onnx", StringComparison.OrdinalIgnoreCase))
        {
            modelPath = modelIdOrPath;
            modelId = Path.GetFileNameWithoutExtension(modelPath);

            var modelDir = Path.GetDirectoryName(modelPath) ?? ".";
            vocabPath = Path.Combine(modelDir, "vocab.txt");

            if (!File.Exists(vocabPath))
            {
                throw new ModelNotFoundException(
                    $"Vocabulary file not found. Expected at: {vocabPath}",
                    modelId);
            }
        }
        // Check if it's a known model ID
        else if (ModelRegistry.TryGetModel(modelIdOrPath, out var modelInfo))
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
            using var downloader = new HuggingFaceDownloader(cacheDir);

            var modelDir = await downloader.DownloadModelAsync(
                modelInfo.RepoId,
                subfolder: modelInfo.Subfolder,
                progress: progress,
                cancellationToken: cancellationToken);

            modelPath = Path.Combine(modelDir, "model.onnx");
            vocabPath = Path.Combine(modelDir, "vocab.txt");
            modelId = modelIdOrPath;
        }
        // Assume it's a HuggingFace repo ID (e.g., "sentence-transformers/all-MiniLM-L6-v2")
        else if (modelIdOrPath.Contains('/'))
        {
            var cacheDir = options.CacheDirectory ?? CacheManager.GetDefaultCacheDirectory();
            using var downloader = new HuggingFaceDownloader(cacheDir);

            // Use auto-discovery to find ONNX files and config
            var (downloadedDir, discovery) = await downloader.DownloadWithDiscoveryAsync(
                modelIdOrPath,
                preferences: ModelPreferences.Default,
                progress: progress,
                cancellationToken: cancellationToken);

            // Find the main model ONNX file
            var mainOnnxFile = discovery.OnnxFiles.FirstOrDefault(f =>
                f.EndsWith("model.onnx", StringComparison.OrdinalIgnoreCase)) ??
                discovery.OnnxFiles.FirstOrDefault();

            if (mainOnnxFile is null)
            {
                throw new ModelNotFoundException(
                    $"No ONNX model file found in repository '{modelIdOrPath}'.",
                    modelIdOrPath);
            }

            // Preserve full path including subfolder (e.g., "onnx/model.onnx")
            modelPath = Path.Combine(downloadedDir, mainOnnxFile);

            // Look for vocab.txt in the same directory as the model
            var modelDir = Path.GetDirectoryName(modelPath)!;
            vocabPath = Path.Combine(modelDir, "vocab.txt");

            // Fall back to root if not in model directory
            if (!File.Exists(vocabPath))
            {
                vocabPath = Path.Combine(downloadedDir, "vocab.txt");
            }

            modelId = modelIdOrPath.Split('/').Last();
        }
        else
        {
            throw new ModelNotFoundException(
                $"Unknown model '{modelIdOrPath}'. Use a known model ID (e.g., 'all-MiniLM-L6-v2'), " +
                "a HuggingFace repo ID (e.g., 'sentence-transformers/all-MiniLM-L6-v2'), " +
                "or a local path to an ONNX model file.",
                modelIdOrPath);
        }

        // Validate files exist
        if (!File.Exists(modelPath))
            throw new ModelNotFoundException("Model file not found", modelPath);
        if (!File.Exists(vocabPath))
            throw new ModelNotFoundException("Vocabulary file not found", vocabPath);

        // Load tokenizer using Text.Core
        var tokenizerDir = Path.GetDirectoryName(vocabPath)!;
        var tokenizer = await TokenizerFactory.CreateWordPieceAsync(tokenizerDir, options.MaxSequenceLength);

        // Load inference engine
        var engine = OnnxInferenceEngine.Create(modelPath, options.Provider);

        // Create pooling strategy
        var poolingStrategy = PoolingFactory.Create(options.PoolingMode);

        return new EmbeddingModel(modelId, engine, tokenizer, poolingStrategy, options, loadedModelInfo);
    }

    /// <summary>
    /// Gets a list of pre-configured model IDs available for download.
    /// </summary>
    public static IEnumerable<string> GetAvailableModels() => ModelRegistry.GetAvailableModels();

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
}
