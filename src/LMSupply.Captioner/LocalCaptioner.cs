using LMSupply.Captioner.Inference;
using LMSupply.Captioner.Models;
using LMSupply.Core.Download;
using LMSupply.Download;
using LMSupply.Exceptions;

namespace LMSupply.Captioner;

/// <summary>
/// Main entry point for loading and using image captioning models.
/// </summary>
public static class LocalCaptioner
{
    /// <summary>
    /// Loads a captioning model by name or path.
    /// </summary>
    /// <param name="modelIdOrPath">
    /// Either a model ID (e.g., "default", "vit-gpt2") for auto-download,
    /// or a local path to a model directory.
    /// </param>
    /// <param name="options">Optional configuration options.</param>
    /// <param name="progress">Optional progress reporting for downloads.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>A loaded captioning model ready for inference.</returns>
    public static async Task<ICaptionerModel> LoadAsync(
        string modelIdOrPath,
        CaptionerOptions? options = null,
        IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(modelIdOrPath);
        options ??= new CaptionerOptions();

        ModelInfo? modelInfo = null;
        string modelDir;

        // Check if it's a local directory path
        if (Directory.Exists(modelIdOrPath))
        {
            modelDir = modelIdOrPath;

            // Try to infer model info from directory contents
            if (!TryInferModelInfo(modelDir, out modelInfo))
            {
                throw new ModelNotFoundException(
                    $"Could not determine model type from directory: {modelDir}",
                    modelIdOrPath);
            }
        }
        // Check if it's a known model alias
        else if (ModelRegistry.TryGetModel(modelIdOrPath, out modelInfo))
        {
            // Download model from HuggingFace
            var cacheDir = options.CacheDirectory ?? CacheManager.GetDefaultCacheDirectory();
            using var downloader = new HuggingFaceDownloader(cacheDir);

            // Build model-specific file list
            var modelFiles = GetRequiredFiles(modelInfo);

            modelDir = await downloader.DownloadModelAsync(
                modelInfo.RepoId,
                files: modelFiles,
                subfolder: modelInfo.Subfolder,
                progress: progress,
                cancellationToken: cancellationToken).ConfigureAwait(false);
        }
        // Check if it's a HuggingFace repo ID
        else if (modelIdOrPath.Contains('/'))
        {
            var cacheDir = options.CacheDirectory ?? CacheManager.GetDefaultCacheDirectory();
            using var downloader = new HuggingFaceDownloader(cacheDir);

            // Use auto-discovery to find ONNX files and config
            var (downloadedDir, discovery) = await downloader.DownloadWithDiscoveryAsync(
                modelIdOrPath,
                preferences: ModelPreferences.Default,
                progress: progress,
                cancellationToken: cancellationToken).ConfigureAwait(false);

            modelDir = downloadedDir;

            // Try to infer model info
            if (!TryInferModelInfo(modelDir, out modelInfo))
            {
                throw new ModelNotFoundException(
                    $"Could not determine model type for: {modelIdOrPath}. " +
                    $"Discovered ONNX files: [{string.Join(", ", discovery.OnnxFiles)}]. " +
                    "Use a known model ID (e.g., 'default', 'vit-gpt2') or ensure the model follows a supported format.",
                    modelIdOrPath);
            }
        }
        else
        {
            throw new ModelNotFoundException(
                $"Unknown model '{modelIdOrPath}'. Use a known model ID (e.g., 'default', 'vit-gpt2'), " +
                "a HuggingFace repo ID (e.g., 'Xenova/vit-gpt2-image-captioning'), " +
                "or a local path to a model directory.",
                modelIdOrPath);
        }

        // Create the appropriate captioner based on model type
        // modelInfo is guaranteed non-null here due to control flow above
        return await CreateCaptionerAsync(modelDir, modelInfo!, options).ConfigureAwait(false);
    }

    /// <summary>
    /// Gets a list of pre-configured model IDs available for download.
    /// </summary>
    public static IEnumerable<string> GetAvailableModels() => ModelRegistry.GetAvailableModels();

    private static async Task<ICaptionerModel> CreateCaptionerAsync(
        string modelDir,
        ModelInfo modelInfo,
        CaptionerOptions options)
    {
        // Currently only ViT-GPT2 style models are supported
        // Future: Add support for Florence-2, SmolVLM, etc.
        return await VitGpt2Captioner.CreateAsync(modelDir, modelInfo, options).ConfigureAwait(false);
    }

    private static bool TryInferModelInfo(string modelDir, out ModelInfo? modelInfo)
    {
        // Check for ViT-GPT2 style model (encoder_model.onnx + decoder_model_merged.onnx)
        var encoderPath = Path.Combine(modelDir, "encoder_model.onnx");
        var decoderPath = Path.Combine(modelDir, "decoder_model_merged.onnx");

        if (File.Exists(encoderPath) && File.Exists(decoderPath))
        {
            // Check for vocab.json (GPT-2 tokenizer)
            var vocabPath = Path.Combine(modelDir, "vocab.json");
            if (File.Exists(vocabPath))
            {
                modelInfo = ModelRegistry.GetModel("vit-gpt2");
                return true;
            }
        }

        modelInfo = null;
        return false;
    }

    /// <summary>
    /// Gets the list of required files for a model.
    /// </summary>
    private static IEnumerable<string> GetRequiredFiles(ModelInfo modelInfo)
    {
        // ONNX model files
        yield return modelInfo.EncoderFile;
        yield return modelInfo.DecoderFile;

        // Additional model files if specified
        foreach (var file in modelInfo.AdditionalFiles)
        {
            yield return file;
        }

        // Common tokenizer and config files (these are typically in root, not subfolder)
        yield return "config.json";
        yield return "vocab.json";
        yield return "merges.txt";
        yield return "tokenizer.json";
        yield return "tokenizer_config.json";
        yield return "special_tokens_map.json";
    }
}
