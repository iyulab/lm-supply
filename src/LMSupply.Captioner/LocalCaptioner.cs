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
    /// Gets the model registry for the Captioner domain.
    /// Provides access to model resolution, alias management, and model enumeration.
    /// </summary>
    public static IModelRegistry<ModelInfo> Registry => CaptionerModelRegistry.Default;

    /// <summary>
    /// Shared pool for named model management. Supports GetOrLoadAsync / UnloadAsync by model ID.
    /// </summary>
    public static LMSupply.Pool.ModelPool<ICaptionerModel, CaptionerOptions> Pool { get; }
        = new(new Pool.CaptionerLoader());

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

        // Parse variant qualifier (e.g., "default:fp16" → modelId="default", hint="fp16")
        var (baseId, qualifier) = LMSupplyOptionsBase.SplitQualifier(modelIdOrPath);
        modelIdOrPath = baseId;
        options.QuantizationHint ??= qualifier;

        ModelInfo? modelInfo = null;
        string modelDir;
        string? tokenizerDir = null; // Separate tokenizer directory for HuggingFace repos with subfolders

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
        // Check if it's a known model alias (only system/user aliases, not HF fallbacks)
        else if (TryResolveKnownModel(modelIdOrPath, out modelInfo))
        {
            // Download model from HuggingFace
            var cacheDir = options.CacheDirectory ?? CacheManager.GetDefaultCacheDirectory();
            using var downloader = new HuggingFaceDownloader(cacheDir);

            // Build model-specific file list
            var modelFiles = GetRequiredFiles(modelInfo!);

            modelDir = await downloader.DownloadModelAsync(
                modelInfo!.RepoId,
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
                cancellationToken: cancellationToken).ConfigureAwait(false);

            // Use discovery result to get correct ONNX directory (handles subfolder structures)
            var onnxDir = discovery.GetOnnxDirectory(downloadedDir);

            // Try to infer model info using the ONNX directory
            if (!TryInferModelInfo(onnxDir, downloadedDir, out modelInfo))
            {
                throw new ModelNotFoundException(
                    $"Could not determine model type for: {modelIdOrPath}. " +
                    $"Discovered ONNX files: [{string.Join(", ", discovery.OnnxFiles)}]. " +
                    "Use a known model ID (e.g., 'default', 'vit-gpt2') or ensure the model follows a supported format.",
                    modelIdOrPath);
            }

            // ONNX files are in onnxDir, tokenizer files are in base downloadedDir
            modelDir = onnxDir;
            tokenizerDir = downloadedDir;
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
        return await CreateCaptionerAsync(modelDir, modelInfo!, options, tokenizerDir).ConfigureAwait(false);
    }

    /// <summary>
    /// Gets a list of pre-configured model IDs available for download.
    /// </summary>
    public static IEnumerable<string> GetAvailableModels() =>
        CaptionerModelRegistry.Default.GetAliases().Select(a => a.Name);

    /// <summary>
    /// Gets all registered model information, deduplicated by model id — aliases that point at the
    /// same model (e.g. "default" and "fast") contribute one entry. Use <see cref="GetAvailableModels"/>
    /// for the alias names themselves.
    /// </summary>
    public static IEnumerable<ModelInfo> GetAllModels() =>
        CaptionerModelRegistry.Default.GetAvailableModels();

    private static async Task<ICaptionerModel> CreateCaptionerAsync(
        string modelDir,
        ModelInfo modelInfo,
        CaptionerOptions options,
        string? tokenizerDir = null)
    {
        // Currently only ViT-GPT2 style models are supported
        // Future: Add support for Florence-2, SmolVLM, etc.
        return await VitGpt2Captioner.CreateAsync(modelDir, modelInfo, options, tokenizerDir).ConfigureAwait(false);
    }

    /// <summary>
    /// Resolves only known models (system aliases, user aliases, registered IDs).
    /// Does NOT create fallback for unknown HuggingFace repos, preserving the
    /// auto-discovery download path for truly unknown models.
    /// </summary>
    private static bool TryResolveKnownModel(string modelIdOrAlias, out ModelInfo? modelInfo)
    {
        var registry = CaptionerModelRegistry.Default;

        // Check if it would resolve to a fallback (contains '/' but not registered)
        // In that case, return false so the auto-discovery path is used instead
        if (modelIdOrAlias.Contains('/'))
        {
            // Only resolve if it's a registered model ID (not a fallback)
            var knownModels = registry.GetAvailableModels();
            var match = knownModels.FirstOrDefault(m =>
                m.RepoId.Equals(modelIdOrAlias, StringComparison.OrdinalIgnoreCase));
            if (match is not null)
            {
                modelInfo = match;
                return true;
            }
            modelInfo = null;
            return false;
        }

        return registry.TryResolve(modelIdOrAlias, out modelInfo);
    }

    private static bool TryInferModelInfo(string modelDir, out ModelInfo? modelInfo)
    {
        return TryInferModelInfo(modelDir, modelDir, out modelInfo);
    }

    /// <summary>
    /// Tries to infer model info from directory contents.
    /// Handles cases where ONNX files and tokenizer files are in different directories.
    /// </summary>
    /// <param name="onnxDir">Directory containing ONNX model files.</param>
    /// <param name="baseDir">Base directory containing tokenizer/config files.</param>
    /// <param name="modelInfo">The inferred model info if successful.</param>
    /// <returns>True if model info was successfully inferred.</returns>
    private static bool TryInferModelInfo(string onnxDir, string baseDir, out ModelInfo? modelInfo)
    {
        // Check for ViT-GPT2 style model (encoder_model.onnx + decoder_model_merged.onnx)
        var encoderPath = Path.Combine(onnxDir, "encoder_model.onnx");
        var decoderPath = Path.Combine(onnxDir, "decoder_model_merged.onnx");

        if (File.Exists(encoderPath) && File.Exists(decoderPath))
        {
            // Check for vocab.json (GPT-2 tokenizer) - may be in base dir or onnx dir
            var vocabPath = Path.Combine(baseDir, "vocab.json");
            if (!File.Exists(vocabPath))
            {
                vocabPath = Path.Combine(onnxDir, "vocab.json");
            }

            if (File.Exists(vocabPath))
            {
                modelInfo = CaptionerModelRegistry.Default.Resolve("default");
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
