using LMSupply.Core.Download;
using LMSupply.Download;
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
    /// <param name="progress">Optional progress reporter for model download.</param>
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
        IProgress<float>? progress = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(modelIdOrPath);

        options ??= new ImageGeneratorOptions();

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
            modelPath = await DownloadModelAsync(
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
    /// Downloads a model from HuggingFace using automatic file discovery.
    /// </summary>
    /// <remarks>
    /// Uses the HuggingFace API to discover all model files including:
    /// - ONNX model files (text_encoder, unet, vae_decoder)
    /// - External data files (.onnx_data) containing model weights
    /// - Tokenizer and configuration files
    /// This approach is more robust than hardcoding file paths.
    /// </remarks>
    private static async Task<string> DownloadModelAsync(
        string repoId,
        ImageGeneratorOptions options,
        IProgress<float>? progress,
        CancellationToken cancellationToken)
    {
        using var downloader = new HuggingFaceDownloader(options.CacheDirectory, localFilesOnly: options.DisableAutoDownload);

        // Track download progress
        var progressAdapter = progress != null
            ? new Progress<DownloadProgress>(p =>
            {
                if (p.TotalBytes > 0)
                {
                    progress.Report((float)p.BytesDownloaded / p.TotalBytes);
                }
            })
            : null;

        // Use discovery-based download to automatically find all model files
        // including ONNX models, external data files, and config/tokenizer files
        var (modelPath, _) = await downloader.DownloadWithDiscoveryAsync(
            repoId,
            preferences: new ModelPreferences
            {
                QuantizationPriority = [Quantization.Default],
                PreferredProvider = ExecutionProvider.Cpu
            },
            revision: "main",
            progress: progressAdapter,
            cancellationToken: cancellationToken);

        return modelPath;
    }
}
