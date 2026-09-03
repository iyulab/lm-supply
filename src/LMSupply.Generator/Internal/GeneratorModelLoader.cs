using LMSupply.Core.Download;
using LMSupply.Download;
using LMSupply.Generator.Abstractions;
using LMSupply.Generator.ChatFormatters;
using LMSupply.Generator.Internal.Llama;
using LMSupply.Generator.Models;

namespace LMSupply.Generator.Internal;

/// <summary>
/// Internal class for loading generator models.
/// </summary>
internal static class GeneratorModelLoader
{
    public static async Task<IGeneratorModel> LoadAsync(
        string modelId,
        GeneratorOptions options,
        IProgress<DownloadProgress>? progress,
        CancellationToken cancellationToken)
    {
        // Detect model format from model ID
        var format = ModelFormatDetector.Detect(modelId);

        // Route to appropriate loader based on format
        return format switch
        {
            ModelFormat.Gguf => await LoadGgufAsync(modelId, options, progress, cancellationToken),
            ModelFormat.Onnx => await LoadOnnxAsync(modelId, options, progress, cancellationToken),
            ModelFormat.Unknown => await LoadGgufAsync(modelId, options, progress, cancellationToken), // GGUF fallback (GGUF-first strategy)
            _ => throw new NotSupportedException($"Unsupported model format: {format}")
        };
    }

    /// <summary>
    /// Loads an ONNX GenAI model from HuggingFace.
    /// </summary>
    private static async Task<IGeneratorModel> LoadOnnxAsync(
        string modelId,
        GeneratorOptions options,
        IProgress<DownloadProgress>? progress,
        CancellationToken cancellationToken)
    {
        // Ensure GenAI runtime binaries are available before loading the model
        await OnnxGeneratorBackendRegistry.Require().EnsureRuntimeAsync(options.Provider, progress, cancellationToken);

        var cacheDir = options.CacheDirectory ?? CacheManager.GetDefaultCacheDirectory();
        using var downloader = new HuggingFaceDownloader(cacheDir);

        // Look up model in registry to get subfolder preference
        GeneratorModelRegistry.Default.TryResolve(modelId, out var modelInfo);

        // Build preferences from registry info if available
        var hwPrefs = ModelPreferences.ForCurrentHardware();
        ModelPreferences preferences;
        if (modelInfo?.Subfolder != null)
        {
            preferences = new ModelPreferences { PreferredSubfolder = modelInfo.Subfolder };
        }
        else if (options.QuantizationHint is { } hint)
        {
            preferences = new ModelPreferences
            {
                PreferLowMemory = hwPrefs.PreferLowMemory,
                QuantizationPriority = ModelPreferences.ForQuantizationHint(hint).QuantizationPriority,
                PreferredProvider = options.Provider != ExecutionProvider.Auto
                    ? options.Provider : hwPrefs.PreferredProvider
            };
        }
        else
        {
            preferences = hwPrefs;
        }

        // Use discovery-based download for all models
        // This handles dynamic ONNX file names (e.g., phi-3.5-mini-instruct-*.onnx)
        var (basePath, discovery) = await downloader.DownloadWithDiscoveryAsync(
            modelId,
            preferences: preferences,
            progress: progress,
            cancellationToken: cancellationToken);

        // Build the actual model path including subfolder if present
        var modelPath = discovery.Subfolder != null
            ? Path.Combine(basePath, discovery.Subfolder.Replace('/', Path.DirectorySeparatorChar))
            : basePath;

        // Pass basePath as configBasePath when subfolder is used,
        // so GenAiConfigReader can find genai_config.json at either location
        var configBasePath = discovery.Subfolder != null ? basePath : null;
        return await LoadFromPathAsync(modelPath, options, modelId, configBasePath);
    }

    /// <summary>
    /// Loads a GGUF model from HuggingFace using llama-server backend.
    /// </summary>
    private static async Task<IGeneratorModel> LoadGgufAsync(
        string modelId,
        GeneratorOptions options,
        IProgress<DownloadProgress>? progress,
        CancellationToken cancellationToken)
    {
        var cacheDir = options.CacheDirectory ?? CacheManager.GetDefaultCacheDirectory();

        // Try to resolve as registry alias first
        var registryInfo = GgufModelRegistry.Resolve(modelId);
        string modelPath;
        string chatFormat;

        if (registryInfo != null)
        {
            // Download from registry
            using var downloader = new GgufModelDownloader(cacheDir);
            modelPath = await downloader.DownloadFromRegistryAsync(
                registryInfo,
                provider: options.Provider,
                preferredQuantization: null,
                progress: progress,
                cancellationToken: cancellationToken);

            chatFormat = options.ChatFormat ?? registryInfo.ChatFormat;
        }
        else
        {
            // Guard: "gguf:*" prefixed IDs are alias-only — passing to HF would use "gguf" as repo ID and 401.
            if (modelId.StartsWith("gguf:", StringComparison.OrdinalIgnoreCase))
            {
                var known = string.Join(", ", GgufModelRegistry.GetAliases());
                throw new ArgumentException(
                    $"'{modelId}' is not a registered GGUF alias. Known aliases: {known}. " +
                    $"Register it in GgufModelRegistry or use a full HuggingFace repo ID (without the 'gguf:' prefix).",
                    nameof(modelId));
            }

            // Assume it's a HuggingFace repo ID
            using var downloader = new GgufModelDownloader(cacheDir);
            modelPath = await downloader.DownloadAsync(
                modelId,
                filename: null,
                preferredQuantization: null,
                progress: progress,
                cancellationToken: cancellationToken);

            // Detect chat format from filename
            chatFormat = options.ChatFormat ?? GgufChatFormatDetector.DetectFromFilename(modelPath);
        }

        // Load the model from downloaded path using llama-server
        var resolvedModelId = registryInfo?.DisplayName ?? modelId;
        var chatFormatter = ChatFormatterFactory.CreateByFormat(chatFormat);

        return await LlamaServerGeneratorModel.LoadAsync(
            resolvedModelId,
            modelPath,
            chatFormatter,
            options,
            progress,
            cancellationToken);
    }

    public static async Task<IGeneratorModel> LoadFromPathAsync(
        string modelPath,
        GeneratorOptions options,
        string? modelId = null,
        string? configBasePath = null)
    {
        // Detect model format from path
        var format = ModelFormatDetector.Detect(modelPath);

        // Route to appropriate loader based on format
        return format switch
        {
            ModelFormat.Gguf => await LoadGgufFromPathAsync(modelPath, options, modelId),
            ModelFormat.Onnx => await LoadOnnxFromPathAsync(modelPath, options, modelId, configBasePath),
            ModelFormat.Unknown => await LoadGgufFromPathAsync(modelPath, options, modelId), // GGUF fallback (GGUF-first strategy)
            _ => throw new NotSupportedException($"Unsupported model format: {format}")
        };
    }

    /// <summary>
    /// Loads an ONNX GenAI model from a local path.
    /// </summary>
    private static async Task<IGeneratorModel> LoadOnnxFromPathAsync(
        string modelPath,
        GeneratorOptions options,
        string? modelId = null,
        string? configBasePath = null)
    {
        // Ensure GenAI runtime binaries are available before loading the model
        await OnnxGeneratorBackendRegistry.Require().EnsureRuntimeAsync(options.Provider, progress: null, CancellationToken.None);

        modelId ??= Path.GetFileName(modelPath);

        // Determine chat formatter
        var chatFormatter = options.ChatFormat != null
            ? ChatFormatterFactory.CreateByFormat(options.ChatFormat)
            : ChatFormatterFactory.Create(modelId);

        // Create and return the model
        return OnnxGeneratorBackendRegistry.Require().CreateModel(
            modelId,
            modelPath,
            chatFormatter,
            options,
            configBasePath);
    }

    /// <summary>
    /// Loads a GGUF model from a local path using llama-server backend.
    /// </summary>
    private static async Task<IGeneratorModel> LoadGgufFromPathAsync(
        string modelPath,
        GeneratorOptions options,
        string? modelId = null)
    {
        modelId ??= Path.GetFileNameWithoutExtension(modelPath);

        // Detect chat format from filename or use provided
        var chatFormat = options.ChatFormat ?? GgufChatFormatDetector.DetectFromFilename(modelPath);
        var chatFormatter = ChatFormatterFactory.CreateByFormat(chatFormat);

        return await LlamaServerGeneratorModel.LoadAsync(
            modelId,
            modelPath,
            chatFormatter,
            options,
            progress: null,
            CancellationToken.None);
    }
}
