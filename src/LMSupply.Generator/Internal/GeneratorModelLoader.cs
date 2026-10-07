using LMSupply.Core.Download;
using LMSupply.Download;
using LMSupply.Generator.Abstractions;
using LMSupply.Generator.ChatFormatters;
using LMSupply.Generator.Gguf;
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
    /// Downloads the weights <see cref="LoadAsync"/> would load for <paramref name="modelId"/> and
    /// returns their local path, without loading anything (no runtime provisioning, no llama-server,
    /// no session). Same format routing as <see cref="LoadAsync"/>.
    /// </summary>
    public static async Task<string> DownloadAsync(
        string modelId,
        GeneratorOptions options,
        IProgress<DownloadProgress>? progress,
        CancellationToken cancellationToken)
    {
        var format = ModelFormatDetector.Detect(modelId);
        return format switch
        {
            ModelFormat.Gguf => (await ResolveGgufAsync(modelId, options, progress, cancellationToken)).ModelPath,
            ModelFormat.Onnx => (await DownloadOnnxAsync(modelId, options, progress, cancellationToken)).ModelPath,
            ModelFormat.Unknown => (await ResolveGgufAsync(modelId, options, progress, cancellationToken)).ModelPath,
            _ => throw new NotSupportedException($"Unsupported model format: {format}")
        };
    }

    /// <summary>
    /// The weight files <see cref="DownloadAsync"/> would fetch for the same arguments, with their listed lengths —
    /// the same format routing and the same file choice, so the plan and the download cannot disagree. Downloads nothing.
    /// </summary>
    public static async Task<DownloadPlan> PlanDownloadAsync(
        string modelId,
        GeneratorOptions options,
        CancellationToken cancellationToken)
    {
        var cacheDir = options.CacheDirectory ?? CacheManager.GetDefaultCacheDirectory();
        var format = ModelFormatDetector.Detect(modelId);
        switch (format)
        {
            case ModelFormat.Onnx:
            {
                using var downloader = new HuggingFaceDownloader(cacheDir, localFilesOnly: options.DisableAutoDownload);
                return await downloader.PlanWithDiscoveryAsync(
                    modelId, OnnxPreferences(modelId, options), cancellationToken: cancellationToken);
            }
            case ModelFormat.Gguf or ModelFormat.Unknown:
            {
                using var downloader = new GgufModelDownloader(cacheDir, localFilesOnly: options.DisableAutoDownload);
                var registryInfo = GgufModelRegistry.Resolve(modelId, options.SelectionProvider, options.MaxContextLength, options.AutoSelectionGoal);
                if (registryInfo is not null)
                {
                    return await downloader.PlanFromRegistryAsync(
                        registryInfo, options.SelectionProvider, options.MaxContextLength, cancellationToken);
                }

                ThrowIfUnregisteredGgufAlias(modelId);
                return await downloader.PlanAsync(modelId, cancellationToken);
            }
            default:
                throw new NotSupportedException($"Unsupported model format: {format}");
        }
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
        // The backend check is cheap and comes first; the runtime binaries come after the model files
        // (LoadFromPathAsync provisions them), so a load that DisableAutoDownload refuses has not pulled the
        // GenAI runtime on its way to the refusal.
        OnnxGeneratorBackendRegistry.Require();
        var (modelPath, configBasePath) = await DownloadOnnxAsync(modelId, options, progress, cancellationToken);
        return await LoadFromPathAsync(modelPath, options, modelId, configBasePath, progress, cancellationToken);
    }

    /// <summary>
    /// The download half of <see cref="LoadOnnxAsync"/>: resolves registry/hardware preferences and
    /// pulls the ONNX files via discovery. Returns the model path (subfolder included when the
    /// export uses one) and the base path GenAiConfigReader needs when a subfolder is in play.
    /// </summary>
    private static async Task<(string ModelPath, string? ConfigBasePath)> DownloadOnnxAsync(
        string modelId,
        GeneratorOptions options,
        IProgress<DownloadProgress>? progress,
        CancellationToken cancellationToken)
    {
        var cacheDir = options.CacheDirectory ?? CacheManager.GetDefaultCacheDirectory();
        using var downloader = new HuggingFaceDownloader(cacheDir, localFilesOnly: options.DisableAutoDownload);

        // Use discovery-based download for all models
        // This handles dynamic ONNX file names (e.g., phi-3.5-mini-instruct-*.onnx)
        var (basePath, discovery) = await downloader.DownloadWithDiscoveryAsync(
            modelId,
            preferences: OnnxPreferences(modelId, options),
            progress: progress,
            cancellationToken: cancellationToken);

        // Build the actual model path including subfolder if present
        var modelPath = discovery.Subfolder != null
            ? Path.Combine(basePath, discovery.Subfolder.Replace('/', Path.DirectorySeparatorChar))
            : basePath;

        // Pass basePath as configBasePath when subfolder is used,
        // so GenAiConfigReader can find genai_config.json at either location
        var configBasePath = discovery.Subfolder != null ? basePath : null;
        return (modelPath, configBasePath);
    }

    // The discovery preferences an ONNX download uses — shared by the download and the plan.
    private static ModelPreferences OnnxPreferences(string modelId, GeneratorOptions options)
    {
        // Look up model in registry to get subfolder preference
        GeneratorModelRegistry.Default.TryResolve(modelId, out var modelInfo);

        // Build preferences from registry info if available
        var hwPrefs = ModelPreferences.ForProvider(options.Provider);
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

        return preferences;
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
        var (modelPath, chatFormat, registryInfo) = await ResolveGgufAsync(modelId, options, progress, cancellationToken);

        // Load the model from downloaded path using llama-server. The identity describes the file the
        // download settled on, which is not the alias's default when a smaller quantization stood in.
        var identity = GgufLoadIdentity.Describe(modelId, registryInfo, modelPath);
        var chatFormatter = ChatFormatterFactory.CreateByFormat(chatFormat);

        return await LlamaServerGeneratorModel.LoadAsync(
            identity,
            modelPath,
            chatFormatter,
            options,
            progress,
            cancellationToken);
    }

    /// <summary>
    /// The download half of <see cref="LoadGgufAsync"/>: resolves a registry alias or a HuggingFace
    /// repo id to a local GGUF file (auto-quantization under <see cref="LMSupplyOptionsBase.Provider"/>'s
    /// memory budget) and the chat format to drive it with. Shared with <see cref="DownloadAsync"/>
    /// so a cache warmed ahead of time holds exactly the file a later load opens.
    /// </summary>
    private static async Task<(string ModelPath, string ChatFormat, GgufModelInfo? RegistryInfo)> ResolveGgufAsync(
        string modelId,
        GeneratorOptions options,
        IProgress<DownloadProgress>? progress,
        CancellationToken cancellationToken)
    {
        var cacheDir = options.CacheDirectory ?? CacheManager.GetDefaultCacheDirectory();

        // Try to resolve as registry alias first
        var registryInfo = GgufModelRegistry.Resolve(modelId, options.SelectionProvider, options.MaxContextLength, options.AutoSelectionGoal);
        string modelPath;
        string chatFormat;

        if (registryInfo != null)
        {
            // Download from registry
            using var downloader = new GgufModelDownloader(cacheDir, localFilesOnly: options.DisableAutoDownload);
            modelPath = await downloader.DownloadFromRegistryAsync(
                registryInfo,
                provider: options.SelectionProvider,
                preferredQuantization: null,
                progress: progress,
                contextLength: options.MaxContextLength,
                cancellationToken: cancellationToken);

            chatFormat = options.ChatFormat ?? registryInfo.ChatFormat;
        }
        else
        {
            ThrowIfUnregisteredGgufAlias(modelId);

            // Assume it's a HuggingFace repo ID
            using var downloader = new GgufModelDownloader(cacheDir, localFilesOnly: options.DisableAutoDownload);
            modelPath = await downloader.DownloadAsync(
                modelId,
                filename: null,
                preferredQuantization: null,
                progress: progress,
                cancellationToken: cancellationToken);

            // Detect chat format from filename
            chatFormat = options.ChatFormat ?? GgufChatFormatDetector.DetectFromFilename(modelPath);
        }

        return (modelPath, chatFormat, registryInfo);
    }

    // "gguf:*" prefixed IDs are alias-only — passing one to HF would use "gguf" as the repo ID and get a 401.
    private static void ThrowIfUnregisteredGgufAlias(string modelId)
    {
        if (!modelId.StartsWith("gguf:", StringComparison.OrdinalIgnoreCase))
            return;

        var known = string.Join(", ", GgufModelRegistry.GetAliases());
        throw new ArgumentException(
            $"'{modelId}' is not a registered GGUF alias. Known aliases: {known}. " +
            $"Register it in GgufModelRegistry or use a full HuggingFace repo ID (without the 'gguf:' prefix).",
            nameof(modelId));
    }

    /// <summary>
    /// Loads a model already on disk. The runtime it needs (the GenAI binaries, or llama-server for GGUF) may still
    /// be downloaded on first use, and <paramref name="progress"/> reports that download.
    /// </summary>
    public static async Task<IGeneratorModel> LoadFromPathAsync(
        string modelPath,
        GeneratorOptions options,
        string? modelId,
        string? configBasePath,
        IProgress<DownloadProgress>? progress,
        CancellationToken cancellationToken)
    {
        // Detect model format from path
        var format = ModelFormatDetector.Detect(modelPath);

        // Route to appropriate loader based on format
        return format switch
        {
            ModelFormat.Gguf => await LoadGgufFromPathAsync(modelPath, options, modelId, progress, cancellationToken),
            ModelFormat.Onnx => await LoadOnnxFromPathAsync(modelPath, options, modelId, configBasePath, progress, cancellationToken),
            ModelFormat.Unknown => await LoadGgufFromPathAsync(modelPath, options, modelId, progress, cancellationToken), // GGUF fallback (GGUF-first strategy)
            _ => throw new NotSupportedException($"Unsupported model format: {format}")
        };
    }

    /// <summary>
    /// Loads an ONNX GenAI model from a local path.
    /// </summary>
    private static async Task<IGeneratorModel> LoadOnnxFromPathAsync(
        string modelPath,
        GeneratorOptions options,
        string? modelId,
        string? configBasePath,
        IProgress<DownloadProgress>? progress,
        CancellationToken cancellationToken)
    {
        // Ensure GenAI runtime binaries are available before loading the model
        await OnnxGeneratorBackendRegistry.Require().EnsureRuntimeAsync(options.Provider, progress, cancellationToken);

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
        string? modelId,
        IProgress<DownloadProgress>? progress,
        CancellationToken cancellationToken)
    {
        modelId ??= Path.GetFileNameWithoutExtension(modelPath);

        // Detect chat format from filename or use provided
        var chatFormat = options.ChatFormat ?? GgufChatFormatDetector.DetectFromFilename(modelPath);
        var chatFormatter = ChatFormatterFactory.CreateByFormat(chatFormat);

        return await LlamaServerGeneratorModel.LoadAsync(
            GgufLoadIdentity.Describe(modelId, registryInfo: null, modelPath),
            modelPath,
            chatFormatter,
            options,
            progress,
            cancellationToken);
    }
}
