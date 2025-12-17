using LMSupply.Core.Download;
using LMSupply.Download;
using LMSupply.Generator.Abstractions;
using LMSupply.Generator.ChatFormatters;

namespace LMSupply.Generator.Internal;

/// <summary>
/// Internal class for loading generator models.
/// </summary>
internal static class GeneratorModelLoader
{
    // Files required for ONNX GenAI models
    private static readonly string[] GenAIModelFiles =
    [
        "genai_config.json",
        "model.onnx",
        "model.onnx.data",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json"
    ];

    public static async Task<IGeneratorModel> LoadAsync(
        string modelId,
        GeneratorOptions options,
        IProgress<DownloadProgress>? progress,
        CancellationToken cancellationToken)
    {
        var cacheDir = options.CacheDirectory ?? CacheManager.GetDefaultCacheDirectory();
        using var downloader = new HuggingFaceDownloader(cacheDir);

        // Look up model in registry to get subfolder
        var modelInfo = ModelRegistry.GetModel(modelId);

        string modelPath;

        if (modelInfo is not null)
        {
            // Known model: use registry subfolder
            modelPath = await downloader.DownloadModelAsync(
                modelId,
                files: GenAIModelFiles,
                revision: "main",
                subfolder: modelInfo.Subfolder,
                progress: progress,
                cancellationToken: cancellationToken);
        }
        else
        {
            // Unknown model: use auto-discovery to find ONNX files
            var (downloadedDir, _) = await downloader.DownloadWithDiscoveryAsync(
                modelId,
                preferences: ModelPreferences.Default,
                progress: progress,
                cancellationToken: cancellationToken);

            modelPath = downloadedDir;
        }

        return await LoadFromPathAsync(modelPath, options, modelId);
    }

    public static Task<IGeneratorModel> LoadFromPathAsync(
        string modelPath,
        GeneratorOptions options,
        string? modelId = null)
    {
        modelId ??= Path.GetFileName(modelPath);

        // Determine chat formatter
        var chatFormatter = options.ChatFormat != null
            ? ChatFormatterFactory.CreateByFormat(options.ChatFormat)
            : ChatFormatterFactory.Create(modelId);

        // Create and return the model
        var model = new OnnxGeneratorModel(
            modelId,
            modelPath,
            chatFormatter,
            options);

        return Task.FromResult<IGeneratorModel>(model);
    }
}
