using LocalAI.Download;
using LocalAI.Generator.Abstractions;
using LocalAI.Generator.ChatFormatters;

namespace LocalAI.Generator.Internal;

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
        GeneratorModelOptions options,
        IProgress<DownloadProgress>? progress,
        CancellationToken cancellationToken)
    {
        var cacheDir = options.CacheDirectory ?? CacheManager.GetDefaultCacheDirectory();
        using var downloader = new HuggingFaceDownloader(cacheDir);

        // Download model files (GenAI models need specific files)
        var modelPath = await downloader.DownloadModelAsync(
            modelId,
            files: GenAIModelFiles,
            revision: "main",
            subfolder: null,
            progress: progress,
            cancellationToken: cancellationToken);

        return await LoadFromPathAsync(modelPath, options, modelId);
    }

    public static Task<IGeneratorModel> LoadFromPathAsync(
        string modelPath,
        GeneratorModelOptions options,
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
