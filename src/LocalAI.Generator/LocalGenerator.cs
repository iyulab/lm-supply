using LocalAI.Generator.Abstractions;

namespace LocalAI.Generator;

/// <summary>
/// Factory class for creating local text generation models.
/// </summary>
public static class LocalGenerator
{
    /// <summary>
    /// Default model to use when no model is specified.
    /// </summary>
    public const string DefaultModel = "microsoft/Phi-3.5-mini-instruct-onnx";

    /// <summary>
    /// Creates a text generator from a HuggingFace model repository.
    /// </summary>
    /// <param name="modelId">The HuggingFace model identifier (e.g., "microsoft/Phi-3.5-mini-instruct-onnx").</param>
    /// <param name="options">Model loading options.</param>
    /// <param name="progress">Progress callback for model downloading.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>A text generator instance.</returns>
    public static Task<IGeneratorModel> CreateAsync(
        string modelId,
        GeneratorModelOptions? options = null,
        IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(modelId);
        options ??= new GeneratorModelOptions();

        return Internal.GeneratorModelLoader.LoadAsync(modelId, options, progress, cancellationToken);
    }

    /// <summary>
    /// Creates a text generator from a local model directory.
    /// </summary>
    /// <param name="modelPath">The path to the local model directory.</param>
    /// <param name="options">Model loading options.</param>
    /// <returns>A text generator instance.</returns>
    public static Task<IGeneratorModel> CreateFromPathAsync(
        string modelPath,
        GeneratorModelOptions? options = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(modelPath);

        if (!Directory.Exists(modelPath))
        {
            throw new DirectoryNotFoundException($"Model directory not found: {modelPath}");
        }

        options ??= new GeneratorModelOptions();

        return Internal.GeneratorModelLoader.LoadFromPathAsync(modelPath, options);
    }

    /// <summary>
    /// Creates a text generator using the default model.
    /// </summary>
    /// <param name="options">Model loading options.</param>
    /// <param name="progress">Progress callback for model downloading.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>A text generator instance.</returns>
    public static Task<IGeneratorModel> CreateDefaultAsync(
        GeneratorModelOptions? options = null,
        IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        return CreateAsync(DefaultModel, options, progress, cancellationToken);
    }
}
