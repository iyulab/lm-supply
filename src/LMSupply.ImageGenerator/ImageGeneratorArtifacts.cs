using LMSupply.Lifecycle;

namespace LMSupply.ImageGenerator;

/// <summary>
/// <see cref="IModelArtifacts"/> for image generators: whether a <see cref="LocalImageGenerator"/> load with downloads
/// disabled would find the pipeline's model files and the ONNX Runtime it runs on in the cache, what is left to download,
/// the download that makes it so, and the delete. The options are the ones the load will use (provider and cache directory
/// decide which files that is).
/// </summary>
public sealed class ImageGeneratorArtifacts : IModelArtifacts
{
    private readonly ImageGeneratorOptions _options;

    /// <param name="options">The options later loads use; copied, so later changes to the instance do not apply.</param>
    public ImageGeneratorArtifacts(ImageGeneratorOptions? options = null)
    {
        _options = options?.Clone() ?? new ImageGeneratorOptions();
    }

    /// <inheritdoc />
    public string Kind => "imagegenerator";

    /// <inheritdoc />
    public async Task<bool> IsAvailableOfflineAsync(string modelId, CancellationToken cancellationToken = default)
        => await LocalImageGenerator.IsModelDownloadedAsync(modelId, _options, cancellationToken)
           && await OnnxRuntimeArtifacts.IsAvailableOfflineAsync(_options.Provider, cancellationToken);

    /// <inheritdoc />
    public async Task<RemainingDownload> GetRemainingDownloadAsync(string modelId, CancellationToken cancellationToken = default)
        => new(await LocalImageGenerator.GetRemainingDownloadBytesAsync(modelId, _options, cancellationToken),
            await OnnxRuntimeArtifacts.GetRemainingDownloadBytesAsync(_options.Provider, cancellationToken));

    /// <inheritdoc />
    public async Task DownloadAsync(string modelId, IProgress<DownloadProgress>? progress = null, CancellationToken cancellationToken = default)
    {
        await LocalImageGenerator.DownloadModelAsync(modelId, _options, progress, cancellationToken);
        await OnnxRuntimeArtifacts.DownloadAsync(_options.Provider, progress, cancellationToken);
    }

    /// <inheritdoc />
    public bool Delete(string modelId) => LocalImageGenerator.DeleteModel(modelId, _options);
}
