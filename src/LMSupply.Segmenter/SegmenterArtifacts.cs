using LMSupply.Lifecycle;

namespace LMSupply.Segmenter;

/// <summary>
/// <see cref="IModelArtifacts"/> for image segmenters, semantic and interactive alike: whether a <see cref="LocalSegmenter"/>
/// load with downloads disabled would find the model's files and the ONNX Runtime it runs on in the cache, what is left to
/// download, the download that makes it so, and the delete. The options are the ones the load will use (provider and cache
/// directory decide which files that is).
/// </summary>
public sealed class SegmenterArtifacts : IModelArtifacts
{
    private readonly SegmenterOptions _options;

    /// <param name="options">The options later loads use; copied, so later changes to the instance do not apply.</param>
    public SegmenterArtifacts(SegmenterOptions? options = null)
    {
        _options = options?.Clone() ?? new SegmenterOptions();
    }

    /// <inheritdoc />
    public string Kind => "segmenter";

    /// <inheritdoc />
    public async Task<bool> IsAvailableOfflineAsync(string modelId, CancellationToken cancellationToken = default)
        => await LocalSegmenter.IsModelDownloadedAsync(modelId, _options, cancellationToken)
           && await OnnxRuntimeArtifacts.IsAvailableOfflineAsync(_options.Provider, cancellationToken);

    /// <inheritdoc />
    public async Task<RemainingDownload> GetRemainingDownloadAsync(string modelId, CancellationToken cancellationToken = default)
        => new(await LocalSegmenter.GetRemainingDownloadBytesAsync(modelId, _options, cancellationToken),
            await OnnxRuntimeArtifacts.GetRemainingDownloadBytesAsync(_options.Provider, cancellationToken));

    /// <inheritdoc />
    public async Task DownloadAsync(string modelId, IProgress<DownloadProgress>? progress = null, CancellationToken cancellationToken = default)
    {
        await LocalSegmenter.DownloadModelAsync(modelId, _options, progress, cancellationToken);
        await OnnxRuntimeArtifacts.DownloadAsync(_options.Provider, progress, cancellationToken);
    }

    /// <inheritdoc />
    public bool Delete(string modelId) => LocalSegmenter.DeleteModel(modelId, _options);
}
