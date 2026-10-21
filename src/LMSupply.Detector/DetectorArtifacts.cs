using LMSupply.Lifecycle;

namespace LMSupply.Detector;

/// <summary>
/// <see cref="IModelArtifacts"/> for object detectors (and the pose and face models of the same package): whether a
/// <see cref="LocalDetector"/> load with downloads disabled would find the model file and the ONNX Runtime it runs on in the
/// cache, what is left to download, the download that makes it so, and the delete. The options are the ones the load will
/// use (provider and cache directory decide which files that is).
/// </summary>
public sealed class DetectorArtifacts : IModelArtifacts
{
    private readonly DetectorOptions _options;

    /// <param name="options">The options later loads use; copied, so later changes to the instance do not apply.</param>
    public DetectorArtifacts(DetectorOptions? options = null)
    {
        _options = options?.Clone() ?? new DetectorOptions();
    }

    /// <inheritdoc />
    public string Kind => "detector";

    /// <inheritdoc />
    public async Task<bool> IsAvailableOfflineAsync(string modelId, CancellationToken cancellationToken = default)
        => await LocalDetector.IsModelDownloadedAsync(modelId, _options, cancellationToken)
           && await OnnxRuntimeArtifacts.IsAvailableOfflineAsync(_options.Provider, cancellationToken);

    /// <inheritdoc />
    public async Task<RemainingDownload> GetRemainingDownloadAsync(string modelId, CancellationToken cancellationToken = default)
        => new(await LocalDetector.GetRemainingDownloadBytesAsync(modelId, _options, cancellationToken),
            await OnnxRuntimeArtifacts.GetRemainingDownloadBytesAsync(_options.Provider, cancellationToken));

    /// <inheritdoc />
    public async Task DownloadAsync(string modelId, IProgress<DownloadProgress>? progress = null, CancellationToken cancellationToken = default)
    {
        await LocalDetector.DownloadModelAsync(modelId, _options, progress, cancellationToken);
        await OnnxRuntimeArtifacts.DownloadAsync(_options.Provider, progress, cancellationToken);
    }

    /// <inheritdoc />
    public bool Delete(string modelId) => LocalDetector.DeleteModel(modelId, _options);
}
