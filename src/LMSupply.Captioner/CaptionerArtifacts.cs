using LMSupply.Lifecycle;

namespace LMSupply.Captioner;

/// <summary>
/// <see cref="IModelArtifacts"/> for captioners: whether a <see cref="LocalCaptioner.LoadAsync"/> with downloads disabled
/// would find every file it opens in the cache, and the download that makes it so. The options are the ones the load will
/// use (provider, quantization hint and cache directory decide which files that is).
/// </summary>
public sealed class CaptionerArtifacts : IModelArtifacts
{
    private readonly CaptionerOptions _options;

    /// <param name="options">The options later loads use; copied, so later changes to the instance do not apply.</param>
    public CaptionerArtifacts(CaptionerOptions? options = null)
    {
        _options = options?.Clone() ?? new CaptionerOptions();
    }

    /// <inheritdoc />
    public string Kind => "captioner";

    /// <inheritdoc />
    public async Task<bool> IsAvailableOfflineAsync(string modelId, CancellationToken cancellationToken = default)
        => await LocalCaptioner.IsModelDownloadedAsync(modelId, _options, cancellationToken)
           && await OnnxRuntimeArtifacts.IsAvailableOfflineAsync(_options.Provider, cancellationToken);

    /// <inheritdoc />
    public async Task<RemainingDownload> GetRemainingDownloadAsync(string modelId, CancellationToken cancellationToken = default)
        => new(await LocalCaptioner.GetRemainingDownloadBytesAsync(modelId, _options, cancellationToken),
            await OnnxRuntimeArtifacts.GetRemainingDownloadBytesAsync(_options.Provider, cancellationToken));

    /// <inheritdoc />
    public async Task DownloadAsync(string modelId, IProgress<DownloadProgress>? progress = null, CancellationToken cancellationToken = default)
    {
        await LocalCaptioner.DownloadModelAsync(modelId, _options, progress, cancellationToken);
        await OnnxRuntimeArtifacts.DownloadAsync(_options.Provider, progress, cancellationToken);
    }

    /// <inheritdoc />
    public bool Delete(string modelId) => LocalCaptioner.DeleteModel(modelId, _options);
}
