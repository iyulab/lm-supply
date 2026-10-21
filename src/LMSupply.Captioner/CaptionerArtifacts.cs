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
    public Task<bool> IsAvailableOfflineAsync(string modelId, CancellationToken cancellationToken = default)
        => LocalCaptioner.IsModelDownloadedAsync(modelId, _options, cancellationToken);

    /// <inheritdoc />
    public async Task<RemainingDownload> GetRemainingDownloadAsync(string modelId, CancellationToken cancellationToken = default)
        => new(await LocalCaptioner.GetRemainingDownloadBytesAsync(modelId, _options, cancellationToken), RuntimeBytes: null);

    /// <inheritdoc />
    public Task DownloadAsync(string modelId, IProgress<DownloadProgress>? progress = null, CancellationToken cancellationToken = default)
        => LocalCaptioner.DownloadModelAsync(modelId, _options, progress, cancellationToken);

    /// <inheritdoc />
    public bool Delete(string modelId) => LocalCaptioner.DeleteModel(modelId, _options);
}
