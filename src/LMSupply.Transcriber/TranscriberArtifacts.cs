using LMSupply.Lifecycle;

namespace LMSupply.Transcriber;

/// <summary>
/// <see cref="IModelArtifacts"/> for transcribers: whether a <see cref="LocalTranscriber"/> load with downloads disabled would
/// find every file it opens in the cache — the model's files, and the speaker-diarization pair when the options preload it —
/// and the download that makes it so. The options are the ones the load will use (provider, quantization hint and cache
/// directory decide which files that is).
/// </summary>
public sealed class TranscriberArtifacts : IModelArtifacts
{
    private readonly TranscriberOptions _options;

    /// <param name="options">The options later loads use; copied, so later changes to the instance do not apply.</param>
    public TranscriberArtifacts(TranscriberOptions? options = null)
    {
        _options = options?.Clone() ?? new TranscriberOptions();
    }

    /// <inheritdoc />
    public string Kind => "transcriber";

    /// <inheritdoc />
    public async Task<bool> IsAvailableOfflineAsync(string modelId, CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(modelId);
        if (!await LocalTranscriber.IsModelDownloadedAsync(modelId, _options, cancellationToken))
            return false;
        return !_options.PreloadDiarization || await LocalTranscriber.IsDiarizationDownloadedAsync(_options, cancellationToken);
    }

    /// <inheritdoc />
    public Task<long> GetRemainingDownloadBytesAsync(string modelId, CancellationToken cancellationToken = default)
        => LocalTranscriber.GetRemainingDownloadBytesAsync(modelId, _options, cancellationToken);

    /// <inheritdoc />
    public Task DownloadAsync(string modelId, IProgress<DownloadProgress>? progress = null, CancellationToken cancellationToken = default)
        => LocalTranscriber.DownloadModelAsync(modelId, _options, progress, cancellationToken);

    /// <inheritdoc />
    public bool Delete(string modelId) => LocalTranscriber.DeleteModel(modelId, _options);
}
