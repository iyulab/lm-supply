using LMSupply.Lifecycle;

namespace LMSupply.Transcriber;

/// <summary>
/// <see cref="IModelArtifacts"/> for transcribers: whether a <see cref="LocalTranscriber"/> load with downloads disabled would
/// find the model's files in the cache, and the download that makes it so. The options are the ones the load will use
/// (provider, quantization hint and cache directory decide which files that is).
/// </summary>
/// <remarks>
/// The speaker-diarization pair is not part of this kind, whatever <see cref="TranscriberOptions.PreloadDiarization"/> says: it
/// is <see cref="DiarizationArtifacts"/>, a kind of its own, so «transcription available» does not wait for speaker labels. A
/// host whose loads preload diarization registers both and treats the transcriber as ready when both are.
/// </remarks>
public sealed class TranscriberArtifacts : IModelArtifacts
{
    private readonly TranscriberOptions _options;

    /// <param name="options">The options later loads use; copied, so later changes to the instance do not apply.</param>
    public TranscriberArtifacts(TranscriberOptions? options = null)
    {
        _options = options?.Clone() ?? new TranscriberOptions();
        // The pair is DiarizationArtifacts' kind; the load option stays for the load.
        _options.PreloadDiarization = false;
    }

    /// <inheritdoc />
    public string Kind => "transcriber";

    /// <inheritdoc />
    public async Task<bool> IsAvailableOfflineAsync(string modelId, CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(modelId);
        return await LocalTranscriber.IsModelDownloadedAsync(modelId, _options, cancellationToken)
               && await OnnxRuntimeArtifacts.IsAvailableOfflineAsync(_options.Provider, cancellationToken);
    }

    /// <inheritdoc />
    public async Task<RemainingDownload> GetRemainingDownloadAsync(string modelId, CancellationToken cancellationToken = default)
        => new(await LocalTranscriber.GetRemainingDownloadBytesAsync(modelId, _options, cancellationToken),
            await OnnxRuntimeArtifacts.GetRemainingDownloadBytesAsync(_options.Provider, cancellationToken));

    /// <inheritdoc />
    public async Task DownloadAsync(string modelId, IProgress<DownloadProgress>? progress = null, CancellationToken cancellationToken = default)
    {
        await LocalTranscriber.DownloadModelAsync(modelId, _options, progress, cancellationToken);
        await OnnxRuntimeArtifacts.DownloadAsync(_options.Provider, progress, cancellationToken);
    }

    /// <inheritdoc />
    public bool Delete(string modelId) => LocalTranscriber.DeleteModel(modelId, _options);
}
