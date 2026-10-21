using LMSupply.Lifecycle;

namespace LMSupply.Transcriber;

/// <summary>
/// <see cref="IModelArtifacts"/> for the speaker-diarization pair that <see cref="TranscribeOptions.Diarize"/> and
/// <see cref="TranscriberOptions.PreloadDiarization"/> use (pyannote segmentation-3.0 + WeSpeaker ResNet34): whether it is in
/// the cache, what is left to download, the download and the delete. It is a kind of its own, beside
/// <see cref="TranscriberArtifacts"/>, so a host can offer speaker labels as an add-on with its own consent: «transcription
/// available» and «speaker labels available» are two statuses in one <see cref="ModelLifecycle"/>.
/// </summary>
/// <remarks>
/// Every transcriber shares one pair, so there is one model id, <see cref="DefaultModelId"/>. The options are the ones later
/// loads use: <see cref="LMSupplyOptionsBase.CacheDirectory"/> says where the pair is, and with
/// <see cref="TranscriberOptions.DisableAutoDownload"/> <see cref="DownloadAsync"/> only confirms the pair is cached.
/// </remarks>
public sealed class DiarizationArtifacts : IModelArtifacts
{
    /// <summary>The one model id of this kind: the speaker-diarization pair.</summary>
    public const string DefaultModelId = "default";

    private readonly TranscriberOptions _options;

    /// <param name="options">The options later loads use; copied, so later changes to the instance do not apply.</param>
    public DiarizationArtifacts(TranscriberOptions? options = null)
    {
        _options = options?.Clone() ?? new TranscriberOptions();
    }

    /// <inheritdoc />
    public string Kind => "diarization";

    private string CacheDirectory => _options.CacheDirectory ?? CacheManager.GetDefaultCacheDirectory();

    /// <inheritdoc />
    public Task<bool> IsAvailableOfflineAsync(string modelId, CancellationToken cancellationToken = default)
    {
        Validate(modelId);
        return LocalTranscriber.IsDiarizationDownloadedAsync(_options, cancellationToken);
    }

    /// <inheritdoc />
    /// <remarks>The pair's size when either file is missing or partial (a partial file counts in full), else 0.</remarks>
    public async Task<RemainingDownload> GetRemainingDownloadAsync(string modelId, CancellationToken cancellationToken = default)
    {
        Validate(modelId);
        var model = await LocalTranscriber.IsDiarizationDownloadedAsync(_options, cancellationToken).ConfigureAwait(false)
            ? 0
            : LocalTranscriber.DiarizationDownloadSizeBytes;
        return new RemainingDownload(model, RuntimeBytes: null);
    }

    /// <inheritdoc />
    public Task DownloadAsync(string modelId, IProgress<DownloadProgress>? progress = null, CancellationToken cancellationToken = default)
    {
        Validate(modelId);
        return Diarization.SpeakerDiarizer.DownloadAsync(CacheDirectory, _options.DisableAutoDownload, progress, cancellationToken);
    }

    /// <inheritdoc />
    /// <remarks>Deletes what LMSupply downloaded of the pair's two repositories; a transcriber model is not touched.</remarks>
    public bool Delete(string modelId)
    {
        Validate(modelId);
        var deleted = false;
        foreach (var repoId in Diarization.SpeakerDiarizer.Repositories)
            deleted |= CacheManager.DeleteModel(CacheDirectory, repoId);
        return deleted;
    }

    private static void Validate(string modelId)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(modelId);
        if (!string.Equals(modelId, DefaultModelId, StringComparison.OrdinalIgnoreCase))
            throw new ArgumentException(
                $"The speaker-diarization kind has one model, '{DefaultModelId}'; '{modelId}' is not one.", nameof(modelId));
    }
}
