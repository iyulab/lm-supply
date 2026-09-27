namespace LMSupply.Transcriber.Diarization;

/// <summary>
/// A transcriber backend that can fetch and load its diarization pair during <see cref="LocalTranscriber.LoadAsync(string, TranscriberOptions?, IProgress{DownloadProgress}?, CancellationToken)"/>
/// instead of on the first <see cref="TranscribeOptions.Diarize"/> call.
/// </summary>
internal interface IDiarizationPreload
{
    Task PreloadDiarizationAsync(IProgress<DownloadProgress>? progress, CancellationToken cancellationToken);
}
