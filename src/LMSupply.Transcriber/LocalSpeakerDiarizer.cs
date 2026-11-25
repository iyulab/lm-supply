using LMSupply.Download;
using LMSupply.Transcriber.Diarization;

namespace LMSupply.Transcriber;

/// <summary>
/// Entry point for speaker diarization without transcription: <c>who spoke when</c> in a recording. The models are the
/// pair <see cref="TranscribeOptions.Diarize"/> uses (pyannote segmentation-3.0 and a WeSpeaker ResNet34 embedding,
/// about 32.5 MB), fetched on first use into the cache a transcriber shares.
/// </summary>
/// <example>
/// <code>
/// await using var diarizer = await LocalSpeakerDiarizer.LoadAsync();
/// var result = await diarizer.DiarizeAsync("meeting.wav", new DiarizeOptions { MaxSpeakers = 4 });
/// foreach (var turn in result.Turns)
///     Console.WriteLine($"{turn.Speaker}: {turn.Start:F1}-{turn.End:F1} s");
/// </code>
/// </example>
public static class LocalSpeakerDiarizer
{
    /// <summary>Loads the diarization models, downloading them first unless <see cref="SpeakerDiarizerOptions.DisableAutoDownload"/> is set.</summary>
    /// <exception cref="Exceptions.ModelNotFoundException">Downloads are disabled and the models are not cached.</exception>
    public static async Task<ISpeakerDiarizer> LoadAsync(
        SpeakerDiarizerOptions? options = null,
        IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        options = options?.Clone() ?? new SpeakerDiarizerOptions();
        var diarizer = await SpeakerDiarizer.LoadAsync(
            CacheDirectoryOf(options), options.DisableAutoDownload, options.Provider,
            LoadProgress.Begin(progress), cancellationToken);
        return new SpeakerDiarizerModel(diarizer);
    }

    /// <summary>
    /// Whether the models are in the cache, complete — whether <see cref="LoadAsync"/> would open them without a download.
    /// Makes no network request and loads nothing.
    /// </summary>
    /// <param name="options">Only <see cref="LMSupplyOptionsBase.CacheDirectory"/> is read; null means the default cache.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    public static Task<bool> IsDownloadedAsync(
        SpeakerDiarizerOptions? options = null, CancellationToken cancellationToken = default)
        => SpeakerDiarizer.IsCachedAsync(CacheDirectoryOf(options), cancellationToken);

    /// <summary>
    /// Fetches the models into the cache without loading them (a no-op for what is cached) — for an install or consent
    /// step. Obeys <see cref="SpeakerDiarizerOptions.DisableAutoDownload"/>: then it only confirms the models are cached.
    /// </summary>
    public static Task DownloadAsync(
        SpeakerDiarizerOptions? options = null,
        IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
        => SpeakerDiarizer.DownloadAsync(
            CacheDirectoryOf(options), options?.DisableAutoDownload ?? false, LoadProgress.Begin(progress), cancellationToken);

    /// <summary>Bytes the models take when downloaded (about 32.5 MB) — for a consent screen.</summary>
    public static long DownloadSizeBytes => SpeakerDiarizer.DownloadSizeBytes;

    private static string CacheDirectoryOf(SpeakerDiarizerOptions? options)
        => options?.CacheDirectory ?? CacheManager.GetDefaultCacheDirectory();
}
