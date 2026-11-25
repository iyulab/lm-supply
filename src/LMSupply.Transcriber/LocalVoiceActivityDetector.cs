using LMSupply.Download;
using LMSupply.Transcriber.Vad;

namespace LMSupply.Transcriber;

/// <summary>
/// Entry point for voice activity detection: where a recording holds speech. The model is Silero VAD (about 2.2 MB, MIT),
/// fetched on first use into the shared cache and run on the CPU.
/// </summary>
/// <example>
/// <code>
/// await using var detector = await LocalVoiceActivityDetector.LoadAsync();
/// var result = await detector.DetectAsync("meeting.wav", new VadOptions { MinSilenceDuration = TimeSpan.FromMilliseconds(500) });
/// foreach (var span in result.Segments)
///     Console.WriteLine($"{span.Start:F1}-{span.End:F1} s");
/// </code>
/// </example>
public static class LocalVoiceActivityDetector
{
    /// <summary>Loads the model, downloading it first unless <see cref="VoiceActivityDetectorOptions.DisableAutoDownload"/> is set.</summary>
    /// <exception cref="Exceptions.ModelNotFoundException">Downloads are disabled and the model is not cached.</exception>
    public static async Task<IVoiceActivityDetector> LoadAsync(
        VoiceActivityDetectorOptions? options = null,
        IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        options = options?.Clone() ?? new VoiceActivityDetectorOptions();
        var vad = await SileroVad.LoadAsync(
            CacheDirectoryOf(options), options.DisableAutoDownload, options.Provider,
            LoadProgress.Begin(progress), cancellationToken);
        return new VoiceActivityDetectorModel(vad);
    }

    /// <summary>
    /// Whether the model is in the cache, complete — whether <see cref="LoadAsync"/> would open it without a download.
    /// Makes no network request and loads nothing.
    /// </summary>
    /// <param name="options">Only <see cref="LMSupplyOptionsBase.CacheDirectory"/> is read; null means the default cache.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    public static Task<bool> IsDownloadedAsync(
        VoiceActivityDetectorOptions? options = null, CancellationToken cancellationToken = default)
        => SileroVad.IsCachedAsync(CacheDirectoryOf(options), cancellationToken);

    /// <summary>
    /// Fetches the model into the cache without loading it (a no-op when cached) — for an install or consent step. Obeys
    /// <see cref="VoiceActivityDetectorOptions.DisableAutoDownload"/>: then it only confirms the model is cached.
    /// </summary>
    public static Task DownloadAsync(
        VoiceActivityDetectorOptions? options = null,
        IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
        => SileroVad.DownloadAsync(
            CacheDirectoryOf(options), options?.DisableAutoDownload ?? false, LoadProgress.Begin(progress), cancellationToken);

    /// <summary>Bytes the model takes when downloaded (about 2.2 MB) — for a consent screen.</summary>
    public static long DownloadSizeBytes => SileroVad.DownloadSizeBytes;

    private static string CacheDirectoryOf(VoiceActivityDetectorOptions? options)
        => options?.CacheDirectory ?? CacheManager.GetDefaultCacheDirectory();
}
