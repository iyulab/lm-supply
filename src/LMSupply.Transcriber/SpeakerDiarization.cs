namespace LMSupply.Transcriber;

/// <summary>
/// A span of a recording in which one speaker talks. At most two speakers are detected at the same instant, so turns of
/// different speakers may overlap.
/// </summary>
/// <param name="Start">Start in seconds from the start of the recording.</param>
/// <param name="End">End in seconds from the start of the recording.</param>
/// <param name="Speaker">
/// The speaker's label — <c>"S1"</c>, <c>"S2"</c>, … numbered in the order the voices first speak, stable within one
/// recording, and the same label <see cref="TranscriptionSegment.Speaker"/> carries when the same recording is
/// transcribed with <see cref="TranscribeOptions.Diarize"/>. A label names a voice, not a person.
/// </param>
public readonly record struct SpeakerTurn(double Start, double End, string Speaker)
{
    /// <summary>The span's length in seconds.</summary>
    public double Duration => End - Start;

    /// <summary>The label of the speaker numbered <paramref name="index"/> (0-based) in speaking order.</summary>
    internal static string LabelOf(int index) => $"S{index + 1}";
}

/// <summary>The result of <see cref="ISpeakerDiarizer.DiarizeAsync(string, DiarizeOptions?, CancellationToken)"/>.</summary>
public sealed class DiarizationResult
{
    /// <summary>The turns, ordered by start. Empty when the recording holds no speech.</summary>
    public required IReadOnlyList<SpeakerTurn> Turns { get; init; }

    /// <summary>How many distinct speakers the turns carry.</summary>
    public int SpeakerCount => Turns.Select(t => t.Speaker).Distinct().Count();

    /// <summary>The recording's length in seconds.</summary>
    public required double DurationSeconds { get; init; }

    /// <summary>The time the diarization took in milliseconds, audio decoding included.</summary>
    public required double InferenceTimeMs { get; init; }
}

/// <summary>
/// How many speakers a diarization call looks for. With none of it set the count is estimated from the audio.
/// </summary>
/// <remarks>The same four settings as <see cref="TranscribeOptions"/>' speaker settings, with the same meaning.</remarks>
public sealed class DiarizeOptions
{
    /// <summary>
    /// The exact number of speakers, when known. Takes precedence over <see cref="SpeakerThreshold"/>,
    /// <see cref="MinSpeakers"/> and <see cref="MaxSpeakers"/>: a recording with fewer distinct voices is split to reach
    /// it, and when the audio cannot yield that many you get the nearest count it can.
    /// <para>Default: null (estimated from the audio)</para>
    /// </summary>
    public int? NumSpeakers { get; set; }

    /// <summary>
    /// The fewest speakers to report; ignored when <see cref="NumSpeakers"/> is set.
    /// <para>Default: null (no lower bound)</para>
    /// </summary>
    public int? MinSpeakers { get; set; }

    /// <summary>
    /// The most speakers to report — an upper bound, so a recording with fewer voices keeps its own count; ignored when
    /// <see cref="NumSpeakers"/> is set.
    /// <para>Default: null (no upper bound)</para>
    /// </summary>
    public int? MaxSpeakers { get; set; }

    /// <summary>
    /// The cosine-distance cut between speakers for the estimate (when <see cref="NumSpeakers"/> is not set): two voices
    /// closer than this are one speaker. Smaller values find more speakers, larger values fewer.
    /// <para>Default: null (0.4, tuned for the bundled embedding model)</para>
    /// </summary>
    public float? SpeakerThreshold { get; set; }
}

/// <summary>Configuration for <see cref="LocalSpeakerDiarizer.LoadAsync"/>.</summary>
public sealed class SpeakerDiarizerOptions : LMSupplyOptionsBase
{
    /// <summary>
    /// Whether to refuse a download: when true, loading throws if the models are not already cached.
    /// This covers model files only; see <see cref="TranscriberOptions.DisableAutoDownload"/> for the native runtime.
    /// <para>Default: false</para>
    /// </summary>
    public bool DisableAutoDownload { get; set; }

    /// <summary>Creates a copy of these options.</summary>
    public SpeakerDiarizerOptions Clone() => new()
    {
        CacheDirectory = CacheDirectory,
        Provider = Provider,
        LogLevel = LogLevel,
        DisableAutoDownload = DisableAutoDownload,
        ThreadCount = ThreadCount,
        QuantizationHint = QuantizationHint
    };
}

/// <summary>
/// Finds who spoke when in a recording — without transcribing it. The audio goes in whole (a file, a stream or bytes in
/// any container the library decodes: WAV, MP3, Opus …) and the speaker turns come out, with the same models and
/// settings <see cref="TranscribeOptions.Diarize"/> uses.
/// </summary>
/// <remarks>
/// One instance serves concurrent calls. The recording is held in memory while it is processed. To label transcription
/// segments instead, set <see cref="TranscribeOptions.Diarize"/> on a transcriber call — it runs this same diarization
/// on the samples it already decoded.
/// </remarks>
public interface ISpeakerDiarizer : IAsyncDisposable
{
    /// <summary>Diarizes the recording in a file.</summary>
    /// <exception cref="InvalidDataException">The audio is empty, holds no samples, or is no format the library reads.</exception>
    /// <exception cref="ArgumentOutOfRangeException">A speaker setting is out of range (non-positive count, minimum above maximum, cut outside (0, 2)).</exception>
    Task<DiarizationResult> DiarizeAsync(
        string audioPath, DiarizeOptions? options = null, CancellationToken cancellationToken = default);

    /// <summary>Diarizes the recording in a stream, read to its end.</summary>
    /// <inheritdoc cref="DiarizeAsync(string, DiarizeOptions?, CancellationToken)" path="/exception"/>
    Task<DiarizationResult> DiarizeAsync(
        Stream audioStream, DiarizeOptions? options = null, CancellationToken cancellationToken = default);

    /// <summary>Diarizes the recording in a byte array.</summary>
    /// <inheritdoc cref="DiarizeAsync(string, DiarizeOptions?, CancellationToken)" path="/exception"/>
    Task<DiarizationResult> DiarizeAsync(
        byte[] audioData, DiarizeOptions? options = null, CancellationToken cancellationToken = default);
}
