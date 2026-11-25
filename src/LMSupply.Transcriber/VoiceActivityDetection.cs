namespace LMSupply.Transcriber;

/// <summary>A span of a recording that holds speech.</summary>
/// <param name="Start">Start in seconds from the start of the recording.</param>
/// <param name="End">End in seconds from the start of the recording.</param>
public readonly record struct SpeechSegment(double Start, double End)
{
    /// <summary>The span's length in seconds.</summary>
    public double Duration => End - Start;
}

/// <summary>The result of <see cref="IVoiceActivityDetector.DetectAsync(string, VadOptions?, CancellationToken)"/>.</summary>
public sealed class VadResult
{
    /// <summary>The speech spans, ordered by start and not overlapping. Empty when the recording holds no speech.</summary>
    public required IReadOnlyList<SpeechSegment> Segments { get; init; }

    /// <summary>The sum of the spans' lengths in seconds.</summary>
    public double SpeechSeconds => Segments.Sum(s => s.Duration);

    /// <summary>The recording's length in seconds.</summary>
    public required double DurationSeconds { get; init; }

    /// <summary>The time the detection took in milliseconds, audio decoding included.</summary>
    public required double InferenceTimeMs { get; init; }
}

/// <summary>
/// How speech is told from what is not, and how the spans are cut. The settings and their defaults are those of the Silero
/// reference utility the model comes with, so a span here is a span there.
/// </summary>
public sealed class VadOptions
{
    /// <summary>
    /// The speech probability from which a 32 ms window counts as speech, in (0, 1). Higher values keep only what the model is
    /// sure of; lower values keep more, noise included. Speech ends only once the probability falls well below this
    /// (by 0.15), so a probability that wavers around it does not cut a span in pieces.
    /// <para>Default: null (0.5)</para>
    /// </summary>
    public float? Threshold { get; set; }

    /// <summary>
    /// The shortest span kept; anything not longer is dropped as a click or a cough.
    /// <para>Default: null (250 ms)</para>
    /// </summary>
    public TimeSpan? MinSpeechDuration { get; set; }

    /// <summary>
    /// How long a pause must last to end a span; shorter pauses stay inside it. Raise it to get whole sentences or turns
    /// instead of phrases.
    /// <para>Default: null (100 ms)</para>
    /// </summary>
    public TimeSpan? MinSilenceDuration { get; set; }

    /// <summary>
    /// How much of the surrounding silence each span takes on both sides, so a word's first and last sounds are not clipped.
    /// Two spans whose paddings meet share the gap between them.
    /// <para>Default: null (30 ms)</para>
    /// </summary>
    public TimeSpan? SpeechPadding { get; set; }

    /// <summary>
    /// The longest span: a longer run of speech is cut at its latest pause of about 100 ms or more, or at the limit when it
    /// holds none — for a recognizer that takes a bounded window (Whisper takes 30 s).
    /// <para>Default: null (no bound)</para>
    /// </summary>
    public TimeSpan? MaxSpeechDuration { get; set; }
}

/// <summary>Configuration for <see cref="LocalVoiceActivityDetector.LoadAsync"/>.</summary>
public sealed class VoiceActivityDetectorOptions : LMSupplyOptionsBase
{
    /// <summary>
    /// Whether to refuse a download: when true, loading throws if the model is not already cached.
    /// This covers the model file only; see <see cref="TranscriberOptions.DisableAutoDownload"/> for the native runtime.
    /// <para>Default: false</para>
    /// </summary>
    public bool DisableAutoDownload { get; set; }

    /// <summary>Creates a copy of these options.</summary>
    public VoiceActivityDetectorOptions Clone() => new()
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
/// Finds where a recording holds speech — without transcribing it. The audio goes in whole (a file, a stream or bytes in any
/// container the library decodes: WAV, MP3, Opus …) and the speech spans come out, on the recording's own time axis.
/// </summary>
/// <remarks>
/// One instance serves concurrent calls. The recording is held in memory while it is processed. The spans are what a
/// recognizer is fed, or what is kept of a recording before it is stored or sent on.
/// </remarks>
public interface IVoiceActivityDetector : IAsyncDisposable
{
    /// <summary>Finds the speech in a file.</summary>
    /// <exception cref="InvalidDataException">The audio is empty, holds no samples, or is no format the library reads.</exception>
    /// <exception cref="ArgumentOutOfRangeException">A setting is out of range (see <see cref="VadOptions"/>).</exception>
    Task<VadResult> DetectAsync(
        string audioPath, VadOptions? options = null, CancellationToken cancellationToken = default);

    /// <summary>Finds the speech in a stream, read to its end.</summary>
    /// <inheritdoc cref="DetectAsync(string, VadOptions?, CancellationToken)" path="/exception"/>
    Task<VadResult> DetectAsync(
        Stream audioStream, VadOptions? options = null, CancellationToken cancellationToken = default);

    /// <summary>Finds the speech in a byte array.</summary>
    /// <inheritdoc cref="DetectAsync(string, VadOptions?, CancellationToken)" path="/exception"/>
    Task<VadResult> DetectAsync(
        byte[] audioData, VadOptions? options = null, CancellationToken cancellationToken = default);
}
