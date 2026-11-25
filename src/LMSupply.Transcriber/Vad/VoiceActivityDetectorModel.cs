using System.Diagnostics;
using LMSupply.Transcriber.Audio;

namespace LMSupply.Transcriber.Vad;

/// <summary>The public face of <see cref="SileroVad"/>: decodes the recording, finds the speech, converts to seconds.</summary>
internal sealed class VoiceActivityDetectorModel : IVoiceActivityDetector
{
    internal const float DefaultThreshold = 0.5f;
    internal static readonly TimeSpan DefaultMinSpeech = TimeSpan.FromMilliseconds(250);
    internal static readonly TimeSpan DefaultMinSilence = TimeSpan.FromMilliseconds(100);
    internal static readonly TimeSpan DefaultPadding = TimeSpan.FromMilliseconds(30);

    private readonly SileroVad _vad;

    public VoiceActivityDetectorModel(SileroVad vad) => _vad = vad;

    public async Task<VadResult> DetectAsync(
        string audioPath, VadOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(audioPath);
        var settings = SettingsOf(options);
        return await DetectSamplesAsync(await AudioProcessor.LoadAudioAsync(audioPath, cancellationToken), settings, cancellationToken);
    }

    public async Task<VadResult> DetectAsync(
        Stream audioStream, VadOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(audioStream);
        var settings = SettingsOf(options);
        return await DetectSamplesAsync(await AudioProcessor.LoadAudioAsync(audioStream, cancellationToken), settings, cancellationToken);
    }

    public async Task<VadResult> DetectAsync(
        byte[] audioData, VadOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(audioData);
        var settings = SettingsOf(options);
        return await DetectSamplesAsync(await AudioProcessor.LoadAudioAsync(audioData, cancellationToken), settings, cancellationToken);
    }

    /// <summary>Checks the options and turns them into sample counts; throws before any audio is read.</summary>
    internal static SegmenterSettings SettingsOf(VadOptions? options)
    {
        var threshold = options?.Threshold ?? DefaultThreshold;
        if (!(threshold > 0f && threshold < 1f))
            throw new ArgumentOutOfRangeException(nameof(options), threshold, $"{nameof(VadOptions)}.{nameof(VadOptions.Threshold)} must be between 0 and 1, exclusive.");

        var minSpeech = NotNegative(options?.MinSpeechDuration ?? DefaultMinSpeech, nameof(VadOptions.MinSpeechDuration));
        var minSilence = NotNegative(options?.MinSilenceDuration ?? DefaultMinSilence, nameof(VadOptions.MinSilenceDuration));
        var padding = NotNegative(options?.SpeechPadding ?? DefaultPadding, nameof(VadOptions.SpeechPadding));

        var maxSamples = int.MaxValue;
        if (options?.MaxSpeechDuration is { } max)
        {
            if (max <= TimeSpan.Zero || max <= minSpeech)
                throw new ArgumentOutOfRangeException(nameof(options), max, $"{nameof(VadOptions)}.{nameof(VadOptions.MaxSpeechDuration)} must be longer than {nameof(VadOptions.MinSpeechDuration)}.");
            maxSamples = Samples(max);
        }

        return new SegmenterSettings(threshold, Samples(minSpeech), Samples(minSilence), Samples(padding), maxSamples);

        static TimeSpan NotNegative(TimeSpan value, string name)
            => value >= TimeSpan.Zero
                ? value
                : throw new ArgumentOutOfRangeException(nameof(options), value, $"{nameof(VadOptions)}.{name} must not be negative.");

        static int Samples(TimeSpan value) => (int)Math.Min(int.MaxValue, Math.Round(value.TotalSeconds * SpeechSegmenter.SampleRate));
    }

    private Task<VadResult> DetectSamplesAsync(float[] samples, SegmenterSettings settings, CancellationToken cancellationToken)
        => Task.Run(() =>
        {
            var sw = Stopwatch.StartNew();
            var probabilities = _vad.Probabilities(samples, cancellationToken);
            var spans = SpeechSegmenter.Segment(probabilities, samples.Length, settings);
            sw.Stop();

            return new VadResult
            {
                Segments = spans
                    .Select(s => new SpeechSegment((double)s.Start / SpeechSegmenter.SampleRate, (double)s.End / SpeechSegmenter.SampleRate))
                    .ToList(),
                DurationSeconds = AudioProcessor.GetDurationSeconds(samples),
                InferenceTimeMs = sw.Elapsed.TotalMilliseconds,
            };
        }, cancellationToken);

    public ValueTask DisposeAsync() => _vad.DisposeAsync();
}
