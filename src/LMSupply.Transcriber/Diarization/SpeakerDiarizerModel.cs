using System.Diagnostics;
using LMSupply.Transcriber.Audio;

namespace LMSupply.Transcriber.Diarization;

/// <summary>The public face of <see cref="SpeakerDiarizer"/>: decodes the recording, diarizes it, labels the turns.</summary>
internal sealed class SpeakerDiarizerModel : ISpeakerDiarizer
{
    private readonly SpeakerDiarizer _diarizer;

    public SpeakerDiarizerModel(SpeakerDiarizer diarizer) => _diarizer = diarizer;

    public async Task<DiarizationResult> DiarizeAsync(
        string audioPath, DiarizeOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(audioPath);
        Validate(options);
        return await DiarizeSamplesAsync(await AudioProcessor.LoadAudioAsync(audioPath, cancellationToken), options, cancellationToken);
    }

    public async Task<DiarizationResult> DiarizeAsync(
        Stream audioStream, DiarizeOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(audioStream);
        Validate(options);
        return await DiarizeSamplesAsync(await AudioProcessor.LoadAudioAsync(audioStream, cancellationToken), options, cancellationToken);
    }

    public async Task<DiarizationResult> DiarizeAsync(
        byte[] audioData, DiarizeOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(audioData);
        Validate(options);
        return await DiarizeSamplesAsync(await AudioProcessor.LoadAudioAsync(audioData, cancellationToken), options, cancellationToken);
    }

    private static void Validate(DiarizeOptions? options)
    {
        if (options is null)
            return;
        DiarizationStage.ValidateBounds(nameof(DiarizeOptions), options.NumSpeakers, options.MinSpeakers,
            options.MaxSpeakers, options.SpeakerThreshold, nameof(options));
    }

    private Task<DiarizationResult> DiarizeSamplesAsync(float[] samples, DiarizeOptions? options, CancellationToken cancellationToken)
        => Task.Run(() =>
        {
            var sw = Stopwatch.StartNew();
            var turns = _diarizer.Diarize(
                samples,
                options?.NumSpeakers ?? 0,
                options?.SpeakerThreshold ?? SpeakerDiarizer.DefaultThreshold,
                options?.MinSpeakers ?? 0,
                options?.MaxSpeakers ?? 0,
                cancellationToken);
            sw.Stop();

            return new DiarizationResult
            {
                Turns = turns.Select(t => new SpeakerTurn(t.Start, t.End, SpeakerTurn.LabelOf(t.Speaker))).ToList(),
                DurationSeconds = AudioProcessor.GetDurationSeconds(samples),
                InferenceTimeMs = sw.Elapsed.TotalMilliseconds,
            };
        }, cancellationToken);

    public ValueTask DisposeAsync() => _diarizer.DisposeAsync();
}
