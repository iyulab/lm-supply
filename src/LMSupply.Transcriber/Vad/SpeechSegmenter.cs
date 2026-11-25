namespace LMSupply.Transcriber.Vad;

/// <summary>The rules that turn per-window speech probabilities into speech spans, in samples.</summary>
/// <param name="Threshold">A window at or above this starts (or continues) speech.</param>
/// <param name="MinSpeechSamples">A span of speech no longer than this is dropped.</param>
/// <param name="MinSilenceSamples">Speech ends after this much silence.</param>
/// <param name="PadSamples">Each span is widened by this much on both sides (neighbours share the gap between them).</param>
/// <param name="MaxSpeechSamples">A span longer than this is cut; <see cref="int.MaxValue"/> for no bound.</param>
internal readonly record struct SegmenterSettings(
    float Threshold, int MinSpeechSamples, int MinSilenceSamples, int PadSamples, int MaxSpeechSamples);

/// <summary>
/// Speech spans from the speech probabilities of consecutive windows — the hysteresis rules of Silero's reference
/// <c>get_speech_timestamps</c>: speech starts at a window at or above the threshold and ends once the probability has been
/// below the negative threshold (threshold − 0.15, at least 0.01) for the minimum silence; probabilities between the two
/// thresholds change nothing. Pure arithmetic, so the rules are tested without the model.
/// </summary>
internal static class SpeechSegmenter
{
    /// <summary>The model's sample rate; the only one the library feeds it.</summary>
    public const int SampleRate = 16000;

    /// <summary>Samples in one window (32 ms).</summary>
    public const int WindowSamples = 512;

    // A pause at least this long (98 ms) is where a span that outgrew the maximum is cut, when it holds one.
    private const int MinCutSilenceSamples = SampleRate * 98 / 1000;

    /// <param name="probabilities">The speech probability of each window; window <c>i</c> starts at <c>i × 512</c> samples.</param>
    /// <param name="totalSamples">The recording's length in samples; spans never reach beyond it.</param>
    /// <param name="settings">The thresholds and durations, in samples.</param>
    /// <returns>(start, end) in samples, ordered, not overlapping.</returns>
    public static IReadOnlyList<(int Start, int End)> Segment(
        IReadOnlyList<float> probabilities, int totalSamples, SegmenterSettings settings)
    {
        var negativeThreshold = Math.Max(settings.Threshold - 0.15f, 0.01f);
        var raw = new List<(int Start, int End)>();

        var triggered = false;
        var start = 0;
        var silenceStart = -1;           // where the current run of silence began; -1 outside one
        var cutStart = -1;               // the latest pause of cut length inside the current span: [cutStart, cutEnd)
        var cutEnd = -1;

        void Emit(int from, int to)
        {
            if (to - from > settings.MinSpeechSamples)
                raw.Add((from, to));
        }

        for (var i = 0; i < probabilities.Count; i++)
        {
            var position = i * WindowSamples;
            var probability = probabilities[i];

            if (probability >= settings.Threshold)
            {
                if (silenceStart >= 0)
                {
                    if (position - silenceStart >= MinCutSilenceSamples)
                    {
                        cutStart = silenceStart;
                        cutEnd = position;
                    }

                    silenceStart = -1;
                }

                if (!triggered)
                {
                    triggered = true;
                    start = position;
                    cutStart = -1;
                    continue;
                }
            }
            else if (triggered && probability < negativeThreshold)
            {
                if (silenceStart < 0)
                    silenceStart = position;

                if (position - silenceStart >= settings.MinSilenceSamples)
                {
                    Emit(start, silenceStart);
                    triggered = false;
                    silenceStart = -1;
                    cutStart = -1;
                    continue;
                }
            }

            if (triggered && position - start > settings.MaxSpeechSamples)
            {
                if (cutStart > start)
                {
                    Emit(start, cutStart);
                    start = cutEnd;
                }
                else
                {
                    Emit(start, position);
                    start = position;
                }

                cutStart = -1;
            }
        }

        if (triggered)
            Emit(start, totalSamples);

        return Pad(raw, totalSamples, settings.PadSamples);
    }

    private static List<(int Start, int End)> Pad(List<(int Start, int End)> spans, int totalSamples, int pad)
    {
        var padded = new List<(int Start, int End)>(spans.Count);
        var previousEnd = 0;
        for (var i = 0; i < spans.Count; i++)
        {
            var (start, end) = spans[i];
            start = Math.Max(previousEnd, start - pad);

            if (i + 1 < spans.Count)
            {
                var next = spans[i + 1].Start;
                var gap = next - end;
                if (gap < 2 * pad)
                {
                    // The two paddings would meet: the gap is shared, and the next span starts where this one ends.
                    end += gap / 2;
                    previousEnd = end;
                }
                else
                {
                    end = Math.Min(totalSamples, end + pad);
                    previousEnd = 0;
                }
            }
            else
            {
                end = Math.Min(totalSamples, end + pad);
            }

            padded.Add((Math.Max(0, start), end));
        }

        return padded;
    }
}
