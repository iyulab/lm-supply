using LMSupply.Transcriber.Audio;

namespace LMSupply.Transcriber.Core;

/// <summary>
/// Where the next 30 s window of a long input starts, and which of the current window's segments are
/// final. This is the seek rule of Whisper's long-form transcription: a window is not a fixed stride.
/// </summary>
/// <remarks>
/// <para>
/// A window ends mid-utterance almost every time. The decoder either leaves that utterance as an open
/// segment with no closing timestamp, or ends the window at end-of-text right after the last segment
/// it could close — and the speech after that point is then never decoded at all. A fixed 30 s stride
/// loses exactly that audio, at every boundary: measured on a 131 s Korean recording, 3–4 s per window
/// vanished and each following window opened mid-sentence (docket iyulab/lm-supply#340).
/// </para>
/// <para>
/// So the next window starts where the last <em>closed</em> segment ended, and anything after it —
/// an open segment, or audio the decoder skipped — is decoded again with its full context. The last
/// window keeps everything. The reference decoder also consumes the whole window when the token
/// sequence ends on a single timestamp ("nothing after this"); that is the case that dropped speech
/// here, so this rule seeks back regardless, at the cost of re-encoding a few seconds per window.
/// </para>
/// </remarks>
internal static class LongFormSeek
{
    /// <summary>
    /// Shortest seek-back advance, in samples (1 s). A window whose last closed segment ends earlier
    /// than this is consumed whole instead, so every window moves the cursor by at least this much
    /// and a decoder that keeps closing a segment near 0 s cannot stall the loop.
    /// </summary>
    internal const int MinAdvanceSamples = AudioProcessor.WhisperSampleRate;

    /// <summary>The outcome for one window.</summary>
    /// <param name="KeepSegments">How many of the window's segments, in order, are final.</param>
    /// <param name="NextStart">Sample index the next window starts at; the input length when done.</param>
    internal readonly record struct Step(int KeepSegments, int NextStart);

    /// <param name="windowStart">Sample index this window started at.</param>
    /// <param name="windowLength">Real (unpadded) samples in this window.</param>
    /// <param name="totalSamples">Length of the whole input.</param>
    /// <param name="segmentCount">Segments the window produced, closed and open.</param>
    /// <param name="closedSegmentCount">Leading segments closed by a timestamp token.</param>
    /// <param name="lastClosedEnd">Window-relative end, in seconds, of the last closed segment.</param>
    internal static Step Plan(
        int windowStart,
        int windowLength,
        int totalSamples,
        int segmentCount,
        int closedSegmentCount,
        double? lastClosedEnd)
    {
        var windowEnd = windowStart + windowLength;
        if (windowEnd >= totalSamples)
        {
            return new Step(segmentCount, totalSamples);
        }

        var advance = closedSegmentCount > 0 && lastClosedEnd is { } end
            ? (int)Math.Min(Math.Round(end * AudioProcessor.WhisperSampleRate), windowLength)
            : 0;

        if (advance >= MinAdvanceSamples)
        {
            return new Step(closedSegmentCount, windowStart + advance);
        }

        // Nothing closed far enough in to seek back to: take the window as it is. A remainder too
        // short to be a window of its own is dropped rather than padded (AudioProcessor.MinTailSamples).
        var next = totalSamples - windowEnd < AudioProcessor.MinTailSamples ? totalSamples : windowEnd;
        return new Step(segmentCount, next);
    }
}
