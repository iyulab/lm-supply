using AwesomeAssertions;
using LMSupply.Transcriber.Vad;

namespace LMSupply.Transcriber.Tests.Vad;

/// <summary>
/// The rules that cut speech spans out of per-window probabilities, in samples (a window is 512 samples). The expected
/// spans are worked out by hand from the rules, so these tests need no model.
/// </summary>
public sealed class SpeechSegmenterTests
{
    private const int Window = SpeechSegmenter.WindowSamples;

    // 0.5 threshold (negative threshold 0.35), 250 ms minimum speech, 100 ms minimum silence.
    private static SegmenterSettings Settings(int pad = 0, int max = int.MaxValue)
        => new(0.5f, MinSpeechSamples: 4000, MinSilenceSamples: 1600, PadSamples: pad, MaxSpeechSamples: max);

    private static float[] Windows(params (float Probability, int Count)[] runs)
        => [.. runs.SelectMany(r => Enumerable.Repeat(r.Probability, r.Count))];

    private static IReadOnlyList<(int Start, int End)> Cut(float[] probabilities, SegmenterSettings settings)
        => SpeechSegmenter.Segment(probabilities, probabilities.Length * Window, settings);

    [Fact]
    public void TwoSpansSeparatedByALongPause_AreCutAtTheSpeechAndPaddedOnBothSides()
    {
        // Speech at windows 10-39 and 50-69; the first pause is closed 100 ms (2048 samples >= 1600) after it begins.
        var probabilities = Windows((0f, 10), (0.9f, 30), (0f, 10), (0.9f, 20), (0f, 10));

        Cut(probabilities, Settings(pad: 480)).Should().Equal(
            (5120 - 480, 20480 + 480),
            (25600 - 480, 35840 + 480));
    }

    [Fact]
    public void SpansWhosePaddingsMeet_ShareTheGapBetweenThem()
    {
        // The gap is 2560 samples, less than twice the 1600 padding: each span takes half, and they touch.
        var probabilities = Windows((0f, 10), (0.9f, 20), (0f, 5), (0.9f, 20), (0f, 10));

        Cut(probabilities, Settings(pad: 1600)).Should().Equal(
            (5120 - 1600, 15360 + 1280),
            (15360 + 1280, 28160 + 1600));
    }

    [Fact]
    public void APauseShorterThanTheMinimumSilence_StaysInsideTheSpan()
    {
        // Three silent windows are 1536 samples of silence at most when speech returns — under the 1600 that ends a span.
        var probabilities = Windows((0.9f, 20), (0f, 3), (0.9f, 20));

        Cut(probabilities, Settings()).Should().Equal((0, 43 * Window));
    }

    [Fact]
    public void ASpanNoLongerThanTheMinimumSpeech_IsDropped()
    {
        Cut(Windows((0f, 10), (0.9f, 5), (0f, 10)), Settings()).Should().BeEmpty();
    }

    [Fact]
    public void AProbabilityBetweenTheTwoThresholds_NeitherStartsNorEndsASpan()
    {
        // 0.4 is under the 0.5 threshold but over the 0.35 negative threshold: speech goes on through it.
        Cut(Windows((0.9f, 10), (0.4f, 30), (0.9f, 10)), Settings()).Should().Equal((0, 50 * Window));

        // 0.3 is under the negative threshold: the same stretch is a pause.
        Cut(Windows((0.9f, 10), (0.3f, 30), (0.9f, 10)), Settings()).Should().Equal((0, 10 * Window), (40 * Window, 50 * Window));
    }

    [Fact]
    public void SpeechRunningToTheEndOfTheRecording_EndsWithIt()
    {
        var probabilities = Windows((0f, 5), (0.9f, 20));

        Cut(probabilities, Settings(pad: 480)).Should().Equal((5 * Window - 480, 25 * Window));
    }

    [Fact]
    public void PaddingNeverLeavesTheRecording()
    {
        Cut(Windows((0.9f, 20)), Settings(pad: 480)).Should().Equal((0, 20 * Window));
    }

    [Fact]
    public void NoSpeech_GivesNoSpans()
    {
        Cut(Windows((0.1f, 40)), Settings()).Should().BeEmpty();
        Cut([], Settings()).Should().BeEmpty();
    }

    [Fact]
    public void ASpanLongerThanTheMaximum_IsCutAtItsLatestPause()
    {
        // The minimum silence is set high, so the pause (5 windows, 2560 samples) does not end the span by itself.
        var settings = new SegmenterSettings(0.5f, 4000, MinSilenceSamples: 16000, PadSamples: 0, MaxSpeechSamples: 8000);
        var probabilities = Windows((0.9f, 8), (0f, 5), (0.9f, 12));

        // Window 16 (8192 samples after the start) outgrows 8000: the span is cut where the pause began (4096), and the
        // next one starts where it ended (6656).
        Cut(probabilities, settings).Should().Equal((0, 8 * Window), (13 * Window, 25 * Window));
    }

    [Fact]
    public void ASpanLongerThanTheMaximumWithNoPause_IsCutAtTheLimit()
    {
        var settings = new SegmenterSettings(0.5f, 4000, MinSilenceSamples: 1600, PadSamples: 0, MaxSpeechSamples: 8000);

        // Window 16 is the first more than 8000 samples after the start.
        Cut(Windows((0.9f, 30)), settings).Should().Equal((0, 16 * Window), (16 * Window, 30 * Window));
    }
}
