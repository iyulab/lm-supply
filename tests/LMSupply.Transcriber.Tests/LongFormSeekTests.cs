using AwesomeAssertions;
using LMSupply.Transcriber.Audio;
using LMSupply.Transcriber.Core;

namespace LMSupply.Transcriber.Tests;

/// <summary>
/// The long-form window rule (docket iyulab/lm-supply#340). A fixed 30 s stride lost the speech at
/// every window boundary: the decoder ends a window at end-of-text after the last segment it can
/// close, and whatever followed in that window was never decoded. The next window has to start where
/// the last closed segment ended.
/// </summary>
public class LongFormSeekTests
{
    private const int Rate = AudioProcessor.WhisperSampleRate;
    private const int Window = AudioProcessor.NumSamples;

    [Fact]
    public void WindowEndingBeforeInputEnd_SeeksToTheLastClosedSegment_AndDropsTheOpenOne()
    {
        // Three segments, two closed by timestamps, the third cut by the window boundary.
        var step = LongFormSeek.Plan(0, Window, 4 * Window, segmentCount: 3, closedSegmentCount: 2, lastClosedEnd: 24.5);

        step.Should().Be(new LongFormSeek.Step(KeepSegments: 2, NextStart: (int)(24.5 * Rate)));
    }

    [Fact]
    public void EveryClosedSegmentEndingEarly_StillSeeksBack()
    {
        // The #340 shape: the decoder closed its last segment at 26.79 s and ended the window there,
        // while the audio went on to 30 s. A fixed stride skipped those 3.2 s entirely.
        var step = LongFormSeek.Plan(0, Window, 4 * Window, segmentCount: 5, closedSegmentCount: 5, lastClosedEnd: 26.79);

        step.KeepSegments.Should().Be(5);
        step.NextStart.Should().Be((int)Math.Round(26.79 * Rate), "the tail after the last closed segment is decoded again");
    }

    [Fact]
    public void SeekIsRelativeToTheWindowStart()
    {
        var start = 3 * Rate;

        var step = LongFormSeek.Plan(start, Window, 4 * Window, segmentCount: 1, closedSegmentCount: 1, lastClosedEnd: 10);

        step.NextStart.Should().Be(start + 10 * Rate);
    }

    [Fact]
    public void LastWindow_KeepsEverySegment_IncludingTheOpenOne()
    {
        var step = LongFormSeek.Plan(Window, 20 * Rate, Window + 20 * Rate, segmentCount: 3, closedSegmentCount: 2, lastClosedEnd: 12);

        step.Should().Be(new LongFormSeek.Step(KeepSegments: 3, NextStart: Window + 20 * Rate));
    }

    [Fact]
    public void NoClosedSegment_ConsumesTheWholeWindow()
    {
        // No timestamps to seek to (one open segment, or silence): the window is taken as it is.
        var step = LongFormSeek.Plan(0, Window, 3 * Window, segmentCount: 1, closedSegmentCount: 0, lastClosedEnd: null);

        step.Should().Be(new LongFormSeek.Step(KeepSegments: 1, NextStart: Window));
    }

    [Fact]
    public void ClosedEndBelowTheMinimumAdvance_ConsumesTheWholeWindow()
    {
        // Seeking 0.4 s forward would re-decode almost the same window; the loop must make progress.
        var step = LongFormSeek.Plan(0, Window, 3 * Window, segmentCount: 2, closedSegmentCount: 1, lastClosedEnd: 0.4);

        step.Should().Be(new LongFormSeek.Step(KeepSegments: 2, NextStart: Window));
    }

    [Fact]
    public void ClosedEndPastTheRealAudio_IsClampedToTheWindow()
    {
        var step = LongFormSeek.Plan(0, Window, 3 * Window, segmentCount: 1, closedSegmentCount: 1, lastClosedEnd: 31.5);

        step.NextStart.Should().Be(Window);
    }

    // A remainder shorter than the minimum is dropped rather than padded into a window of its own:
    // 70 ms padded to 30 s is a window of silence, and Whisper hallucinates text into it (docket #59:
    // "[BLANK_AUDIO]" / "-감사합니다." placed at 60→70 s of a 60.07 s file).

    [Fact]
    public void RemainderShorterThanMinimum_EndsTheInput()
    {
        var total = 2 * Window + 1120; // 60 s + 70 ms, the docket #59 shape

        var step = LongFormSeek.Plan(Window, Window, total, segmentCount: 1, closedSegmentCount: 0, lastClosedEnd: null);

        step.NextStart.Should().Be(total);
    }

    [Fact]
    public void RemainderAtMinimum_IsAWindowOfItsOwn()
    {
        var total = Window + AudioProcessor.MinTailSamples;

        var step = LongFormSeek.Plan(0, Window, total, segmentCount: 1, closedSegmentCount: 0, lastClosedEnd: null);

        step.NextStart.Should().Be(Window);
    }

    [Theory]
    [InlineData(null)]
    [InlineData(0.0)]
    [InlineData(0.99)]
    [InlineData(1.0)]
    [InlineData(17.3)]
    [InlineData(29.99)]
    public void RepeatedWindows_AlwaysReachTheEnd(double? closedEnd)
    {
        // Whatever the decoder reports, every window moves the cursor forward, so the loop ends.
        var total = 7 * Window + 12345;
        var start = 0;
        var windows = 0;

        while (start < total)
        {
            var length = Math.Min(Window, total - start);
            var step = LongFormSeek.Plan(start, length, total, segmentCount: 2, closedSegmentCount: closedEnd is null ? 0 : 1, lastClosedEnd: closedEnd);

            step.NextStart.Should().BeGreaterThanOrEqualTo(start + Math.Min(LongFormSeek.MinAdvanceSamples, length));
            start = step.NextStart;
            windows++;
            windows.Should().BeLessThan(total / LongFormSeek.MinAdvanceSamples + 2);
        }

        start.Should().Be(total);
    }
}
