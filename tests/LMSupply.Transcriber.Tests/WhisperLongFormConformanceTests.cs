using AwesomeAssertions;

namespace LMSupply.Transcriber.Tests;

/// <summary>
/// Conformance against a real Whisper model (excluded from CI via <c>Category=Integration</c>; the
/// model is downloaded on first run): a 131 s Korean clip — five 30 s windows — must come back with
/// the speech that sits across its window boundaries. Before 0.68.2 windows advanced by a fixed 30 s:
/// the decoder ended each window after the last segment it could close, so the 3–4 s before every
/// boundary were never decoded, the next window opened mid-sentence, and one window fell into a
/// repetition loop (docket iyulab/lm-supply#340). The fixture was synthesized once, offline, from a
/// local TTS voice and checked in.
/// </summary>
[Trait("Category", "Integration")]
public sealed class WhisperLongFormConformanceTests
{
    private static readonly string s_fixture = Path.Combine(AppContext.BaseDirectory, "Fixtures", "korean-meeting-minutes-131s.mp3");

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public async Task SpeechAcrossWindowBoundaries_IsTranscribed(bool segmentTimestamps)
    {
        await using var model = await LocalTranscriber.LoadAsync("default", cancellationToken: TestContext.Current.CancellationToken);

        var result = await model.TranscribeAsync(
            s_fixture,
            new TranscribeOptions { WordTimestamps = segmentTimestamps },
            cancellationToken: TestContext.Current.CancellationToken);

        // Each of these is spoken just before a 30 s boundary; a fixed stride dropped all three.
        result.Text.Should().Contain("포장 디자인", "the sentence spoken at 26.8–30 s");
        result.Text.Should().Contain("동영상 광고", "the sentence spoken at 57–60 s");
        result.Text.Should().Contain("다음 회의는", "the sentence spoken at 1:56–2:00");
        result.Text.Should().Contain("수고하셨습니다", "the clip's last words");
    }

    [Fact]
    public async Task Segments_AreOrderedAndDoNotOverlap()
    {
        await using var model = await LocalTranscriber.LoadAsync("default", cancellationToken: TestContext.Current.CancellationToken);

        var result = await model.TranscribeAsync(
            s_fixture,
            new TranscribeOptions { WordTimestamps = true },
            cancellationToken: TestContext.Current.CancellationToken);

        result.Segments.Should().HaveCountGreaterThanOrEqualTo(20, "the clip is 26 sentences");
        for (var i = 1; i < result.Segments.Count; i++)
        {
            result.Segments[i].Start.Should().BeGreaterThanOrEqualTo(result.Segments[i - 1].End - 0.05,
                "a window re-decoded from a seek point must not repeat the previous window's segments");
        }

        result.Segments.Select(s => s.Text).Should().OnlyHaveUniqueItems("a repetition loop repeats segment texts");
    }

    /// <summary>
    /// The last window starts at a seek point with only a few seconds of audio left and is padded to
    /// 30 s. A language hint that does not match the speech drives the decoder into temperature
    /// fallback, and a sampled decode over the padding closed segments up to 16 s past the end of the
    /// input (docket iyulab/lm-supply#347). Several runs, because the fallback samples.
    /// </summary>
    [Fact]
    public async Task WithAMismatchedLanguageHint_NoSegmentEndsPastTheAudio()
    {
        await using var model = await LocalTranscriber.LoadAsync("default", cancellationToken: TestContext.Current.CancellationToken);

        for (var run = 0; run < 3; run++)
        {
            var result = await model.TranscribeAsync(
                s_fixture,
                new TranscribeOptions { WordTimestamps = true, Language = "en" },
                cancellationToken: TestContext.Current.CancellationToken);

            result.DurationSeconds.Should().NotBeNull();
            var duration = result.DurationSeconds!.Value;
            foreach (var segment in result.Segments)
            {
                segment.End.Should().BeLessThanOrEqualTo(duration + 0.02,
                    $"run {run}: segment {segment} must not end past the {duration:F2} s input");
            }
        }
    }
}
