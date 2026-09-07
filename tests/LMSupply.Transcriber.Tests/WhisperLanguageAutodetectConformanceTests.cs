using AwesomeAssertions;

namespace LMSupply.Transcriber.Tests;

/// <summary>
/// Conformance against a real Whisper model (excluded from CI via <c>Category=Integration</c>; the
/// model is downloaded on first run): a non-English clip transcribed <b>without a language hint</b>
/// must come back in its own language, with a detection probability — the behaviour the
/// documentation promised ("null for auto-detect") and the SOT prompt made impossible until 0.58.0.
/// The fixture is an 11 s Korean clip synthesized once, offline, from a local TTS voice and checked
/// in (never generated at test time). The hinted call is the positive control: if it fails, the model
/// or the audio is the problem, not detection.
/// </summary>
[Trait("Category", "Integration")]
public sealed class WhisperLanguageAutodetectConformanceTests
{
    private static readonly string s_fixture = Path.Combine(AppContext.BaseDirectory, "Fixtures", "korean-meeting-notice-11s.wav");

    [Fact]
    public async Task KoreanClip_NoHint_IsDetectedAsKoreanAndTranscribed()
    {
        await using var model = await LocalTranscriber.LoadAsync("default", cancellationToken: TestContext.Current.CancellationToken);

        var result = await model.TranscribeAsync(s_fixture, cancellationToken: TestContext.Current.CancellationToken);

        result.Language.Should().Be("ko", "the clip is Korean and no hint was given — detection must actually run");
        result.LanguageProbability.Should().NotBeNull("a detection that ran leaves its probability behind");
        result.LanguageProbability!.Value.Should().BeGreaterThan(0.5f);
        result.Text.Should().NotBeNullOrWhiteSpace("Korean decoded as English used to collapse into an empty transcript");
        result.Text.Should().MatchRegex("[가-힣]", "the transcript must contain Hangul, not an English mis-decode");
    }

    [Fact]
    public async Task KoreanClip_WithHint_IsTranscribed_PositiveControl()
    {
        await using var model = await LocalTranscriber.LoadAsync("default", cancellationToken: TestContext.Current.CancellationToken);

        var result = await model.TranscribeAsync(
            s_fixture,
            new TranscribeOptions { Language = "ko" },
            cancellationToken: TestContext.Current.CancellationToken);

        result.Language.Should().Be("ko");
        result.Text.Should().MatchRegex("[가-힣]");
    }
}
