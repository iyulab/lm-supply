using AwesomeAssertions;

namespace LMSupply.Transcriber.Tests;

/// <summary>
/// Conformance against a real Whisper model (excluded from CI via <c>Category=Integration</c>; the
/// model is downloaded on first run): a 16 s English clip — one 30 s window — must come back with
/// its <b>final sentence</b> and with its ordinary words intact. Before 0.63.1 a blanket repetition
/// penalty on the last ten tokens let end-of-text beat the final sentence's opener ("The", used a
/// few tokens earlier), so the transcript stopped one sentence short while the segment still claimed
/// to cover the whole clip; the same penalty bent "the" into "The" mid-sentence. The fixture was
/// synthesized once, offline, from a local TTS voice and checked in.
/// </summary>
[Trait("Category", "Integration")]
public sealed class WhisperTrailingSentenceConformanceTests
{
    private static readonly string s_fixture = Path.Combine(AppContext.BaseDirectory, "Fixtures", "english-meeting-notes-16s.wav");

    [Fact]
    public async Task SingleWindowClip_KeepsItsFinalSentenceAndArticles()
    {
        await using var model = await LocalTranscriber.LoadAsync("default", cancellationToken: TestContext.Current.CancellationToken);

        var result = await model.TranscribeAsync(s_fixture, cancellationToken: TestContext.Current.CancellationToken);

        result.Text.Should().Contain("next meeting is on Tuesday",
            "the clip's last sentence is ordinary speech; a transcript that stops before it is a silent loss");
        result.Text.Should().Contain("the budget sheet",
            "an article repeated a few tokens later is not a hallucination loop");
    }
}
