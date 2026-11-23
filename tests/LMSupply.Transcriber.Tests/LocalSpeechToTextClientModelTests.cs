#pragma warning disable MEAI001 // Microsoft.Extensions.AI's speech-to-text contract is experimental; so is the client under test.

using System.Text.RegularExpressions;
using AwesomeAssertions;
using Microsoft.Extensions.AI;

namespace LMSupply.Transcriber.Tests;

/// <summary>
/// The client over a real Whisper model, on recordings whose words are known (see <c>Fixtures/README.md</c>): what comes
/// back is read, not only counted. A test that asked for "some text" would pass on noise.
/// </summary>
[Trait("Category", "Integration")]
public partial class LocalSpeechToTextClientModelTests
{
    private static string Fixture(string name) => Path.Combine(AppContext.BaseDirectory, "Fixtures", name);

    private static Task<ITranscriberModel> ModelAsync(CancellationToken ct) =>
        LocalTranscriber.LoadAsync("default", cancellationToken: ct);

    /// <summary>The words alone: lower case, no punctuation, single spaces.</summary>
    private static string Words(string text) =>
        NotWord().Replace(text.ToLowerInvariant(), " ").Trim() is var t ? Spaces().Replace(t, " ") : "";

    [GeneratedRegex(@"[^\p{L}\p{N}]+")]
    private static partial Regex NotWord();

    [GeneratedRegex(@"\s+")]
    private static partial Regex Spaces();

    [Fact]
    public async Task EnglishRecording_ComesBackAsItsWords_AcrossSegmentsThatFollowTheRecordingsTimeAxis()
    {
        var ct = TestContext.Current.CancellationToken;
        await using var model = await ModelAsync(ct);
        using var audio = File.OpenRead(Fixture("english-deploy-notice-21s.wav"));

        var response = await model.AsSpeechToTextClient().GetTextAsync(audio, new SpeechToTextOptions { SpeechLanguage = "en" }, ct);

        // The model hears the on-call engineer of the script as an "uncall" engineer: the text is pinned as this model gives it,
        // so that a change of model or decoding shows up here and not in a consumer.
        Words(response.Text).Should().Be(
            "good morning everyone the deployment window for the reporting service opens at 9 o clock on thursday " +
            "please merge your pending changes before wednesday evening and make sure the release notes mention the new export format " +
            "if anything looks wrong after the rollout page the uncall engineer immediately thank you");
        var starts = response.Contents.Select(c => (double)c.AdditionalProperties![LocalSpeechToTextClient.SegmentStartSecondsKey]!).ToArray();
        var ends = response.Contents.Select(c => (double)c.AdditionalProperties![LocalSpeechToTextClient.SegmentEndSecondsKey]!).ToArray();
        response.Contents.Count.Should().BeGreaterThan(1, "21 s of speech with timestamps is more than one segment");
        starts.Should().BeInAscendingOrder();
        ends.Zip(starts, (end, start) => end > start).Should().OnlyContain(ok => ok);
        starts.Skip(1).Zip(ends, (start, previousEnd) => start >= previousEnd - 0.05).Should().OnlyContain(ok => ok);
        ends[^1].Should().BeInRange(15, 22, "the recording is 21.5 s long");
        ((double)response.AdditionalProperties![LocalSpeechToTextClient.DurationSecondsKey]!).Should().BeApproximately(21.5, 0.5);
        response.AdditionalProperties[LocalSpeechToTextClient.LanguageKey].Should().Be("en");
        response.AdditionalProperties.Should().NotContainKey(LocalSpeechToTextClient.LanguageProbabilityKey, "a language was given");
    }

    [Fact]
    public async Task Streaming_YieldsTheSameSegmentsAndTextAsTheOtherForm()
    {
        var ct = TestContext.Current.CancellationToken;
        await using var model = await ModelAsync(ct);
        var client = model.AsSpeechToTextClient();
        var options = new SpeechToTextOptions { SpeechLanguage = "en" };

        using var first = File.OpenRead(Fixture("english-meeting-notes-16s.wav"));
        var whole = await client.GetTextAsync(first, options, ct);
        using var second = File.OpenRead(Fixture("english-meeting-notes-16s.wav"));
        var updates = new List<SpeechToTextResponseUpdate>();
        await foreach (var update in client.GetStreamingTextAsync(second, options, ct))
            updates.Add(update);

        string.Concat(updates.Select(u => u.Text)).Should().Be(whole.Text);
        // A TimeSpan holds ticks (100 ns): the start in seconds survives to a microsecond.
        updates.Select(u => u.StartTime!.Value.TotalSeconds).Zip(
                whole.Contents.Select(c => (double)c.AdditionalProperties![LocalSpeechToTextClient.SegmentStartSecondsKey]!),
                (streamed, seconds) => Math.Abs(streamed - seconds) < 1e-6)
            .Should().OnlyContain(same => same).And.HaveCount(whole.Contents.Count);
    }

    [Fact]
    public async Task ThreeVoices_ComeBackWithThreeSpeakers_WhenTheLibrarysOptionsAskForThem()
    {
        var ct = TestContext.Current.CancellationToken;
        await using var model = await ModelAsync(ct);
        using var audio = File.OpenRead(Fixture("meeting-three-voices.mp3"));
        var options = new SpeechToTextOptions
        {
            SpeechLanguage = "en",
            RawRepresentationFactory = _ => new TranscribeOptions { Diarize = true, WordTimestamps = true },
        };

        var response = await model.AsSpeechToTextClient().GetTextAsync(audio, options, ct);

        string? SpeakerOf(string words) => response.Contents
            .First(c => c.RawRepresentation is TranscriptionSegment s && s.Text.Contains(words, StringComparison.OrdinalIgnoreCase))
            .AdditionalProperties![LocalSpeechToTextClient.SpeakerKey] as string;
        new[] { SpeakerOf("planning meeting"), SpeakerOf("short update"), SpeakerOf("server migration") }
            .Distinct().Should().HaveCount(3, "David, Zira and Mark are three speakers");
        response.Contents.Select(c => c.AdditionalProperties![LocalSpeechToTextClient.SpeakerKey]).Distinct().Should().HaveCount(3);
    }

    [Fact]
    public async Task KoreanSpeech_AsEnglishText_IsTranslated_NotReturnedInKorean()
    {
        var ct = TestContext.Current.CancellationToken;
        await using var model = await ModelAsync(ct);
        using var audio = File.OpenRead(Fixture("korean-meeting-notice-11s.wav"));

        var response = await model.AsSpeechToTextClient().GetTextAsync(audio, new SpeechToTextOptions { SpeechLanguage = "ko", TextLanguage = "en" }, ct);

        // A transcript of this recording is in Hangul; asked for English text it has none. (The translation of a small
        // model is rough, so the words are not pinned: what is pinned is that the translation task ran.)
        response.Text.Should().NotBeNullOrWhiteSpace().And.MatchRegex("[A-Za-z]{3}").And.NotMatchRegex("[\uAC00-\uD7A3]");

        audio.Position = 0;
        var plain = await model.AsSpeechToTextClient().GetTextAsync(audio, new SpeechToTextOptions { SpeechLanguage = "ko" }, ct);
        plain.Text.Should().MatchRegex("[\uAC00-\uD7A3]{2}", "without TextLanguage the speech comes back in its own language");
    }
}
