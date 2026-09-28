using AwesomeAssertions;
using LMSupply.Download;
using LMSupply.Transcriber.Audio;
using LMSupply.Transcriber.Diarization;

namespace LMSupply.Transcriber.Tests.Diarization;

/// <summary>
/// Speaker counts on two synthetic meetings with the same 18-line script (see <c>Fixtures/README.md</c>): one read by
/// three voices, two of them male, and one read by a single voice. The three-voice truth is
/// <c>A A A A A B B C C C A A A C A B A A</c>.
/// </summary>
[Trait("Category", "Integration")]
public class DiarizationMeetingConformanceTests
{
    private static readonly string ThreeVoices = Path.Combine(AppContext.BaseDirectory, "Fixtures", "meeting-three-voices.mp3");
    private static readonly string OneVoice = Path.Combine(AppContext.BaseDirectory, "Fixtures", "meeting-one-voice.mp3");

    private static Task<ITranscriberModel> ModelAsync(CancellationToken ct) =>
        LocalTranscriber.LoadAsync("default", cancellationToken: ct);

    private static Task<SpeakerDiarizer> DiarizerAsync(CancellationToken ct) =>
        SpeakerDiarizer.LoadAsync(CacheManager.GetDefaultCacheDirectory(), localFilesOnly: false, ExecutionProvider.Cpu, progress: null, ct);

    /// <summary>One line of each voice, picked by its words, must carry three different labels.</summary>
    private static void ShouldSeparateTheThreeVoices(TranscriptionResult result)
    {
        string SpeakerOf(string words) => result.Segments.First(s => s.Text.Contains(words, StringComparison.OrdinalIgnoreCase)).Speaker!;

        var a = SpeakerOf("planning meeting");
        var b = SpeakerOf("short update");
        var c = SpeakerOf("server migration");
        new[] { a, b, c }.Distinct().Should().HaveCount(3, "David, Zira and Mark are three speakers");
        result.Segments.Select(s => s.Speaker).Distinct().Should().HaveCount(3);
    }

    [Fact]
    public async Task ThreeVoices_TheDefaultFindsThree()
    {
        var ct = TestContext.Current.CancellationToken;
        await using var model = await ModelAsync(ct);

        var result = await model.TranscribeAsync(ThreeVoices, new TranscribeOptions { Diarize = true, Language = "en" }, ct);

        ShouldSeparateTheThreeVoices(result);
    }

    /// <summary>
    /// A requested count is met by large clusters. Before the small-cluster rule, a one-window outlier took one of the
    /// requested slots and vanished in aggregation, so asking for 3 gave 2 and asking for 2 gave 1.
    /// </summary>
    [Fact]
    public async Task ThreeVoices_NumSpeakersIsHonoured()
    {
        var ct = TestContext.Current.CancellationToken;
        await using var model = await ModelAsync(ct);

        var three = await model.TranscribeAsync(ThreeVoices, new TranscribeOptions { Diarize = true, Language = "en", NumSpeakers = 3 }, ct);
        var two = await model.TranscribeAsync(ThreeVoices, new TranscribeOptions { Diarize = true, Language = "en", NumSpeakers = 2 }, ct);

        ShouldSeparateTheThreeVoices(three);
        two.Segments.Select(s => s.Speaker).Distinct().Should().HaveCount(2);
    }

    /// <summary>One voice stays one speaker at the default cut and under an upper bound (an attendee count).</summary>
    [Fact]
    public async Task OneVoice_StaysOneSpeaker_ByDefaultAndUnderMaxSpeakers()
    {
        var ct = TestContext.Current.CancellationToken;
        await using var model = await ModelAsync(ct);

        var byDefault = await model.TranscribeAsync(OneVoice, new TranscribeOptions { Diarize = true, Language = "en" }, ct);
        var bounded = await model.TranscribeAsync(OneVoice, new TranscribeOptions { Diarize = true, Language = "en", MaxSpeakers = 4 }, ct);

        byDefault.Segments.Select(s => s.Speaker).Distinct().Should().ContainSingle();
        bounded.Segments.Select(s => s.Speaker).Distinct().Should().ContainSingle();
    }

    /// <summary>
    /// A recording of one segmentation window (10 s or less) goes through clustering too, so a count applies to it.
    /// Before, it returned the window's local speakers as they were and never read the count or the threshold.
    /// </summary>
    [Fact]
    public async Task AShortRecording_HonoursTheSpeakerCount()
    {
        var ct = TestContext.Current.CancellationToken;
        var samples = await AudioProcessor.LoadAudioAsync(ThreeVoices, ct);
        var clip = samples.AsSpan(16 * 16000, 9 * 16000).ToArray(); // the end of A's agenda, then B's first line
        await using var diarizer = await DiarizerAsync(ct);

        var estimated = diarizer.Diarize(clip, cancellationToken: ct);
        var one = diarizer.Diarize(clip, numSpeakers: 1, cancellationToken: ct);

        estimated.Select(t => t.Speaker).Distinct().Should().HaveCount(2);
        one.Select(t => t.Speaker).Distinct().Should().ContainSingle();
    }
}
