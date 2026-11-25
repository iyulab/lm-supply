using AwesomeAssertions;
using LMSupply.Download;
using LMSupply.Transcriber.Audio;
using LMSupply.Transcriber.Diarization;

namespace LMSupply.Transcriber.Tests.Diarization;

/// <summary>
/// The diarizer reproduces sherpa-onnx's offline pyannote diarization on sherpa's own test recording with the same
/// models and settings (threshold 0.5, min on 0.3 s, min off 0.5 s). The reference turns were produced with the
/// sherpa-onnx Python package from the same files. The library's default cut is 0.4, so these facts pass sherpa's
/// 0.5 explicitly; the small-cluster rule the library adds does not change the reference turns.
/// </summary>
[Trait("Category", "Integration")]
public class SpeakerDiarizerIntegrationTests
{
    private const string RecordingUrl =
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/0-four-speakers-zh.wav";

    private static readonly (double Start, double End, int Speaker)[] SherpaReference =
    [
        (0.318, 6.865, 0), (7.017, 10.747, 1), (11.455, 17.041, 1), (22.137, 24.837, 0), (27.638, 29.478, 1),
        (30.001, 31.553, 1), (33.680, 37.932, 1), (48.040, 50.470, 1), (52.529, 54.605, 1),
    ];

    private static async Task<string> RecordingAsync(CancellationToken ct)
    {
        var path = Path.Combine(CacheManager.GetDefaultCacheDirectory(), "test-fixtures", "0-four-speakers-zh.wav");
        if (!File.Exists(path))
        {
            Directory.CreateDirectory(Path.GetDirectoryName(path)!);
            using var http = new HttpClient();
            await File.WriteAllBytesAsync(path, await http.GetByteArrayAsync(RecordingUrl, ct), ct);
        }
        return path;
    }

    [Fact]
    public async Task MatchesSherpaOnnxOnItsTestRecording()
    {
        var ct = TestContext.Current.CancellationToken;
        var samples = await AudioProcessor.LoadAudioAsync(await RecordingAsync(ct), ct);
        await using var diarizer = await SpeakerDiarizer.LoadAsync(
            CacheManager.GetDefaultCacheDirectory(), localFilesOnly: false, ExecutionProvider.Cpu, progress: null, ct);

        var turns = diarizer.Diarize(samples, threshold: 0.5f, cancellationToken: ct);

        turns.Select(t => t.Speaker).Distinct().Should().HaveCount(2);
        turns.Should().HaveCount(SherpaReference.Length);
        for (var i = 0; i < turns.Count; i++)
        {
            turns[i].Start.Should().BeApproximately(SherpaReference[i].Start, 0.05);
            turns[i].End.Should().BeApproximately(SherpaReference[i].End, 0.05);
            turns[i].Speaker.Should().Be(SherpaReference[i].Speaker);
        }
    }

    /// <summary>
    /// The user-facing path: a Whisper transcription of a multi-speaker recording with <c>Diarize</c> labels every
    /// segment, the labels follow the diarizer's turns, and without <c>Diarize</c> nothing is labelled.
    /// </summary>
    [Fact]
    public async Task TranscribeWithDiarize_LabelsSegmentsBySpeaker()
    {
        var ct = TestContext.Current.CancellationToken;
        var path = await RecordingAsync(ct);
        await using var model = await LocalTranscriber.LoadAsync("default", cancellationToken: ct);

        var labelled = await model.TranscribeAsync(
            path, new TranscribeOptions { Diarize = true, Language = "zh", SpeakerThreshold = 0.5f }, ct);
        var plain = await model.TranscribeAsync(path, new TranscribeOptions { Language = "zh" }, ct);

        labelled.Segments.Should().NotBeEmpty();
        labelled.Segments.Should().OnlyContain(s => s.Speaker != null);
        labelled.Segments.Select(s => s.Speaker).Distinct().Should().HaveCount(2, "sherpa's reference finds two voices at 0.5");
        labelled.Segments[0].Speaker.Should().Be("S1", "labels are numbered in speaking order");
        plain.Segments.Should().OnlyContain(s => s.Speaker == null);
    }

    [Fact]
    public async Task AFixedSpeakerCount_OverridesTheThreshold()
    {
        var ct = TestContext.Current.CancellationToken;
        var samples = await AudioProcessor.LoadAudioAsync(await RecordingAsync(ct), ct);
        await using var diarizer = await SpeakerDiarizer.LoadAsync(
            CacheManager.GetDefaultCacheDirectory(), localFilesOnly: false, ExecutionProvider.Cpu, progress: null, ct);

        var turns = diarizer.Diarize(samples, numSpeakers: 4, cancellationToken: ct);

        turns.Select(t => t.Speaker).Distinct().Should().HaveCount(4, "the recording has four speakers");
    }

    /// <summary>
    /// The public surface gives sherpa's reference turns on its own test recording — the same turns the internal
    /// diarizer gives — from a path, from a stream and from bytes, with labels in speaking order.
    /// </summary>
    [Fact]
    public async Task PublicSurface_ReproducesTheSherpaReference_FromPathStreamAndBytes()
    {
        var ct = TestContext.Current.CancellationToken;
        var path = await RecordingAsync(ct);
        await using var diarizer = await LocalSpeakerDiarizer.LoadAsync(cancellationToken: ct);
        var options = new DiarizeOptions { SpeakerThreshold = 0.5f };

        await using var stream = File.OpenRead(path);
        var results = new[]
        {
            await diarizer.DiarizeAsync(path, options, ct),
            await diarizer.DiarizeAsync(stream, options, ct),
            await diarizer.DiarizeAsync(await File.ReadAllBytesAsync(path, ct), options, ct),
        };

        foreach (var result in results)
        {
            result.Turns.Should().HaveCount(SherpaReference.Length);
            result.SpeakerCount.Should().Be(2);
            result.DurationSeconds.Should().BeApproximately(56.0, 5.0, "a recording of about a minute");
            for (var i = 0; i < SherpaReference.Length; i++)
            {
                result.Turns[i].Start.Should().BeApproximately(SherpaReference[i].Start, 0.05);
                result.Turns[i].End.Should().BeApproximately(SherpaReference[i].End, 0.05);
                result.Turns[i].Speaker.Should().Be($"S{SherpaReference[i].Speaker + 1}");
            }
        }
    }

    [Fact]
    public async Task PublicSurface_ASpeakerCountFixesTheCount_AndABoundCapsIt()
    {
        var ct = TestContext.Current.CancellationToken;
        var path = await RecordingAsync(ct);
        await using var diarizer = await LocalSpeakerDiarizer.LoadAsync(cancellationToken: ct);

        var four = await diarizer.DiarizeAsync(path, new DiarizeOptions { NumSpeakers = 4 }, ct);
        var capped = await diarizer.DiarizeAsync(path, new DiarizeOptions { NumSpeakers = null, MaxSpeakers = 1 }, ct);

        four.SpeakerCount.Should().Be(4, "the recording has four speakers");
        capped.SpeakerCount.Should().Be(1);
        capped.Turns.Should().OnlyContain(t => t.Speaker == "S1");
    }

    /// <summary>The standalone diarizer and a diarized transcription name the same voices the same way.</summary>
    [Fact]
    public async Task PublicSurface_LabelsAgreeWithADiarizedTranscription()
    {
        var ct = TestContext.Current.CancellationToken;
        var path = await RecordingAsync(ct);
        await using var diarizer = await LocalSpeakerDiarizer.LoadAsync(cancellationToken: ct);
        await using var model = await LocalTranscriber.LoadAsync("default", cancellationToken: ct);

        var turns = (await diarizer.DiarizeAsync(path, new DiarizeOptions { SpeakerThreshold = 0.5f }, ct)).Turns;
        var transcript = await model.TranscribeAsync(
            path, new TranscribeOptions { Diarize = true, Language = "zh", SpeakerThreshold = 0.5f }, ct);

        transcript.Segments.Should().NotBeEmpty();
        foreach (var segment in transcript.Segments)
        {
            var mid = (segment.Start + segment.End) / 2;
            var inside = turns.Where(t => t.Start <= mid && mid <= t.End).Select(t => t.Speaker).ToList();
            if (inside.Count > 0)
                inside.Should().Contain(segment.Speaker, $"segment at {mid:F1} s is labelled with a voice talking then");
        }
    }

    [Fact]
    public async Task PublicSurface_RefusesAudioThatIsNoAudio()
    {
        var ct = TestContext.Current.CancellationToken;
        await using var diarizer = await LocalSpeakerDiarizer.LoadAsync(cancellationToken: ct);

        await FluentActions.Awaiting(() => diarizer.DiarizeAsync(Array.Empty<byte>(), cancellationToken: ct))
            .Should().ThrowAsync<InvalidDataException>();
        await FluentActions.Awaiting(() => diarizer.DiarizeAsync(new byte[] { 1, 2, 3, 4 }, cancellationToken: ct))
            .Should().ThrowAsync<InvalidDataException>();
        await FluentActions.Awaiting(() => diarizer.DiarizeAsync(new byte[10], new DiarizeOptions { NumSpeakers = 0 }, ct))
            .Should().ThrowAsync<ArgumentOutOfRangeException>("the settings are checked before the audio is read");
    }
}
