using AwesomeAssertions;
using LMSupply.Download;
using LMSupply.Transcriber.Audio;
using LMSupply.Transcriber.Diarization;

namespace LMSupply.Transcriber.Tests.Diarization;

/// <summary>
/// The diarizer reproduces sherpa-onnx's offline pyannote diarization on sherpa's own test recording with the same
/// models and settings (threshold 0.5, min on 0.3 s, min off 0.5 s). The reference turns were produced with the
/// sherpa-onnx Python package from the same files.
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
            CacheManager.GetDefaultCacheDirectory(), localFilesOnly: false, ExecutionProvider.Cpu, ct);

        var turns = diarizer.Diarize(samples, cancellationToken: ct);

        turns.Select(t => t.Speaker).Distinct().Should().HaveCount(2);
        turns.Should().HaveCount(SherpaReference.Length);
        for (var i = 0; i < turns.Count; i++)
        {
            turns[i].Start.Should().BeApproximately(SherpaReference[i].Start, 0.05);
            turns[i].End.Should().BeApproximately(SherpaReference[i].End, 0.05);
            turns[i].Speaker.Should().Be(SherpaReference[i].Speaker);
        }
    }

    [Fact]
    public async Task AFixedSpeakerCount_OverridesTheThreshold()
    {
        var ct = TestContext.Current.CancellationToken;
        var samples = await AudioProcessor.LoadAudioAsync(await RecordingAsync(ct), ct);
        await using var diarizer = await SpeakerDiarizer.LoadAsync(
            CacheManager.GetDefaultCacheDirectory(), localFilesOnly: false, ExecutionProvider.Cpu, ct);

        var turns = diarizer.Diarize(samples, numSpeakers: 4, cancellationToken: ct);

        turns.Select(t => t.Speaker).Distinct().Should().HaveCount(4, "the recording has four speakers");
    }
}
