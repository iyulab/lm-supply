using AwesomeAssertions;
using LMSupply.Download;
using LMSupply.Transcriber.Audio;
using LMSupply.Transcriber.Vad;

namespace LMSupply.Transcriber.Tests.Vad;

/// <summary>
/// The detector on recordings whose speech and silence are known by construction: two spoken fixtures placed at known
/// offsets inside digital silence. The model is Silero VAD (fetched once into the shared cache).
/// </summary>
[Trait("Category", "Integration")]
public class VoiceActivityDetectorIntegrationTests
{
    private const int Rate = 16000;
    private static string Fixture(string name) => Path.Combine(AppContext.BaseDirectory, "Fixtures", name);

    private static float[] Silence(double seconds) => new float[(int)(seconds * Rate)];

    // A 16-bit PCM mono WAV of the samples, for the byte-array and stream inputs.
    private static byte[] Wav(float[] samples)
    {
        using var stream = new MemoryStream();
        using (var w = new BinaryWriter(stream, System.Text.Encoding.ASCII, leaveOpen: true))
        {
            w.Write("RIFF"u8); w.Write(36 + samples.Length * 2); w.Write("WAVEfmt "u8);
            w.Write(16); w.Write((short)1); w.Write((short)1); w.Write(Rate); w.Write(Rate * 2); w.Write((short)2); w.Write((short)16);
            w.Write("data"u8); w.Write(samples.Length * 2);
            foreach (var s in samples)
                w.Write((short)Math.Clamp(Math.Round(s * 32767), -32768, 32767));
        }
        return stream.ToArray();
    }

    // Where a recording is audibly louder than its own noise floor, found from the samples alone (20 ms frames over -40 dBFS):
    // a fixture opens and closes with a breath of silence, and that is where speech detection is expected to stop.
    private static (double Start, double End) LoudBounds(float[] samples)
    {
        const int frame = Rate / 50;
        var loud = new List<int>();
        for (var f = 0; f + frame <= samples.Length; f += frame)
        {
            var sum = 0.0;
            for (var i = f; i < f + frame; i++)
                sum += samples[i] * samples[i];
            if (Math.Sqrt(sum / frame) > 0.01)
                loud.Add(f);
        }
        return (loud[0] / (double)Rate, (loud[^1] + frame) / (double)Rate);
    }

    private sealed record Scene(float[] Samples, (double Start, double End) Korean, (double Start, double End) English);

    // 2 s silence, the Korean notice, 3 s silence, the English meeting notes, 2 s silence — and where each is audible.
    private static async Task<Scene> BuildAsync(CancellationToken ct)
    {
        var korean = await AudioProcessor.LoadAudioAsync(Fixture("korean-meeting-notice-11s.wav"), ct);
        var english = await AudioProcessor.LoadAudioAsync(Fixture("english-meeting-notes-16s.wav"), ct);
        var samples = new[] { Silence(2.0), korean, Silence(3.0), english, Silence(2.0) }.SelectMany(a => a).ToArray();
        var koreanStart = 2.0;
        var englishStart = 2.0 + korean.Length / (double)Rate + 3.0;
        var k = LoudBounds(korean);
        var e = LoudBounds(english);
        return new Scene(samples, (koreanStart + k.Start, koreanStart + k.End), (englishStart + e.Start, englishStart + e.End));
    }

    [Fact]
    public async Task SpeechInsideSilence_IsFoundAtItsKnownPlace_OneSpanPerSpokenFixture()
    {
        var ct = TestContext.Current.CancellationToken;
        var scene = await BuildAsync(ct);
        await using var detector = await LocalVoiceActivityDetector.LoadAsync(cancellationToken: ct);

        // The pauses inside a fixture are under a second, the silences between them over two: a 1.2 s minimum silence
        // leaves exactly the two spoken stretches.
        var result = await detector.DetectAsync(Wav(scene.Samples), new VadOptions { MinSilenceDuration = TimeSpan.FromSeconds(1.2) }, ct);

        result.DurationSeconds.Should().BeApproximately(scene.Samples.Length / (double)Rate, 0.01);
        result.Segments.Should().HaveCount(2);
        // A span begins no earlier than the padding before the first audible sound, no later than half a second after it (a
        // breath is audible but not speech), and ends symmetrically.
        result.Segments[0].Start.Should().BeInRange(scene.Korean.Start - 0.1, scene.Korean.Start + 0.5);
        result.Segments[0].End.Should().BeInRange(scene.Korean.End - 0.5, scene.Korean.End + 0.1);
        result.Segments[1].Start.Should().BeInRange(scene.English.Start - 0.1, scene.English.Start + 0.5);
        result.Segments[1].End.Should().BeInRange(scene.English.End - 0.5, scene.English.End + 0.1);
        result.SpeechSeconds.Should().BeGreaterThan(0.9 * ((scene.Korean.End - scene.Korean.Start) + (scene.English.End - scene.English.Start)));
    }

    [Fact]
    public async Task ASmallerMinimumSilence_FindsThePausesInsideASpokenStretch()
    {
        var ct = TestContext.Current.CancellationToken;
        var scene = await BuildAsync(ct);
        await using var detector = await LocalVoiceActivityDetector.LoadAsync(cancellationToken: ct);

        var whole = await detector.DetectAsync(Wav(scene.Samples), new VadOptions { MinSilenceDuration = TimeSpan.FromSeconds(1.2) }, ct);
        var phrases = await detector.DetectAsync(Wav(scene.Samples), new VadOptions { MinSilenceDuration = TimeSpan.FromMilliseconds(300) }, ct);

        // The Korean notice has two pauses of about 0.7 s and 0.9 s between its sentences.
        phrases.Segments.Count(s => s.Start >= scene.Korean.Start - 0.1 && s.End <= scene.Korean.End + 0.1).Should().BeGreaterThanOrEqualTo(3);
        phrases.Segments.Should().HaveCountGreaterThan(whole.Segments.Count);
        // No span reaches into the silences between the fixtures.
        foreach (var span in phrases.Segments)
        {
            var inKorean = span.Start >= scene.Korean.Start - 0.1 && span.End <= scene.Korean.End + 0.1;
            var inEnglish = span.Start >= scene.English.Start - 0.1 && span.End <= scene.English.End + 0.1;
            (inKorean || inEnglish).Should().BeTrue($"the span {span.Start:F2}-{span.End:F2} s lies within one of the spoken fixtures");
        }
    }

    // The model card's procedure: windows of 512 samples, each preceded by the last 64 samples of the one before, with the
    // recurrent state carried from window to window. The values are the probabilities this implementation gives on the fixture
    // (a window each where the context, the state or neither matters most), so they guard the procedure against a slip — a
    // window fed without its context moves by 0.1 to 0.13, one fed without the carried state by 0.4 to 0.95.
    [Fact]
    public async Task TheProbabilities_FollowTheModelCardsProcedure()
    {
        var ct = TestContext.Current.CancellationToken;
        var samples = await AudioProcessor.LoadAudioAsync(Fixture("korean-meeting-notice-11s.wav"), ct);
        await using var vad = await SileroVad.LoadAsync(CacheManager.GetDefaultCacheDirectory(), localFilesOnly: false, ExecutionProvider.Cpu, progress: null, ct);

        var probabilities = vad.Probabilities(samples, ct);

        probabilities.Should().HaveCount((samples.Length + 511) / 512, "the last, partial window is padded and read too");
        foreach (var (window, expected) in new[] { (13, 0.9981f), (36, 0.9987f), (86, 0.8051f), (266, 0.9525f), (334, 0.3296f) })
            probabilities[window].Should().BeApproximately(expected, 0.03f, $"window {window}");
        probabilities.Take(357).Count(p => p >= 0.5f).Should().BeInRange(272, 278);
    }

    [Fact]
    public async Task APathAStreamAndBytes_GiveTheSameSpans()
    {
        var ct = TestContext.Current.CancellationToken;
        var path = Fixture("english-deploy-notice-21s.wav");
        var bytes = await File.ReadAllBytesAsync(path, ct);
        await using var detector = await LocalVoiceActivityDetector.LoadAsync(cancellationToken: ct);

        var fromPath = await detector.DetectAsync(path, cancellationToken: ct);
        await using var stream = new MemoryStream(bytes);
        var fromStream = await detector.DetectAsync(stream, cancellationToken: ct);
        var fromBytes = await detector.DetectAsync(bytes, cancellationToken: ct);

        fromPath.Segments.Should().NotBeEmpty();
        fromStream.Segments.Should().Equal(fromPath.Segments);
        fromBytes.Segments.Should().Equal(fromPath.Segments);
    }

    [Fact]
    public async Task AMaximumSpeechDuration_BoundsEverySpan()
    {
        var ct = TestContext.Current.CancellationToken;
        await using var detector = await LocalVoiceActivityDetector.LoadAsync(cancellationToken: ct);
        var path = Fixture("english-deploy-notice-21s.wav");

        var unbounded = await detector.DetectAsync(path, new VadOptions { MinSilenceDuration = TimeSpan.FromSeconds(2) }, ct);
        var bounded = await detector.DetectAsync(
            path, new VadOptions { MinSilenceDuration = TimeSpan.FromSeconds(2), MaxSpeechDuration = TimeSpan.FromSeconds(5) }, ct);

        unbounded.Segments.Max(s => s.Duration).Should().BeGreaterThan(5, "the notice is spoken in one stretch longer than the bound");
        // Padding (30 ms on each side) may widen a cut span a little beyond the bound.
        bounded.Segments.Should().OnlyContain(s => s.Duration <= 5.1);
        bounded.Segments.Count.Should().BeGreaterThan(unbounded.Segments.Count);
    }

    [Fact]
    public async Task NoiseAndATone_AreNotSpeech()
    {
        var ct = TestContext.Current.CancellationToken;
        var random = new Random(7);
        var noise = Enumerable.Range(0, 6 * Rate).Select(_ => (float)(random.NextDouble() * 2 - 1) * 0.1f).ToArray();
        var tone = Enumerable.Range(0, 4 * Rate).Select(i => 0.3f * MathF.Sin(2 * MathF.PI * 440 * i / Rate)).ToArray();
        await using var detector = await LocalVoiceActivityDetector.LoadAsync(cancellationToken: ct);

        var result = await detector.DetectAsync(Wav([.. noise, .. tone, .. Silence(2)]), cancellationToken: ct);

        result.Segments.Should().BeEmpty();
        result.SpeechSeconds.Should().Be(0);
    }

    [Fact]
    public async Task RefusesAudioThatIsNoAudio_AndBadSettingsBeforeReadingAny()
    {
        var ct = TestContext.Current.CancellationToken;
        await using var detector = await LocalVoiceActivityDetector.LoadAsync(cancellationToken: ct);

        await FluentActions.Awaiting(() => detector.DetectAsync(Array.Empty<byte>(), cancellationToken: ct))
            .Should().ThrowAsync<InvalidDataException>();
        await FluentActions.Awaiting(() => detector.DetectAsync(new byte[] { 1, 2, 3, 4 }, cancellationToken: ct))
            .Should().ThrowAsync<InvalidDataException>();
        await FluentActions.Awaiting(() => detector.DetectAsync(new byte[10], new VadOptions { Threshold = 2f }, ct))
            .Should().ThrowAsync<ArgumentOutOfRangeException>("the settings are checked before the audio is read");
    }
}
