using AwesomeAssertions;
using LMSupply.Transcriber.Audio;

namespace LMSupply.Transcriber.Tests;

/// <summary>
/// Audio as a browser records it — WebM/Opus from Chromium's MediaRecorder (written live, so its Segment size is
/// unknown), Ogg/Opus from Firefox — decodes on the device, from a file or from bytes with no file name. MP4/AAC
/// (Safari) is refused with a message that says what is supported. Fixtures: a 1 s 440 Hz tone, encoded with ffmpeg.
/// </summary>
public class OpusAudioTests
{
    private const int WhisperSampleRate = 16000;

    private static string Fixture(string name) => Path.Combine(AppContext.BaseDirectory, "Fixtures", name);

    /// <summary>Sign changes per second over the middle half second — about 880 for a 440 Hz tone.</summary>
    private static double ZeroCrossingsPerSecond(float[] samples)
    {
        var from = samples.Length / 4;
        var to = from + (WhisperSampleRate / 2);
        var crossings = 0;
        for (var i = from + 1; i < to && i < samples.Length; i++)
        {
            if ((samples[i - 1] < 0) != (samples[i] < 0))
                crossings++;
        }

        return crossings * 2.0;
    }

    [Theory]
    [InlineData("tone-440hz-1s-live.webm")]
    [InlineData("tone-440hz-1s.ogg")]
    [InlineData("tone-440hz-1s-stereo.webm")]
    public async Task LoadAudioAsync_BrowserRecording_DecodesToTheTone(string fixture)
    {
        var samples = await AudioProcessor.LoadAudioAsync(Fixture(fixture), TestContext.Current.CancellationToken);

        samples.Length.Should().BeInRange((int)(WhisperSampleRate * 0.95), (int)(WhisperSampleRate * 1.05));
        ZeroCrossingsPerSecond(samples).Should().BeInRange(800, 960, "a 440 Hz tone crosses zero about 880 times a second");
        samples.Max(MathF.Abs).Should().BeGreaterThan(0.05f, "the tone is 1/8 full scale (ffmpeg lowers it 3 dB when it upmixes to stereo)");
    }

    [Theory]
    [InlineData("tone-440hz-1s-live.webm")]
    [InlineData("tone-440hz-1s.ogg")]
    public async Task LoadAudioAsync_BytesWithoutAFileName_MatchTheFileOverload(string fixture)
    {
        var fromFile = await AudioProcessor.LoadAudioAsync(Fixture(fixture), TestContext.Current.CancellationToken);
        var bytes = await File.ReadAllBytesAsync(Fixture(fixture), TestContext.Current.CancellationToken);

        var fromBytes = await AudioProcessor.LoadAudioAsync(bytes, TestContext.Current.CancellationToken);
        await using var stream = new MemoryStream(bytes);
        var fromStream = await AudioProcessor.LoadAudioAsync(stream, TestContext.Current.CancellationToken);

        fromBytes.Should().Equal(fromFile);
        fromStream.Should().Equal(fromFile);
    }

    [Fact]
    public async Task LoadAudioAsync_Mp4Aac_IsRefusedWithWhatIsSupported()
    {
        var bytes = await File.ReadAllBytesAsync(Fixture("tone-440hz-1s.m4a"), TestContext.Current.CancellationToken);

        var act = () => AudioProcessor.LoadAudioAsync(bytes, TestContext.Current.CancellationToken);

        (await act.Should().ThrowAsync<NotSupportedException>()).Which.Message.Should().Contain("WebM").And.Contain("MP4/AAC");
    }

    [Fact]
    public void Detect_RecognisesTheContainersFromTheirFirstBytes()
    {
        OpusAudio.Detect([0x1A, 0x45, 0xDF, 0xA3, 0x01]).Should().Be(OpusAudio.Container.WebM);
        OpusAudio.Detect("OggS\0"u8).Should().Be(OpusAudio.Container.Ogg);
        OpusAudio.Detect([0, 0, 0, 0x20, (byte)'f', (byte)'t', (byte)'y', (byte)'p']).Should().Be(OpusAudio.Container.Mp4);
        OpusAudio.Detect("RIFF\0\0\0\0WAVE"u8).Should().Be(OpusAudio.Container.Unknown);
    }
}
