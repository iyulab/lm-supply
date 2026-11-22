using AwesomeAssertions;
using LMSupply.Transcriber.Audio;

namespace LMSupply.Transcriber.Tests;

/// <summary>
/// Audio the library cannot use — bytes that are no audio format it reads, or a recording whose header is there and whose samples are
/// not — is refused with the one exception type the caller can recognise as «this input is the problem» (<see cref="InvalidDataException"/>,
/// as for empty audio), not with a decoder's own type, and not by transcribing silence into a made-up sentence.
/// </summary>
public sealed class UnreadableAudioTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "lmsupply-unreadable-audio-" + Guid.NewGuid().ToString("N"));

    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    public UnreadableAudioTests() => Directory.CreateDirectory(_dir);

    public void Dispose() => Directory.Delete(_dir, recursive: true);

    /// <summary>A valid PCM WAV header (mono, 16 kHz, 16 bit) followed by <paramref name="dataBytes"/> bytes of silence.</summary>
    private static byte[] Wav(int dataBytes)
    {
        using var stream = new MemoryStream();
        using var writer = new BinaryWriter(stream);
        writer.Write("RIFF"u8);
        writer.Write(36 + dataBytes);
        writer.Write("WAVEfmt "u8);
        writer.Write(16);
        writer.Write((short)1);
        writer.Write((short)1);
        writer.Write(16000);
        writer.Write(32000);
        writer.Write((short)2);
        writer.Write((short)16);
        writer.Write("data"u8);
        writer.Write(dataBytes);
        writer.Write(new byte[dataBytes]);
        return stream.ToArray();
    }

    [Fact]
    public async Task Bytes_ThatAreNoAudioFormat_AreRefusedAsInvalidData()
    {
        var act = () => AudioProcessor.LoadAudioAsync(new byte[44], Ct);

        (await act.Should().ThrowAsync<InvalidDataException>()).Which.Message.Should().Contain("not in a format");
    }

    [Fact]
    public async Task AWavWithoutSamples_IsRefusedAsEmpty()
    {
        var act = () => AudioProcessor.LoadAudioAsync(Wav(0), Ct);

        (await act.Should().ThrowAsync<InvalidDataException>()).Which.Message.Should().Contain("no samples");
    }

    [Fact]
    public async Task AWavFileWithoutSamples_NamesTheFile()
    {
        var path = Path.Combine(_dir, "header-only.wav");
        await File.WriteAllBytesAsync(path, Wav(0), Ct);

        var act = () => AudioProcessor.LoadAudioAsync(path, Ct);

        (await act.Should().ThrowAsync<InvalidDataException>()).Which.Message.Should().Contain("header-only.wav").And.Contain("no samples");
    }

    [Fact]
    public async Task AWavWithSamples_IsStillRead()
    {
        var samples = await AudioProcessor.LoadAudioAsync(Wav(32000), Ct);

        samples.Should().HaveCount(16000);
    }
}
