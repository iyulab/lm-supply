using AwesomeAssertions;
using LMSupply.Transcriber.Audio;

namespace LMSupply.Transcriber.Tests;

/// <summary>
/// Audio that holds no bytes — a recording that never started, a file cut to nothing — says so, instead of failing inside the
/// decoder with an end-of-stream error that does not.
/// </summary>
public sealed class EmptyAudioTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "lmsupply-empty-audio-" + Guid.NewGuid().ToString("N"));

    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    public EmptyAudioTests() => Directory.CreateDirectory(_dir);

    public void Dispose() => Directory.Delete(_dir, recursive: true);

    [Fact]
    public async Task AnEmptyByteArray_IsSaidToBeEmpty()
    {
        var act = () => AudioProcessor.LoadAudioAsync(Array.Empty<byte>(), Ct);

        (await act.Should().ThrowAsync<InvalidDataException>()).Which.Message.Should().Contain("empty").And.Contain("0 bytes");
    }

    [Fact]
    public async Task AnEmptyStream_IsSaidToBeEmpty()
    {
        var act = () => AudioProcessor.LoadAudioAsync(new MemoryStream(), Ct);

        (await act.Should().ThrowAsync<InvalidDataException>()).Which.Message.Should().Contain("empty");
    }

    [Theory]
    [InlineData("silence.wav")]
    [InlineData("silence.mp3")]
    public async Task AnEmptyFile_NamesTheFile(string name)
    {
        var path = Path.Combine(_dir, name);
        await File.WriteAllBytesAsync(path, [], Ct);

        var act = () => AudioProcessor.LoadAudioAsync(path, Ct);

        (await act.Should().ThrowAsync<InvalidDataException>()).Which.Message.Should().Contain(name).And.Contain("0 bytes");
    }

    [Fact]
    public async Task AMissingFile_StillFailsAsAMissingFile()
    {
        var act = () => AudioProcessor.LoadAudioAsync(Path.Combine(_dir, "nothing.wav"), Ct);

        await act.Should().ThrowAsync<FileNotFoundException>();
    }
}
