using AwesomeAssertions;
using LMSupply.Exceptions;
using LMSupply.Transcriber.Diarization;

namespace LMSupply.Transcriber.Tests.Diarization;

/// <summary>The public diarization surface (<see cref="LocalSpeakerDiarizer"/>) in the parts that need no model.</summary>
public sealed class SpeakerDiarizerSurfaceTests : IDisposable
{
    private readonly string _cache = Path.Combine(Path.GetTempPath(), "lmsupply-diar-surface-" + Guid.NewGuid().ToString("N"));

    public void Dispose()
    {
        if (Directory.Exists(_cache))
            Directory.Delete(_cache, recursive: true);
    }

    [Fact]
    public async Task EmptyCache_IsNotDownloaded_AndALoadWithDownloadsOffFails()
    {
        var ct = TestContext.Current.CancellationToken;
        var options = new SpeakerDiarizerOptions { CacheDirectory = _cache, DisableAutoDownload = true };

        (await LocalSpeakerDiarizer.IsDownloadedAsync(options, ct)).Should().BeFalse();
        await FluentActions.Awaiting(() => LocalSpeakerDiarizer.LoadAsync(options, cancellationToken: ct))
            .Should().ThrowAsync<ModelNotFoundException>();
        Directory.Exists(_cache).Should().BeFalse("the probe and the refused load write nothing");
    }

    [Fact]
    public async Task DownloadWithDownloadsOff_OnlyConfirmsTheCache()
    {
        var options = new SpeakerDiarizerOptions { CacheDirectory = _cache, DisableAutoDownload = true };

        await FluentActions.Awaiting(() => LocalSpeakerDiarizer.DownloadAsync(options, cancellationToken: TestContext.Current.CancellationToken))
            .Should().ThrowAsync<ModelNotFoundException>();
    }

    [Fact]
    public void TheSizeIsTheOneTheTranscriberReports() =>
        LocalSpeakerDiarizer.DownloadSizeBytes.Should().Be(LocalTranscriber.DiarizationDownloadSizeBytes);

    [Theory]
    [InlineData(0, null, null, null, "DiarizeOptions.NumSpeakers")]
    [InlineData(null, 0, null, null, "DiarizeOptions.MinSpeakers")]
    [InlineData(null, null, -1, null, "DiarizeOptions.MaxSpeakers")]
    [InlineData(null, 3, 2, null, "MinSpeakers (3) must not exceed MaxSpeakers (2)")]
    [InlineData(null, null, null, 0f, "DiarizeOptions.SpeakerThreshold")]
    [InlineData(null, null, null, 2f, "DiarizeOptions.SpeakerThreshold")]
    [InlineData(null, null, null, float.NaN, "DiarizeOptions.SpeakerThreshold")]
    public void OutOfRangeSettings_AreRefusedByName(int? num, int? min, int? max, float? threshold, string message)
    {
        var act = () => DiarizationStage.ValidateBounds(
            nameof(DiarizeOptions), num, min, max, threshold, "options");

        act.Should().Throw<ArgumentOutOfRangeException>().WithMessage($"*{message}*");
    }

    [Fact]
    public void ValidSettings_PassAndTheTranscriberKeepsItsOwnWording()
    {
        DiarizationStage.ValidateBounds(nameof(DiarizeOptions), 2, 1, 3, 0.4f, "options");

        var act = () => DiarizationStage.Validate(new TranscribeOptions { NumSpeakers = 0 });
        act.Should().Throw<ArgumentOutOfRangeException>().WithMessage("*TranscribeOptions.NumSpeakers must be positive*");
    }

    [Fact]
    public void ALabelNamesTheVoiceInSpeakingOrder()
    {
        SpeakerTurn.LabelOf(0).Should().Be("S1");
        SpeakerTurn.LabelOf(3).Should().Be("S4");
        new SpeakerTurn(1.5, 4.0, "S2").Duration.Should().Be(2.5);
    }

    [Fact]
    public void ACloneCarriesEverySetting_AndIsIndependent()
    {
        var original = new SpeakerDiarizerOptions
        {
            CacheDirectory = _cache, Provider = ExecutionProvider.Cpu, LogLevel = LMSupply.OrtLogLevel.Warning,
            DisableAutoDownload = true, ThreadCount = 3, QuantizationHint = "fp16",
        };

        var copy = original.Clone();
        copy.Should().BeEquivalentTo(original);
        copy.DisableAutoDownload = false;
        original.DisableAutoDownload.Should().BeTrue();
    }
}
