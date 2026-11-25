using AwesomeAssertions;
using LMSupply.Exceptions;
using LMSupply.Transcriber.Vad;

namespace LMSupply.Transcriber.Tests.Vad;

/// <summary>The public voice-activity surface (<see cref="LocalVoiceActivityDetector"/>) in the parts that need no model.</summary>
public sealed class VoiceActivityDetectorSurfaceTests : IDisposable
{
    private readonly string _cache = Path.Combine(Path.GetTempPath(), "lmsupply-vad-surface-" + Guid.NewGuid().ToString("N"));

    public void Dispose()
    {
        if (Directory.Exists(_cache))
            Directory.Delete(_cache, recursive: true);
    }

    [Fact]
    public async Task EmptyCache_IsNotDownloaded_AndALoadWithDownloadsOffFails()
    {
        var ct = TestContext.Current.CancellationToken;
        var options = new VoiceActivityDetectorOptions { CacheDirectory = _cache, DisableAutoDownload = true };

        (await LocalVoiceActivityDetector.IsDownloadedAsync(options, ct)).Should().BeFalse();
        await FluentActions.Awaiting(() => LocalVoiceActivityDetector.LoadAsync(options, cancellationToken: ct))
            .Should().ThrowAsync<ModelNotFoundException>();
        Directory.Exists(_cache).Should().BeFalse("the probe and the refused load write nothing");
    }

    [Fact]
    public async Task DownloadWithDownloadsOff_OnlyConfirmsTheCache()
    {
        var options = new VoiceActivityDetectorOptions { CacheDirectory = _cache, DisableAutoDownload = true };

        await FluentActions.Awaiting(() => LocalVoiceActivityDetector.DownloadAsync(options, cancellationToken: TestContext.Current.CancellationToken))
            .Should().ThrowAsync<ModelNotFoundException>();
    }

    [Fact]
    public void TheModelIsOneSmallMitFile() =>
        LocalVoiceActivityDetector.DownloadSizeBytes.Should().BeInRange(1_000_000, 5_000_000);

    [Fact]
    public void NoOptions_MeanSileroDefaults()
    {
        var settings = VoiceActivityDetectorModel.SettingsOf(null);

        settings.Should().Be(new SegmenterSettings(
            Threshold: 0.5f, MinSpeechSamples: 4000, MinSilenceSamples: 1600, PadSamples: 480, MaxSpeechSamples: int.MaxValue));
    }

    [Fact]
    public void OptionsAreConvertedToSamples()
    {
        var settings = VoiceActivityDetectorModel.SettingsOf(new VadOptions
        {
            Threshold = 0.3f,
            MinSpeechDuration = TimeSpan.FromSeconds(1),
            MinSilenceDuration = TimeSpan.FromMilliseconds(500),
            SpeechPadding = TimeSpan.Zero,
            MaxSpeechDuration = TimeSpan.FromSeconds(30),
        });

        settings.Should().Be(new SegmenterSettings(0.3f, 16000, 8000, 0, 480000));
    }

    [Theory]
    [InlineData(0f, 0, 0, 0, 0, "VadOptions.Threshold")]
    [InlineData(1f, 0, 0, 0, 0, "VadOptions.Threshold")]
    [InlineData(float.NaN, 0, 0, 0, 0, "VadOptions.Threshold")]
    [InlineData(0.5f, -1, 0, 0, 0, "VadOptions.MinSpeechDuration")]
    [InlineData(0.5f, 0, -1, 0, 0, "VadOptions.MinSilenceDuration")]
    [InlineData(0.5f, 0, 0, -1, 0, "VadOptions.SpeechPadding")]
    [InlineData(0.5f, 250, 0, 0, 250, "VadOptions.MaxSpeechDuration")]
    [InlineData(0.5f, 0, 0, 0, -5, "VadOptions.MaxSpeechDuration")]
    public void OutOfRangeSettings_AreRefusedByName(float threshold, int minSpeechMs, int minSilenceMs, int paddingMs, int maxSpeechMs, string name)
    {
        var options = new VadOptions
        {
            Threshold = threshold,
            MinSpeechDuration = TimeSpan.FromMilliseconds(minSpeechMs),
            MinSilenceDuration = TimeSpan.FromMilliseconds(minSilenceMs),
            SpeechPadding = TimeSpan.FromMilliseconds(paddingMs),
            // 0 means «not given» in this table, to keep the rows short.
            MaxSpeechDuration = maxSpeechMs == 0 ? null : TimeSpan.FromMilliseconds(maxSpeechMs),
        };

        var act = () => VoiceActivityDetectorModel.SettingsOf(options);

        act.Should().Throw<ArgumentOutOfRangeException>().WithMessage($"*{name}*");
    }

    [Fact]
    public void ASegmentKnowsItsLength_AndAResultItsSpeechTotal()
    {
        new SpeechSegment(1.5, 4.0).Duration.Should().Be(2.5);

        var result = new VadResult
        {
            Segments = [new SpeechSegment(0, 1), new SpeechSegment(3, 5.5)],
            DurationSeconds = 10,
            InferenceTimeMs = 1,
        };
        result.SpeechSeconds.Should().Be(3.5);
    }

    [Fact]
    public void ACloneCarriesEverySetting_AndIsIndependent()
    {
        var original = new VoiceActivityDetectorOptions
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
