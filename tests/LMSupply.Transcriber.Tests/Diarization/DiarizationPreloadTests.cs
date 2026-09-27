using AwesomeAssertions;
using LMSupply.Download;
using LMSupply.Exceptions;
using LMSupply.Transcriber.Diarization;

namespace LMSupply.Transcriber.Tests.Diarization;

/// <summary>
/// A consumer whose downloads happen only at an install step asks, without a request, whether the diarization pair is
/// cached. The probe follows the downloader's local-only rule, so it answers for the exact files a diarized call opens.
/// </summary>
public sealed class DiarizationPreloadTests : IDisposable
{
    private readonly string _cache = Path.Combine(Path.GetTempPath(), "lmsupply-diar-probe-" + Guid.NewGuid().ToString("N"));

    public void Dispose()
    {
        if (Directory.Exists(_cache))
            Directory.Delete(_cache, recursive: true);
    }

    private TranscriberOptions Options => new() { CacheDirectory = _cache };

    private void Place(string repo, string file, string content = "onnx bytes")
    {
        var dir = CacheManager.GetModelDirectory(_cache, repo);
        Directory.CreateDirectory(dir);
        File.WriteAllText(Path.Combine(dir, file), content);
    }

    [Fact]
    public async Task EmptyCache_IsNotDownloaded()
    {
        (await LocalTranscriber.IsDiarizationDownloadedAsync(Options, TestContext.Current.CancellationToken))
            .Should().BeFalse();
    }

    [Fact]
    public async Task BothFilesCached_IsDownloaded()
    {
        Place(SpeakerDiarizer.SegmentationRepo, SpeakerDiarizer.SegmentationFile);
        Place(SpeakerDiarizer.EmbeddingRepo, SpeakerDiarizer.EmbeddingFile);

        (await LocalTranscriber.IsDiarizationDownloadedAsync(Options, TestContext.Current.CancellationToken))
            .Should().BeTrue();
    }

    [Fact]
    public async Task OneFileMissing_IsNotDownloaded()
    {
        Place(SpeakerDiarizer.SegmentationRepo, SpeakerDiarizer.SegmentationFile);

        (await LocalTranscriber.IsDiarizationDownloadedAsync(Options, TestContext.Current.CancellationToken))
            .Should().BeFalse();
    }

    [Fact]
    public async Task LfsPointerInPlaceOfAModel_IsNotDownloaded()
    {
        Place(SpeakerDiarizer.SegmentationRepo, SpeakerDiarizer.SegmentationFile);
        Place(SpeakerDiarizer.EmbeddingRepo, SpeakerDiarizer.EmbeddingFile,
            "version https://git-lfs.github.com/spec/v1\noid sha256:0\nsize 26530550\n");

        (await LocalTranscriber.IsDiarizationDownloadedAsync(Options, TestContext.Current.CancellationToken))
            .Should().BeFalse();
    }

    [Fact]
    public async Task Probe_NeverWritesTheCache()
    {
        await LocalTranscriber.IsDiarizationDownloadedAsync(Options, TestContext.Current.CancellationToken);

        Directory.Exists(_cache).Should().BeFalse("a probe reads the cache and must not create or fill it");
    }

    [Fact]
    public void DownloadSize_IsThePairAsTheRepositoriesListIt()
    {
        LocalTranscriber.DiarizationDownloadSizeBytes.Should().Be(5_992_913 + 26_530_550);
    }

    [Fact]
    public void Clone_CarriesPreloadDiarization()
    {
        new TranscriberOptions { PreloadDiarization = true }.Clone().PreloadDiarization.Should().BeTrue();
    }
}

/// <summary>
/// The install step end to end: one load fetches the transcription model and the diarization pair through the same
/// progress sink, after which the probe says the pair is cached; with downloads disabled a missing pair fails the load
/// the way a missing transcription model does.
/// </summary>
[Trait("Category", "Integration")]
public sealed class DiarizationPreloadIntegrationTests : IDisposable
{
    private readonly string _cache = Path.Combine(Path.GetTempPath(), "lmsupply-diar-preload-" + Guid.NewGuid().ToString("N"));

    public void Dispose()
    {
        if (Directory.Exists(_cache))
            Directory.Delete(_cache, recursive: true);
    }

    private sealed class Sink : IProgress<DownloadProgress>
    {
        public List<DownloadProgress> Reports { get; } = [];
        public void Report(DownloadProgress value)
        {
            lock (Reports)
                Reports.Add(value);
        }
    }

    [Fact]
    public async Task PreloadFetchesTranscriberAndPairThroughTheLoadsProgress()
    {
        var ct = TestContext.Current.CancellationToken;
        var sink = new Sink();

        await using (await LocalTranscriber.LoadAsync(
            "fast",
            new TranscriberOptions { CacheDirectory = _cache, Provider = ExecutionProvider.Cpu, PreloadDiarization = true },
            sink,
            ct))
        {
        }

        var files = sink.Reports.Select(r => Path.GetFileName(r.FileName)).Distinct().ToList();
        files.Should().Contain(SpeakerDiarizer.SegmentationFile);
        files.Should().Contain(SpeakerDiarizer.EmbeddingFile);
        files.Should().Contain(f => f.StartsWith("encoder", StringComparison.OrdinalIgnoreCase),
            "the transcription model's own download reports through the load's progress");
        (await LocalTranscriber.IsDiarizationDownloadedAsync(new TranscriberOptions { CacheDirectory = _cache }, ct))
            .Should().BeTrue();

        // Downloads disabled, pair gone: the load fails like a missing transcription model, not mid-transcription.
        File.Delete(Path.Combine(
            CacheManager.GetModelDirectory(_cache, SpeakerDiarizer.EmbeddingRepo), SpeakerDiarizer.EmbeddingFile));
        var offline = new TranscriberOptions
        {
            CacheDirectory = _cache, Provider = ExecutionProvider.Cpu, PreloadDiarization = true, DisableAutoDownload = true,
        };
        var load = () => LocalTranscriber.LoadAsync("fast", offline, null, ct);
        await load.Should().ThrowAsync<ModelNotFoundException>();

        // Without preload the same offline load succeeds — the pair is only needed by a diarized call.
        offline.PreloadDiarization = false;
        await using (await LocalTranscriber.LoadAsync("fast", offline, null, ct))
        {
        }
    }
}
