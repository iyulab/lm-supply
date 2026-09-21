using AwesomeAssertions;
using LMSupply.Reranker.Utils;

namespace LMSupply.Reranker.Tests;

/// <summary>
/// <see cref="LocalReranker.IsModelDownloaded"/> is the probe a consumer uses to keep a download out of a
/// user-facing call. For a GGUF id it has to answer from the same cache layout and the same file choice an
/// offline load opens — a probe that says "no" for a model the loader would open keeps the consumer's gate
/// shut forever, and one that says "yes" for a file the loader would reject opens it onto a failure.
/// </summary>
public sealed class GgufCacheProbeTests : IDisposable
{
    private const string Repo = "example/some-reranker-GGUF";

    private readonly string _cacheDir = Path.Combine(Path.GetTempPath(), "lmsupply-reranker-gguf-probe-" + Guid.NewGuid().ToString("N"));

    public void Dispose()
    {
        if (Directory.Exists(_cacheDir))
            Directory.Delete(_cacheDir, recursive: true);
    }

    private string Place(string fileName, string content = "GGUF-not-a-real-model")
    {
        var dir = Path.Combine(_cacheDir, "gguf-rerankers", Repo.Replace('/', '_'));
        Directory.CreateDirectory(dir);
        var path = Path.Combine(dir, fileName);
        File.WriteAllText(path, content);
        return path;
    }

    [Fact]
    public void ACachedGgufModel_IsReportedAsDownloaded_AndAnUncachedOneIsNot()
    {
        LocalReranker.IsModelDownloaded("gguf:" + Repo, _cacheDir).Should().BeFalse("nothing is cached yet");

        Place("some-reranker-Q4_K_M.gguf");

        LocalReranker.IsModelDownloaded("gguf:" + Repo, _cacheDir).Should().BeTrue();
        LocalReranker.IsModelDownloaded(Repo, _cacheDir).Should().BeTrue("a repository named *-GGUF takes the GGUF route without the prefix, as it does on load");
        LocalReranker.IsModelDownloaded("gguf:example/another-reranker-GGUF", _cacheDir).Should().BeFalse();
    }

    [Fact]
    public async Task TheProbe_AgreesWithTheFileAnOfflineLoadOpens()
    {
        Place("some-reranker-Q8_0.gguf");
        var preferred = Place("some-reranker-Q4_K_M.gguf");

        using var offline = new GgufDownloader(_cacheDir, localFilesOnly: true);
        var opened = await offline.DownloadAsync(Repo, LocalReranker.DefaultGgufQuantization,
            cancellationToken: TestContext.Current.CancellationToken);

        opened.Should().Be(preferred);
        GgufDownloader.TrySelectFromLocalCache(_cacheDir, Repo, LocalReranker.DefaultGgufQuantization).Should().Be(opened);
        LocalReranker.IsModelDownloaded("gguf:" + Repo, _cacheDir).Should().BeTrue();
    }

    [Fact]
    public async Task PreDownload_OfACachedGgufModel_TouchesNothing_AndOpensTheQuantizationAsked()
    {
        Place("some-reranker-Q4_K_M.gguf");
        Place("some-reranker-Q8_0.gguf");

        async Task<string?> FileChosen(string? hint)
        {
            string? seen = null;
            var progress = new SynchronousProgress(p => seen = p.FileName);
            await LocalReranker.DownloadModelAsync(
                "gguf:" + Repo,
                new RerankerOptions { CacheDirectory = _cacheDir, DisableAutoDownload = true, QuantizationHint = hint },
                progress,
                TestContext.Current.CancellationToken);
            return seen;
        }

        (await FileChosen(null)).Should().Be("some-reranker-Q4_K_M.gguf");
        (await FileChosen("Q8_0")).Should().Be("some-reranker-Q8_0.gguf", "a quantization hint is a request, not a comment");

        Directory.EnumerateFiles(_cacheDir, "*", SearchOption.AllDirectories).Should().HaveCount(2,
            "the raw-repository download path would have written a second layout next to the GGUF one");
    }

    [Fact]
    public async Task PreDownload_OfAnUncachedGgufModel_Offline_FailsInsteadOfFetchingTheWholeRepository()
    {
        var download = () => LocalReranker.DownloadModelAsync(
            "gguf:" + Repo,
            new RerankerOptions { CacheDirectory = _cacheDir, DisableAutoDownload = true },
            cancellationToken: TestContext.Current.CancellationToken);

        await download.Should().ThrowAsync<LMSupply.Exceptions.ModelNotFoundException>().WithMessage("*downloads are disabled*");
    }

    private sealed class SynchronousProgress(Action<LMSupply.DownloadProgress> onReport) : IProgress<LMSupply.DownloadProgress>
    {
        public void Report(LMSupply.DownloadProgress value) => onReport(value);
    }

    [Fact]
    public void ALocalGgufFile_IsDownloadedExactlyWhenItExists()
    {
        var path = Place("local.gguf");

        LocalReranker.IsModelDownloaded(path, _cacheDir).Should().BeTrue();
        LocalReranker.IsModelDownloaded("gguf:" + path, _cacheDir).Should().BeTrue();
        LocalReranker.IsModelDownloaded(Path.Combine(_cacheDir, "missing.gguf"), _cacheDir).Should().BeFalse();
    }

    [Fact]
    public async Task AnInterruptedDownloadAndAnLfsPointer_AreNotAModel()
    {
        // What a download that stopped half way leaves behind, and what a git checkout without LFS leaves.
        Place("some-reranker-Q4_K_M.gguf.part");
        Place("some-reranker-Q4_K_M.gguf", "version https://git-lfs.github.com/spec/v1\noid sha256:0000\nsize 459000000\n");

        LocalReranker.IsModelDownloaded("gguf:" + Repo, _cacheDir).Should().BeFalse();

        using var offline = new GgufDownloader(_cacheDir, localFilesOnly: true);
        var load = () => offline.DownloadAsync(Repo, LocalReranker.DefaultGgufQuantization,
            cancellationToken: TestContext.Current.CancellationToken);
        await load.Should().ThrowAsync<LMSupply.Exceptions.ModelNotFoundException>(
            "the loader refuses the same files the probe refuses");
    }
}
