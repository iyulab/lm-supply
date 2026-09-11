using System.Net;
using LMSupply.Download;
using LMSupply.Exceptions;

namespace LMSupply.Core.Tests.Download;

/// <summary>
/// <see cref="CacheManager.GetMissingFiles"/> must answer the question a download answers by acting: which
/// files would be fetched. The agreement tests hold it against a local-files-only downloader on the same
/// cache — "nothing missing" must mean the load succeeds without a request, and "missing" must mean it fails.
/// </summary>
public sealed class CacheManagerMissingFilesTests : IDisposable
{
    private const string Repo = "acme/split-model";
    private const string Subfolder = "languages/korean";
    private readonly string _cacheDir = Path.Combine(Path.GetTempPath(), "lmsupply-missing-" + Guid.NewGuid().ToString("N"));

    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    public void Dispose()
    {
        if (Directory.Exists(_cacheDir))
            Directory.Delete(_cacheDir, recursive: true);
    }

    [Fact]
    public void NothingCached_ReportsEveryFile_InOrder()
    {
        var missing = CacheManager.GetMissingFiles(_cacheDir, Repo, ["rec.onnx", "dict.txt"], Subfolder);

        Assert.Equal(["rec.onnx", "dict.txt"], missing);
    }

    [Fact]
    public async Task AnLfsPointer_CountsAsMissing()
    {
        var dir = SubfolderDir();
        Directory.CreateDirectory(dir);
        await File.WriteAllTextAsync(Path.Combine(dir, "rec.onnx"), "version https://git-lfs.github.com/spec/v1\noid sha256:0\nsize 1\n", Ct);
        await File.WriteAllTextAsync(Path.Combine(dir, "dict.txt"), "a\nb\n", Ct);

        var missing = CacheManager.GetMissingFiles(_cacheDir, Repo, ["rec.onnx", "dict.txt"], Subfolder);

        Assert.Equal(["rec.onnx"], missing);
    }

    // Another subfolder's files with the same names must not count — the reason subfolders exist.
    [Fact]
    public async Task TheSameFileNameInAnotherSubfolder_DoesNotCount()
    {
        var english = Path.Combine(CacheManager.GetModelDirectory(_cacheDir, Repo), "languages", "english");
        Directory.CreateDirectory(english);
        await File.WriteAllTextAsync(Path.Combine(english, "rec.onnx"), "graph", Ct);

        var missing = CacheManager.GetMissingFiles(_cacheDir, Repo, ["rec.onnx"], Subfolder);

        Assert.Equal(["rec.onnx"], missing);
    }

    [Fact]
    public async Task NothingMissing_MeansAnOfflineLoadSucceeds_WithoutARequest()
    {
        var dir = SubfolderDir();
        Directory.CreateDirectory(dir);
        await File.WriteAllTextAsync(Path.Combine(dir, "rec.onnx"), "graph", Ct);
        await File.WriteAllTextAsync(Path.Combine(dir, "dict.txt"), "a\nb\n", Ct);

        Assert.Empty(CacheManager.GetMissingFiles(_cacheDir, Repo, ["rec.onnx", "dict.txt"], Subfolder));

        var hub = new CountingHub();
        using var downloader = new HuggingFaceDownloader(_cacheDir, hub, localFilesOnly: true);
        var result = await downloader.DownloadModelAsync(Repo, ["rec.onnx", "dict.txt"], subfolder: Subfolder, cancellationToken: Ct);

        Assert.Equal(Path.GetFullPath(dir), Path.GetFullPath(result));
        Assert.Equal(0, hub.Count);
    }

    [Fact]
    public async Task SomethingMissing_MeansAnOfflineLoadFails()
    {
        Assert.NotEmpty(CacheManager.GetMissingFiles(_cacheDir, Repo, ["rec.onnx"], Subfolder));

        var hub = new CountingHub();
        using var downloader = new HuggingFaceDownloader(_cacheDir, hub, localFilesOnly: true);

        await Assert.ThrowsAsync<ModelNotFoundException>(() =>
            downloader.DownloadModelAsync(Repo, ["rec.onnx"], subfolder: Subfolder, cancellationToken: Ct));
        Assert.Equal(0, hub.Count);
    }

    [Fact]
    public void ASubfolderOutsideTheSnapshot_IsRefused()
    {
        Assert.Throws<InvalidOperationException>(() =>
            CacheManager.GetMissingFiles(_cacheDir, Repo, ["rec.onnx"], "../../elsewhere"));
    }

    private string SubfolderDir()
        => Path.Combine(CacheManager.GetModelDirectory(_cacheDir, Repo), Subfolder.Replace('/', Path.DirectorySeparatorChar));

    private sealed class CountingHub : HttpMessageHandler
    {
        private int _count;

        public int Count => Volatile.Read(ref _count);

        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            Interlocked.Increment(ref _count);
            return Task.FromResult(new HttpResponseMessage(HttpStatusCode.NotFound));
        }
    }
}
