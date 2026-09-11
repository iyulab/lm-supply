using System.Net;
using System.Text;
using LMSupply.Download;
using LMSupply.Exceptions;

namespace LMSupply.Core.Tests.Download;

/// <summary>
/// A downloader created with <c>localFilesOnly</c> serves what is cached and makes no request for what
/// is not. Every test counts the requests the downloader causes — discovery included, since the test seam
/// hands discovery the same transport — because "throws ModelNotFoundException" alone would also pass if
/// the downloader had asked the hub and been told the repository does not exist.
/// </summary>
public sealed class HuggingFaceDownloaderOfflineTests : IDisposable
{
    private const string Repo = "acme/offline-model";
    private readonly string _cacheDir = Path.Combine(Path.GetTempPath(), "lmsupply-offline-" + Guid.NewGuid().ToString("N"));

    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    public void Dispose()
    {
        if (Directory.Exists(_cacheDir))
            Directory.Delete(_cacheDir, recursive: true);
    }

    [Fact]
    public async Task AFileNotInTheCache_ThrowsWithoutARequest()
    {
        var hub = new CountingHub();
        using var downloader = new HuggingFaceDownloader(_cacheDir, hub, localFilesOnly: true);

        await Assert.ThrowsAsync<ModelNotFoundException>(() => downloader.DownloadModelAsync(Repo, ["model.onnx"], cancellationToken: Ct));

        Assert.Equal(0, hub.Count);
    }

    [Fact]
    public async Task CachedFiles_AreReturnedWithoutARequest_AndAMissingOptionalFileIsSkipped()
    {
        var dir = CacheManager.GetModelDirectory(_cacheDir, Repo);
        Directory.CreateDirectory(dir);
        await File.WriteAllTextAsync(Path.Combine(dir, "model.onnx"), "graph", Ct);
        var hub = new CountingHub();
        using var downloader = new HuggingFaceDownloader(_cacheDir, hub, localFilesOnly: true);

        var result = await downloader.DownloadModelAsync(Repo, ["model.onnx", "tokenizer.json"], cancellationToken: Ct);

        Assert.Equal(dir, result);
        Assert.Equal(0, hub.Count);
    }

    [Fact]
    public async Task Discovery_WithNothingCached_ThrowsWithoutARequest()
    {
        var hub = new CountingHub();
        using var downloader = new HuggingFaceDownloader(_cacheDir, hub, localFilesOnly: true);

        await Assert.ThrowsAsync<ModelNotFoundException>(() => downloader.DownloadWithDiscoveryAsync(Repo, cancellationToken: Ct));

        Assert.Equal(0, hub.Count);
    }

    // The cached file list expires after a day so that an online run picks up repository changes. Offline
    // that expiry used to send discovery to the hub for a model whose every file was on disk.
    [Fact]
    public async Task Discovery_AfterAnOnlineRun_WorksOffline_OnceTheFileListIsOlderThanADay()
    {
        await DownloadOnlineAsync();
        foreach (var file in Directory.GetFiles(Path.Combine(_cacheDir, ".discovery-cache")))
            File.SetLastWriteTimeUtc(file, DateTime.UtcNow.AddDays(-2));
        // Only the stale list is left to go on — the manifest would otherwise answer instead.
        File.Delete(Path.Combine(CacheManager.GetModelDirectory(_cacheDir, Repo), ".lmsupply-manifest.json"));

        var hub = new CountingHub();
        using var downloader = new HuggingFaceDownloader(_cacheDir, hub, localFilesOnly: true);
        var (dir, discovery) = await downloader.DownloadWithDiscoveryAsync(Repo, cancellationToken: Ct);

        Assert.True(File.Exists(Path.Combine(dir, "model.onnx")));
        Assert.Contains("model.onnx", discovery.OnnxFiles);
        Assert.Equal(0, hub.Count);
    }

    [Fact]
    public async Task Discovery_WithOnlyTheDownloadManifest_WorksOffline()
    {
        await DownloadOnlineAsync();
        Directory.Delete(Path.Combine(_cacheDir, ".discovery-cache"), recursive: true);

        var hub = new CountingHub();
        using var downloader = new HuggingFaceDownloader(_cacheDir, hub, localFilesOnly: true);
        var (dir, _) = await downloader.DownloadWithDiscoveryAsync(Repo, cancellationToken: Ct);

        Assert.True(File.Exists(Path.Combine(dir, "model.onnx")));
        Assert.Equal(0, hub.Count);
    }

    [Fact]
    public async Task DownloadingASingleFile_IsRefused()
    {
        var hub = new CountingHub();
        using var downloader = new HuggingFaceDownloader(_cacheDir, hub, localFilesOnly: true);

        await Assert.ThrowsAsync<ModelNotFoundException>(() =>
            downloader.DownloadFileAsync(Repo, "model.onnx", Path.Combine(_cacheDir, "model.onnx"), cancellationToken: Ct));

        Assert.Equal(0, hub.Count);
    }

    // Populates the cache the ordinary way, and doubles as the positive control for the counter: the
    // online run's discovery request must be seen, or a zero above would prove nothing.
    private async Task DownloadOnlineAsync()
    {
        var hub = new CountingHub(servesModel: true);
        using var downloader = new HuggingFaceDownloader(_cacheDir, hub);

        await downloader.DownloadWithDiscoveryAsync(Repo, cancellationToken: Ct);

        Assert.Contains(hub.Paths, p => p.StartsWith("/api/models/", StringComparison.Ordinal));
        Assert.Contains(hub.Paths, p => p.EndsWith("/resolve/main/model.onnx", StringComparison.Ordinal));
    }

    /// <summary>Counts every request; optionally serves a one-file repository (tree listing and file).</summary>
    private sealed class CountingHub(bool servesModel = false) : HttpMessageHandler
    {
        private readonly List<string> _paths = [];

        public int Count
        {
            get
            {
                lock (_paths)
                    return _paths.Count;
            }
        }

        public IReadOnlyList<string> Paths
        {
            get
            {
                lock (_paths)
                    return [.. _paths];
            }
        }

        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            var path = Uri.UnescapeDataString(request.RequestUri!.AbsolutePath);
            lock (_paths)
                _paths.Add(path);

            if (servesModel && path == $"/api/models/{Repo}/tree/main")
                return Ok("""[{"path":"model.onnx","type":"file","size":5}]""");
            if (servesModel && path == $"/{Repo}/resolve/main/model.onnx")
                return Ok("graph");

            return Task.FromResult(new HttpResponseMessage(HttpStatusCode.NotFound));
        }

        private static Task<HttpResponseMessage> Ok(string body)
            => Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK) { Content = new ByteArrayContent(Encoding.UTF8.GetBytes(body)) });
    }
}
