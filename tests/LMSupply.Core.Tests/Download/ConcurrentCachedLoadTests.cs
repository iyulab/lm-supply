using LMSupply.Core.Download;
using System.Net;
using System.Text;
using LMSupply.Download;

namespace LMSupply.Core.Tests.Download;

/// <summary>
/// Several loads of one cached model at once, with downloads allowed — a host resolving the same model from two
/// services at startup, or a warmup beside a first request. Every load used to rewrite the snapshot's manifest, and
/// the writers (and a reader beside them) collided on the file: «being used by another process».
/// </summary>
public sealed class ConcurrentCachedLoadTests : IDisposable
{
    private const string Repo = "acme/concurrent-model";
    private const int Loads = 16;
    private readonly string _cacheDir = Path.Combine(Path.GetTempPath(), "lmsupply-concurrent-" + Guid.NewGuid().ToString("N"));

    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    public void Dispose()
    {
        if (Directory.Exists(_cacheDir))
            Directory.Delete(_cacheDir, recursive: true);
    }

    [Fact]
    public async Task ParallelLoadsOfACachedModel_AllSucceed_AndLeaveTheManifestAsItWas()
    {
        var dir = await DownloadOnceAsync();
        var manifestPath = Path.Combine(dir, DownloadManifest.LegacyFileName);
        var before = await File.ReadAllTextAsync(manifestPath, Ct);
        var writtenAt = File.GetLastWriteTimeUtc(manifestPath);

        for (var round = 0; round < 3; round++)
        {
            var loads = Enumerable.Range(0, Loads).Select(_ => Task.Run(async () =>
            {
                using var downloader = new HuggingFaceDownloader(_cacheDir, new Hub());
                var (loaded, discovery) = await downloader.DownloadWithDiscoveryAsync(Repo, cancellationToken: Ct);
                return (loaded, discovery);
            }, Ct));

            foreach (var (loaded, discovery) in await Task.WhenAll(loads))
            {
                Assert.Equal(dir, loaded);
                Assert.Contains("model.onnx", discovery.OnnxFiles);
            }
        }

        // A load that finds every file in place has nothing to record: the manifest is not rewritten.
        Assert.Equal(before, await File.ReadAllTextAsync(manifestPath, Ct));
        Assert.Equal(writtenAt, File.GetLastWriteTimeUtc(manifestPath));
    }

    [Fact]
    public async Task ConcurrentManifestWrites_BesideReaders_NeverFail_AndLeaveAWholeManifest()
    {
        var dir = Path.Combine(_cacheDir, "plain");
        Directory.CreateDirectory(dir);

        var writers = Enumerable.Range(0, Loads).Select(i => Task.Run(() => DownloadManifest.WriteAsync(dir, new DownloadManifest
        {
            Version = DownloadManifest.VerifiedVersion,
            RepoId = Repo,
            Revision = "main",
            Files = [new ManifestFileEntry { Path = $"file-{i}.onnx", Size = i + 1 }]
        }, Ct), Ct));
        var readers = Enumerable.Range(0, Loads).Select(_ => Task.Run(async () =>
        {
            for (var n = 0; n < 20; n++)
                await DownloadManifest.ReadAsync(dir, Ct);
        }, Ct));

        await Task.WhenAll(writers.Concat(readers));

        var manifest = await DownloadManifest.ReadAsync(dir, Ct);
        Assert.NotNull(manifest);
        Assert.Single(manifest.Files);
        // No temporary file is left beside the manifest.
        Assert.Equal([DownloadManifest.LegacyFileName], Directory.GetFiles(dir).Select(Path.GetFileName));
    }

    // Populates the cache in the layout of the report (snapshots/main, manifest inside the directory).
    private async Task<string> DownloadOnceAsync()
    {
        using var downloader = new HuggingFaceDownloader(_cacheDir, new Hub());
        var (dir, _) = await downloader.DownloadWithDiscoveryAsync(Repo, cancellationToken: Ct);
        Assert.True(File.Exists(Path.Combine(dir, DownloadManifest.LegacyFileName)), "the first download writes the manifest");
        return dir;
    }

    /// <summary>Serves a one-file repository: the tree listing and the file; everything else is a 404.</summary>
    private sealed class Hub : HttpMessageHandler
    {
        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            var path = Uri.UnescapeDataString(request.RequestUri!.AbsolutePath);
            if (path == $"/api/models/{Repo}/tree/main")
                return Ok("""[{"path":"model.onnx","type":"file","size":5}]""");
            if (path == $"/{Repo}/resolve/main/model.onnx")
                return Ok("graph");

            return Task.FromResult(new HttpResponseMessage(HttpStatusCode.NotFound));
        }

        private static Task<HttpResponseMessage> Ok(string body)
            => Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK) { Content = new ByteArrayContent(Encoding.UTF8.GetBytes(body)) });
    }
}
