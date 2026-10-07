using LMSupply.Core.Download;
using System.Net;
using System.Text;
using LMSupply.Download;
using LMSupply.Exceptions;

namespace LMSupply.Core.Tests.Download;

/// <summary>
/// A model downloaded once loads again without a network, with downloads still enabled — the cache is complete, so
/// it is used. The default file list names optional files most repositories do not have (external weight
/// companions, tokenizer assets of other families); the online load records which ones the repository lacks, so a
/// warm load makes no request for them. When the network fails anyway (an older manifest without that record, or a
/// file list older than a day) an optional file is skipped and a cached model still loads; a required file the cache
/// lacks fails with a message that names the next step.
/// </summary>
public sealed class HuggingFaceDownloaderNetworkFailureTests : IDisposable
{
    private const string Repo = "acme/net-failure";
    private const string Commit = "0123456789abcdef0123456789abcdef01234567";
    private const string Subfolder = "onnx";

    private static readonly Dictionary<string, string> s_files = new(StringComparer.Ordinal)
    {
        ["onnx/model.onnx"] = "graph-bytes",
        ["config.json"] = "{\"hidden_size\":4}",
        ["tokenizer.json"] = "{\"model\":{}}",
        ["tokenizer_config.json"] = "{}",
    };

    private readonly string _cacheDir = Path.Combine(Path.GetTempPath(), "lmsupply-netfail-" + Guid.NewGuid().ToString("N"));

    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    public void Dispose()
    {
        if (Directory.Exists(_cacheDir))
            Directory.Delete(_cacheDir, recursive: true);
    }

    [Fact]
    public async Task AWarmLoad_WithTheDefaultFileList_MakesNoRequest_AndLoadsWhileTheNetworkIsDown()
    {
        var onlineDir = await LoadOnlineAsync();

        var hub = new Hub { Offline = true };
        using var downloader = new HuggingFaceDownloader(_cacheDir, hub);
        var dir = await downloader.DownloadModelAsync(Repo, subfolder: Subfolder, cancellationToken: Ct);

        Assert.Equal(onlineDir, dir);
        Assert.Equal("graph-bytes", await File.ReadAllTextAsync(Path.Combine(dir, "model.onnx"), Ct));
        Assert.Empty(hub.Requests);
    }

    [Fact]
    public async Task AWarmLoad_Online_DoesNotAskForFilesTheRepositoryLacks()
    {
        await LoadOnlineAsync();

        // Online, with the file list expired: the listing is fetched again, but the optional files it lacks are not
        // requested one by one.
        AgeFileList();
        var hub = new Hub();
        using var downloader = new HuggingFaceDownloader(_cacheDir, hub);
        await downloader.DownloadModelAsync(Repo, subfolder: Subfolder, cancellationToken: Ct);

        Assert.DoesNotContain(hub.Requests, p => p.Contains("/resolve/", StringComparison.Ordinal));
    }

    [Fact]
    public async Task AManifestWithoutTheAbsenceRecord_StillLoadsWhileTheNetworkIsDown()
    {
        var onlineDir = await LoadOnlineAsync();

        // A manifest an earlier version wrote: sizes only, and no file list to fall back on.
        var manifest = await DownloadManifest.ReadAsync(onlineDir, Ct);
        Assert.NotNull(manifest);
        Assert.NotEmpty(manifest.AbsentFiles ?? []);
        manifest.AbsentFiles = null;
        await DownloadManifest.WriteForSnapshotAsync(RepoDir, Path.GetDirectoryName(onlineDir)!, Subfolder, manifest);
        File.Delete(ModelDiscoveryService.GetListingPath(_cacheDir, Repo, "main"));

        var hub = new Hub { Offline = true };
        using var downloader = new HuggingFaceDownloader(_cacheDir, hub);
        var dir = await downloader.DownloadModelAsync(Repo, subfolder: Subfolder, cancellationToken: Ct);

        Assert.Equal("graph-bytes", await File.ReadAllTextAsync(Path.Combine(dir, "model.onnx"), Ct));
        // The positive control for the zero above: this load did go to the network, and its failure was absorbed.
        Assert.NotEmpty(hub.Requests);
    }

    [Fact]
    public async Task ARequiredFileNotInTheCache_WhileTheNetworkIsDown_FailsWithTheNextStep()
    {
        var hub = new Hub { Offline = true };
        using var downloader = new HuggingFaceDownloader(_cacheDir, hub);

        var ex = await Assert.ThrowsAsync<ModelDownloadException>(() =>
            downloader.DownloadModelAsync(Repo, subfolder: Subfolder, cancellationToken: Ct));

        Assert.Contains("model.onnx", ex.Message, StringComparison.Ordinal);
        Assert.Contains("DisableAutoDownload", ex.Message, StringComparison.Ordinal);
        Assert.Equal(Repo, ex.ModelId);
        Assert.IsType<HttpRequestException>(ex.InnerException);
    }

    [Fact]
    public async Task AnHttpStatusFailure_IsNotTreatedAsOffline()
    {
        // A server that answers is not "the network is down": a required file it refuses still fails as before.
        var hub = new Hub { FileStatus = HttpStatusCode.Forbidden };
        using var downloader = new HuggingFaceDownloader(_cacheDir, hub);

        var ex = await Assert.ThrowsAsync<HttpRequestException>(() =>
            downloader.DownloadModelAsync(Repo, ["model.onnx"], subfolder: Subfolder, cancellationToken: Ct));

        Assert.Equal(HttpStatusCode.Forbidden, ex.StatusCode);
    }

    [Fact]
    public async Task Discovery_WithAFileListOlderThanADay_LoadsWhileTheNetworkIsDown()
    {
        using (var online = new HuggingFaceDownloader(_cacheDir, new Hub()))
            await online.DownloadWithDiscoveryAsync(Repo, cancellationToken: Ct);
        AgeFileList();

        var hub = new Hub { Offline = true };
        using var downloader = new HuggingFaceDownloader(_cacheDir, hub);
        var (dir, discovery) = await downloader.DownloadWithDiscoveryAsync(Repo, cancellationToken: Ct);

        Assert.Contains("onnx/model.onnx", discovery.OnnxFiles);
        Assert.True(File.Exists(Path.Combine(dir, "onnx", "model.onnx")));
        Assert.NotEmpty(hub.Requests);
    }

    private string RepoDir => CacheManager.GetRepositoryDirectory(_cacheDir, Repo);

    // Loads the model online with the default file list, the way a catalog alias does, and returns its directory.
    private async Task<string> LoadOnlineAsync()
    {
        var hub = new Hub();
        using var downloader = new HuggingFaceDownloader(_cacheDir, hub);

        var dir = await downloader.DownloadModelAsync(Repo, subfolder: Subfolder, cancellationToken: Ct);

        Assert.Contains(hub.Requests, p => p.EndsWith("/resolve/" + Commit + "/onnx/model.onnx", StringComparison.Ordinal));
        return dir;
    }

    private void AgeFileList()
    {
        File.SetLastWriteTimeUtc(ModelDiscoveryService.GetListingPath(_cacheDir, Repo, "main"), DateTime.UtcNow.AddDays(-2));
    }

    /// <summary>
    /// Serves a repository with its model under <c>onnx/</c> and tokenizer files at the root; every other file is 404.
    /// When <see cref="Offline"/>, every request fails at the connection level, with no HTTP status. Records every
    /// request either way.
    /// </summary>
    private sealed class Hub : HttpMessageHandler
    {
        private readonly List<string> _requests = [];

        public bool Offline { get; init; }
        public HttpStatusCode? FileStatus { get; init; }

        public IReadOnlyList<string> Requests
        {
            get
            {
                lock (_requests)
                    return [.. _requests];
            }
        }

        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            var path = Uri.UnescapeDataString(request.RequestUri!.AbsolutePath);
            lock (_requests)
                _requests.Add(path);

            if (Offline)
                throw new HttpRequestException("No connection could be made because the target machine actively refused it.");

            if (path == $"/api/models/{Repo}/tree/main")
            {
                var entries = string.Join(",", s_files.Select(f => $$"""{"path":"{{f.Key}}","type":"file","size":{{Encoding.UTF8.GetByteCount(f.Value)}}}"""));
                return Task.FromResult(Json("[" + entries + "]"));
            }

            if (path == $"/api/models/{Repo}/revision/main")
                return Task.FromResult(Json($$"""{"id":"{{Repo}}","sha":"{{Commit}}"}"""));

            foreach (var revision in new[] { "main", Commit })
            {
                var prefix = $"/{Repo}/resolve/{revision}/";
                if (!path.StartsWith(prefix, StringComparison.Ordinal))
                    continue;

                if (FileStatus is { } status)
                    return Task.FromResult(new HttpResponseMessage(status));

                return Task.FromResult(s_files.TryGetValue(path[prefix.Length..], out var body)
                    ? new HttpResponseMessage(HttpStatusCode.OK) { Content = new ByteArrayContent(Encoding.UTF8.GetBytes(body)) }
                    : new HttpResponseMessage(HttpStatusCode.NotFound));
            }

            return Task.FromResult(new HttpResponseMessage(HttpStatusCode.NotFound));
        }

        private static HttpResponseMessage Json(string body) =>
            new(HttpStatusCode.OK) { Content = new StringContent(body, Encoding.UTF8, "application/json") };
    }
}
