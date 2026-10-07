using System.Net;
using System.Net.Http.Headers;
using System.Text;
using LMSupply.Embedder.Utils;

namespace LMSupply.Embedder.Tests;

/// <summary>
/// The embedding GGUF downloader used to write straight to the final path — no ".part", no length
/// check — so a body that ended early left a file named as the model that llama.cpp would refuse to
/// load, and the cache check saw only that it existed. It now goes through the shared resumable
/// download with the length the repository listed.
/// </summary>
public sealed class GgufDownloaderIntegrityTests : IDisposable
{
    private const string Repo = "acme/gguf-embedder";
    private const string File_ = "embed-Q8_0.gguf";
    private static readonly byte[] Model = Enumerable.Range(0, 4000).Select(i => (byte)(i % 241)).ToArray();

    private readonly string _cacheDir = Path.Combine(Path.GetTempPath(), "lmsupply-emb-gguf-integrity-" + Guid.NewGuid().ToString("N"));

    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    public void Dispose()
    {
        if (Directory.Exists(_cacheDir))
            Directory.Delete(_cacheDir, recursive: true);
    }

    // Where a download writes when the revision cannot be resolved to a commit (the fake hub below answers 404 for
    // the revision unless given one), and the private tree earlier versions wrote to — still read, never written.
    private string CachedPath => Path.Combine(_cacheDir, "models--acme--gguf-embedder", "snapshots", "main", File_);
    private string LegacyPath => Path.Combine(_cacheDir, "gguf-embeddings", "acme_gguf-embedder", File_);

    [Fact]
    public async Task ABodyThatEndsEarly_IsResumedFromThePart_NotLeftAsTheModel()
    {
        var hub = new Hub { BodyLimits = new Queue<int>([2400]) };
        using var downloader = new GgufDownloader(_cacheDir, localFilesOnly: false, hub);

        var path = await downloader.DownloadAsync(Repo, cancellationToken: Ct);

        Assert.Equal(CachedPath, path);
        Assert.Equal(Model, await File.ReadAllBytesAsync(path, Ct));
        Assert.False(File.Exists(path + ".part"));
        Assert.Equal(["bytes=2400-"], hub.Ranges);
    }

    [Fact]
    public async Task ACachedFileShorterThanTheListing_IsDownloadedAgain()
    {
        Directory.CreateDirectory(Path.GetDirectoryName(LegacyPath)!);
        await File.WriteAllBytesAsync(LegacyPath, Model[..2400], Ct);
        var hub = new Hub();
        using var downloader = new GgufDownloader(_cacheDir, localFilesOnly: false, hub);

        var path = await downloader.DownloadAsync(Repo, cancellationToken: Ct);

        Assert.Equal(Model, await File.ReadAllBytesAsync(path, Ct));
        Assert.Equal(1, hub.ResolveRequests);
    }

    [Fact]
    public async Task AFileInTheEarlierPrivateTree_IsStillLoaded_WithoutADownload()
    {
        Directory.CreateDirectory(Path.GetDirectoryName(LegacyPath)!);
        await File.WriteAllBytesAsync(LegacyPath, Model, Ct);
        var hub = new Hub();

        using (var online = new GgufDownloader(_cacheDir, localFilesOnly: false, hub))
            Assert.Equal(LegacyPath, await online.DownloadAsync(Repo, cancellationToken: Ct));
        using (var offline = new GgufDownloader(_cacheDir, localFilesOnly: true, hub))
            Assert.Equal(LegacyPath, await offline.DownloadAsync(Repo, cancellationToken: Ct));

        Assert.Equal(0, hub.ResolveRequests);
        // The repository directory may hold this library's listing record (.lmsupply/listings); no snapshot holds the file.
        Assert.False(Directory.Exists(Path.Combine(_cacheDir, "models--acme--gguf-embedder", "snapshots")), "the old file is not moved or copied");
    }

    [Fact]
    public async Task WithAResolvableCommit_TheFileIsWrittenInTheHubLayout()
    {
        var hub = new Hub { Commit = HubCommit };
        using var downloader = new GgufDownloader(_cacheDir, localFilesOnly: false, hub);

        var path = await downloader.DownloadAsync(Repo, cancellationToken: Ct);

        var repoDir = Path.Combine(_cacheDir, "models--acme--gguf-embedder");
        Assert.Equal(Path.Combine(repoDir, "snapshots", HubCommit, File_), path);
        Assert.Equal(Model, await File.ReadAllBytesAsync(path, Ct));
        Assert.Equal(HubCommit, await File.ReadAllTextAsync(Path.Combine(repoDir, "refs", "main"), Ct));
        if (new FileInfo(path).LinkTarget is not null)
            Assert.True(File.Exists(Path.Combine(repoDir, "blobs", LfsOid)));
        Assert.True(File.Exists(Path.Combine(repoDir, ".lmsupply", "manifests", HubCommit + ".json")));
        Assert.False(Directory.Exists(Path.Combine(_cacheDir, "gguf-embeddings")));

        // Found again through the ref, offline, and deleted as this library's own.
        using (var offline = new GgufDownloader(_cacheDir, localFilesOnly: true))
            Assert.Equal(path, await offline.DownloadAsync(Repo, cancellationToken: Ct));
        Assert.True(LMSupply.Download.CacheManager.DeleteModel(_cacheDir, Repo));
        Assert.False(Directory.Exists(repoDir));
    }

    private const string HubCommit = "0123456789abcdef0123456789abcdef01234567";
    private static readonly string LfsOid = Convert.ToHexStringLower(System.Security.Cryptography.SHA256.HashData(Model));

    private sealed class Hub : HttpMessageHandler
    {
        private readonly List<string> _ranges = [];
        private int _resolveRequests;

        /// <summary>The commit "main" resolves to; null answers the revision lookup with 404.</summary>
        public string? Commit { get; init; }

        public Queue<int> BodyLimits { get; init; } = new();
        public int ResolveRequests => Volatile.Read(ref _resolveRequests);
        public IReadOnlyList<string> Ranges { get { lock (_ranges) return [.. _ranges]; } }

        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            var path = Uri.UnescapeDataString(request.RequestUri!.AbsolutePath);
            if (path == $"/api/models/{Repo}/tree/main")
                return Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK)
                {
                    Content = new StringContent(
                        $$$"""[{"path":"{{{File_}}}","type":"file","size":4000,"oid":"{{{new string('4', 40)}}}","lfs":{"oid":"{{{LfsOid}}}","size":4000}}]""",
                        Encoding.UTF8, "application/json"),
                });
            if (path == $"/api/models/{Repo}/revision/main" && Commit is not null)
                return Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK)
                {
                    Content = new StringContent($$"""{"sha":"{{Commit}}"}""", Encoding.UTF8, "application/json"),
                });
            if (path != $"/{Repo}/resolve/main/{File_}" && path != $"/{Repo}/resolve/{Commit}/{File_}")
                return Task.FromResult(new HttpResponseMessage(HttpStatusCode.NotFound));

            Interlocked.Increment(ref _resolveRequests);
            var from = 0L;
            if (request.Headers.Range?.Ranges.FirstOrDefault() is { From: { } f })
            {
                lock (_ranges) _ranges.Add($"bytes={f}-");
                from = f;
            }
            int? limit = null;
            lock (BodyLimits) { if (BodyLimits.TryDequeue(out var l)) limit = l; }

            var remaining = Model.Length - from;
            var body = Model.AsMemory((int)from, (int)Math.Min(remaining, limit ?? remaining));
            var content = new ByteArrayContent(body.ToArray());
            content.Headers.ContentLength = remaining;
            var response = new HttpResponseMessage(from > 0 ? HttpStatusCode.PartialContent : HttpStatusCode.OK) { Content = content };
            if (from > 0)
                content.Headers.ContentRange = new ContentRangeHeaderValue(from, Model.Length - 1, Model.Length);
            return Task.FromResult(response);
        }
    }
}
