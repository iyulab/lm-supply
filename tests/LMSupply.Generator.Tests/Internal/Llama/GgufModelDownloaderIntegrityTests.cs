using System.Net;
using System.Net.Http.Headers;
using System.Text;
using LMSupply.Generator.Internal.Llama;

namespace LMSupply.Generator.Tests.Internal.Llama;

/// <summary>
/// A GGUF file is the model only at the length the repository lists. The Llama downloader used to rename
/// whatever the body delivered into place and to trust any cached file that existed — the same shape
/// that left a 470 MB ONNX model on disk as 43 MB. It now goes through the shared resumable download:
/// a body that ends early is resumed from its ".part", and a cached file of the wrong length is fetched
/// again.
/// </summary>
public sealed class GgufModelDownloaderIntegrityTests : IDisposable
{
    private const string Repo = "acme/gguf-model";
    private const string File_ = "model-Q4_K_M.gguf";
    private static readonly byte[] Model = Enumerable.Range(0, 4000).Select(i => (byte)(i % 239)).ToArray();

    private readonly string _cacheDir = Path.Combine(Path.GetTempPath(), "lmsupply-gguf-integrity-" + Guid.NewGuid().ToString("N"));

    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    public void Dispose()
    {
        if (Directory.Exists(_cacheDir))
            Directory.Delete(_cacheDir, recursive: true);
    }

    private string CachedPath => Path.Combine(_cacheDir, "models--acme--gguf-model", "snapshots", "main", File_);

    [Fact]
    public async Task ABodyThatEndsEarly_IsResumedFromThePart()
    {
        var hub = new Hub { BodyLimits = new Queue<int>([2400]) };
        using var downloader = new GgufModelDownloader(_cacheDir, hfToken: null, localFilesOnly: false, hub);

        var path = await downloader.DownloadAsync(Repo, File_, cancellationToken: Ct);

        Assert.Equal(CachedPath, path);
        Assert.Equal(Model, await File.ReadAllBytesAsync(path, Ct));
        Assert.False(File.Exists(path + ".part"));
        Assert.Equal(["bytes=2400-"], hub.Ranges);
    }

    [Fact]
    public async Task ACachedFileShorterThanTheListing_IsDownloadedAgain()
    {
        Directory.CreateDirectory(Path.GetDirectoryName(CachedPath)!);
        await File.WriteAllBytesAsync(CachedPath, Model[..2400], Ct);
        var hub = new Hub();
        using var downloader = new GgufModelDownloader(_cacheDir, hfToken: null, localFilesOnly: false, hub);

        var path = await downloader.DownloadAsync(Repo, File_, cancellationToken: Ct);

        Assert.Equal(Model, await File.ReadAllBytesAsync(path, Ct));
        Assert.Equal(1, hub.ResolveRequests);
    }

    [Fact]
    public async Task ACachedFileAtTheListedLength_IsServedWithoutADownload()
    {
        Directory.CreateDirectory(Path.GetDirectoryName(CachedPath)!);
        await File.WriteAllBytesAsync(CachedPath, Model, Ct);
        var hub = new Hub();
        using var downloader = new GgufModelDownloader(_cacheDir, hfToken: null, localFilesOnly: false, hub);

        await downloader.DownloadAsync(Repo, File_, cancellationToken: Ct);

        Assert.Equal(0, hub.ResolveRequests);
    }

    private sealed class Hub : HttpMessageHandler
    {
        private readonly List<string> _ranges = [];
        private int _resolveRequests;

        public Queue<int> BodyLimits { get; init; } = new();
        public int ResolveRequests => Volatile.Read(ref _resolveRequests);
        public IReadOnlyList<string> Ranges { get { lock (_ranges) return [.. _ranges]; } }

        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            var path = Uri.UnescapeDataString(request.RequestUri!.AbsolutePath);
            if (path == $"/api/models/{Repo}/tree/main")
                return Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK)
                {
                    Content = new StringContent($$"""[{"path":"{{File_}}","type":"file","size":4000}]""", Encoding.UTF8, "application/json"),
                });
            if (path != $"/{Repo}/resolve/main/{File_}")
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
            content.Headers.ContentLength = remaining; // what the server announces, whatever it delivers
            var response = new HttpResponseMessage(from > 0 ? HttpStatusCode.PartialContent : HttpStatusCode.OK) { Content = content };
            if (from > 0)
                content.Headers.ContentRange = new ContentRangeHeaderValue(from, Model.Length - 1, Model.Length);
            return Task.FromResult(response);
        }
    }
}
