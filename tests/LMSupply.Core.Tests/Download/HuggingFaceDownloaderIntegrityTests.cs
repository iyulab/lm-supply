using System.Net;
using System.Net.Http.Headers;
using System.Text;
using LMSupply.Download;
using LMSupply.Exceptions;

namespace LMSupply.Core.Tests.Download;

/// <summary>
/// A downloaded file is the file only when it is as long as the server announced and as the repository
/// listed. These facts drive the downloader over a fake hub that can end a body early, drop the
/// connection, or serve the same file to two callers at once — the shapes that left a 470 MB model on
/// disk as 43 MB with a manifest certifying it complete. A fake handler does no HTTP framing, so a
/// body shorter than its Content-Length reaches the downloader as a plain end of stream, exactly as the
/// truncation did in the field. The file is larger than 1 KiB so the downloader streams it (a smaller
/// binary is read whole to check for a Git LFS pointer).
/// </summary>
public sealed class HuggingFaceDownloaderIntegrityTests : IDisposable
{
    private const string Repo = "acme/integrity-model";
    private static readonly byte[] Model = Enumerable.Range(0, 4000).Select(i => (byte)(i % 251)).ToArray();

    private readonly string _cacheDir = Path.Combine(Path.GetTempPath(), "lmsupply-integrity-" + Guid.NewGuid().ToString("N"));

    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    public void Dispose()
    {
        if (Directory.Exists(_cacheDir))
            Directory.Delete(_cacheDir, recursive: true);
    }

    private string ModelDir => CacheManager.GetModelDirectory(_cacheDir, Repo);
    private string ModelPath => Path.Combine(ModelDir, "model.onnx");

    [Fact]
    public async Task ABodyThatEndsEarly_IsResumed_AndTheManifestRecordsTheListedLength()
    {
        // First response: Content-Length 4000, body 2400 and then end of stream. Second: 206 for the rest.
        var hub = new Hub { BodyLimits = new Queue<int>([2400]) };
        using var downloader = new HuggingFaceDownloader(_cacheDir, hub);

        var dir = await downloader.DownloadModelAsync(Repo, ["model.onnx"], cancellationToken: Ct);

        Assert.Equal(Model, await File.ReadAllBytesAsync(ModelPath, Ct));
        Assert.False(File.Exists(ModelPath + ".part"));
        Assert.Equal(["bytes=2400-"], hub.Ranges);
        var manifest = await DownloadManifest.ReadAsync(dir);
        Assert.NotNull(manifest);
        Assert.Equal(DownloadManifest.VerifiedVersion, manifest.Version);
        Assert.Equal(4000, Assert.Single(manifest.Files, f => f.Path == "model.onnx").Size);
    }

    [Fact]
    public async Task ABodyThatAlwaysEndsAtTheSameOffset_LeavesOnlyThePart_AndThrows()
    {
        var hub = new Hub { BodyLimits = new Queue<int>([2400, 2400, 2400, 2400, 2400]), ResumeFromRange = false };
        using var downloader = new HuggingFaceDownloader(_cacheDir, hub);

        var ex = await Assert.ThrowsAnyAsync<ModelDownloadException>(() => downloader.DownloadModelAsync(Repo, ["model.onnx"], cancellationToken: Ct));

        Assert.Contains("2400 of 4000", ex.Message);
        Assert.False(File.Exists(ModelPath));
        Assert.True(File.Exists(ModelPath + ".part"));
        Assert.Null(await DownloadManifest.ReadAsync(ModelDir));
    }

    [Fact]
    public async Task AFinalFileShorterThanTheListing_IsDownloadedAgain()
    {
        Directory.CreateDirectory(ModelDir);
        await File.WriteAllBytesAsync(ModelPath, Model[..2400], Ct);
        var hub = new Hub();
        using var downloader = new HuggingFaceDownloader(_cacheDir, hub);

        await downloader.DownloadModelAsync(Repo, ["model.onnx"], cancellationToken: Ct);

        Assert.Equal(Model, await File.ReadAllBytesAsync(ModelPath, Ct));
        Assert.Equal(1, hub.ResolveRequests);
        Assert.Empty(hub.Ranges); // a final file of the wrong length is not a prefix to resume from
    }

    [Fact]
    public async Task AVerifiedManifest_AnswersForACachedFile_WithoutARequest()
    {
        Directory.CreateDirectory(ModelDir);
        await File.WriteAllBytesAsync(ModelPath, Model, Ct);
        await DownloadManifest.WriteAsync(ModelDir, new DownloadManifest
        {
            Version = DownloadManifest.VerifiedVersion,
            RepoId = Repo,
            Files = [new ManifestFileEntry { Path = "model.onnx", Size = 4000 }],
        });
        var hub = new Hub();
        using var downloader = new HuggingFaceDownloader(_cacheDir, hub);

        await downloader.DownloadModelAsync(Repo, ["model.onnx"], cancellationToken: Ct);

        Assert.Equal(0, hub.Requests);
    }

    [Fact]
    public async Task AVersion1Manifest_CertifiesNothing_TheListingIsConsulted()
    {
        Directory.CreateDirectory(ModelDir);
        await File.WriteAllBytesAsync(ModelPath, Model[..2400], Ct);
        await DownloadManifest.WriteAsync(ModelDir, new DownloadManifest
        {
            Version = 1,
            RepoId = Repo,
            Files = [new ManifestFileEntry { Path = "model.onnx", Size = 2400 }],
        });
        var hub = new Hub();
        using var downloader = new HuggingFaceDownloader(_cacheDir, hub);

        await downloader.DownloadModelAsync(Repo, ["model.onnx"], cancellationToken: Ct);

        Assert.Equal(Model, await File.ReadAllBytesAsync(ModelPath, Ct));
        Assert.Equal(DownloadManifest.VerifiedVersion, (await DownloadManifest.ReadAsync(ModelDir))!.Version);
    }

    [Fact]
    public async Task ADroppedConnection_IsResumedFromThePart()
    {
        var hub = new Hub { ThrowAfter = new Queue<int>([2400]) };
        using var downloader = new HuggingFaceDownloader(_cacheDir, hub);

        await downloader.DownloadModelAsync(Repo, ["model.onnx"], cancellationToken: Ct);

        Assert.Equal(Model, await File.ReadAllBytesAsync(ModelPath, Ct));
        Assert.Equal(["bytes=2400-"], hub.Ranges);
    }

    [Fact]
    public async Task TwoCallersForTheSameFile_ShareOneDownload()
    {
        var hub = new Hub { BodyDelay = TimeSpan.FromMilliseconds(300) };
        using var downloader = new HuggingFaceDownloader(_cacheDir, hub);

        await Task.WhenAll(
            downloader.DownloadModelAsync(Repo, ["model.onnx"], cancellationToken: Ct),
            downloader.DownloadModelAsync(Repo, ["model.onnx"], cancellationToken: Ct));

        Assert.Equal(Model, await File.ReadAllBytesAsync(ModelPath, Ct));
        Assert.Equal(1, hub.ResolveRequests);
    }

    [Fact]
    public async Task DiscoveryPath_ChecksCachedFilesAgainstTheListing()
    {
        Directory.CreateDirectory(ModelDir);
        await File.WriteAllBytesAsync(ModelPath, Model[..2400], Ct);
        var hub = new Hub();
        using var downloader = new HuggingFaceDownloader(_cacheDir, hub);

        var (dir, _) = await downloader.DownloadWithDiscoveryAsync(Repo, cancellationToken: Ct);

        Assert.Equal(Model, await File.ReadAllBytesAsync(Path.Combine(dir, "model.onnx"), Ct));
        var manifest = await DownloadManifest.ReadAsync(dir);
        Assert.Equal(DownloadManifest.VerifiedVersion, manifest!.Version);
        Assert.Equal(4000, Assert.Single(manifest.Files, f => f.Path == "model.onnx").Size);
    }

    [Fact]
    public async Task AFileTheRepositoryDoesNotHave_LeavesNoPartBehind()
    {
        var hub = new Hub();
        using var downloader = new HuggingFaceDownloader(_cacheDir, hub);

        // The default file list asks for optional files (vocab, merges, external data) this repository lacks.
        await downloader.DownloadModelAsync(Repo, cancellationToken: Ct);

        Assert.Equal(Model, await File.ReadAllBytesAsync(ModelPath, Ct));
        Assert.Empty(Directory.GetFiles(ModelDir, "*.part"));
        Assert.True(ModelDirectoryValidator.Validate(ModelDir).IsValid);
    }

    [Fact]
    public async Task ATransientStatus_IsRetried_ByDefault()
    {
        // 503 on the first resolve, then the file: the library-wide transient rule applies without any
        // caller naming it.
        var hub = new Hub { StatusesBeforeBody = new Queue<HttpStatusCode>([HttpStatusCode.ServiceUnavailable]) };
        using var downloader = new HuggingFaceDownloader(_cacheDir, hub);

        await downloader.DownloadModelAsync(Repo, ["model.onnx"], cancellationToken: Ct);

        Assert.Equal(Model, await File.ReadAllBytesAsync(ModelPath, Ct));
        Assert.Equal(2, hub.ResolveRequests);
    }

    /// <summary>A hub with one file, "model.onnx" (4000 bytes), whose transport misbehaves on request.</summary>
    private sealed class Hub : HttpMessageHandler
    {
        private readonly List<string> _ranges = [];
        private int _requests;
        private int _resolveRequests;

        /// <summary>Per response, how many body bytes to send before ending the stream (Content-Length stays the real length).</summary>
        public Queue<int> BodyLimits { get; init; } = new();

        /// <summary>Per response, after how many bytes the body stream throws <see cref="IOException"/>.</summary>
        public Queue<int> ThrowAfter { get; init; } = new();

        /// <summary>Per resolve request, a failure status to answer with instead of a body.</summary>
        public Queue<HttpStatusCode> StatusesBeforeBody { get; init; } = new();

        /// <summary>Whether a Range request is honoured with 206; when false the server sends the whole file again.</summary>
        public bool ResumeFromRange { get; init; } = true;

        public TimeSpan BodyDelay { get; init; }

        public int Requests => Volatile.Read(ref _requests);
        public int ResolveRequests => Volatile.Read(ref _resolveRequests);

        public IReadOnlyList<string> Ranges
        {
            get { lock (_ranges) return [.. _ranges]; }
        }

        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            Interlocked.Increment(ref _requests);
            var path = Uri.UnescapeDataString(request.RequestUri!.AbsolutePath);

            if (path == $"/api/models/{Repo}/tree/main")
                return Task.FromResult(Json("""[{"path":"model.onnx","type":"file","size":4000},{"path":"config.json","type":"file","size":2}]"""));
            if (path == $"/{Repo}/resolve/main/config.json")
                return Task.FromResult(Bytes("{}"u8.ToArray(), 0, null));
            if (path != $"/{Repo}/resolve/main/model.onnx")
                return Task.FromResult(new HttpResponseMessage(HttpStatusCode.NotFound));

            Interlocked.Increment(ref _resolveRequests);
            lock (StatusesBeforeBody)
            {
                if (StatusesBeforeBody.TryDequeue(out var status))
                    return Task.FromResult(new HttpResponseMessage(status));
            }
            var from = 0L;
            if (request.Headers.Range?.Ranges.FirstOrDefault() is { From: { } f })
            {
                lock (_ranges) _ranges.Add($"bytes={f}-");
                if (ResumeFromRange)
                    from = f;
            }

            int? limit = null, throwAfter = null;
            lock (BodyLimits) { if (BodyLimits.TryDequeue(out var l)) limit = l; }
            lock (ThrowAfter) { if (ThrowAfter.TryDequeue(out var t)) throwAfter = t; }

            var response = Bytes(Model, from, limit, throwAfter, BodyDelay);
            return Task.FromResult(response);
        }

        private static HttpResponseMessage Json(string body) =>
            new(HttpStatusCode.OK) { Content = new StringContent(body, Encoding.UTF8, "application/json") };

        private static HttpResponseMessage Bytes(byte[] all, long from, int? limit, int? throwAfter = null, TimeSpan delay = default)
        {
            var remaining = all.Length - from;
            var body = all.AsMemory((int)from, (int)Math.Min(remaining, limit ?? remaining));
            var content = new StreamContent(new MisbehavingStream(body, throwAfter, delay));
            content.Headers.ContentLength = remaining; // what the server announces, whatever it delivers
            var response = new HttpResponseMessage(from > 0 ? HttpStatusCode.PartialContent : HttpStatusCode.OK) { Content = content };
            if (from > 0)
                content.Headers.ContentRange = new ContentRangeHeaderValue(from, all.Length - 1, all.Length);
            return response;
        }
    }

    /// <summary>Serves a fixed body, optionally throwing after N bytes or pausing before the first byte.</summary>
    private sealed class MisbehavingStream(ReadOnlyMemory<byte> body, int? throwAfter, TimeSpan delay) : Stream
    {
        private int _position;
        private bool _delayed;

        public override async ValueTask<int> ReadAsync(Memory<byte> buffer, CancellationToken cancellationToken = default)
        {
            if (!_delayed && delay > TimeSpan.Zero)
            {
                _delayed = true;
                await Task.Delay(delay, cancellationToken);
            }
            if (throwAfter is { } t && _position >= t)
                throw new IOException("connection reset by peer (simulated)");
            var take = Math.Min(buffer.Length, body.Length - _position);
            if (throwAfter is { } limit)
                take = Math.Min(take, limit - _position);
            body.Slice(_position, take).CopyTo(buffer);
            _position += take;
            return take;
        }

        public override int Read(byte[] buffer, int offset, int count) => ReadAsync(buffer.AsMemory(offset, count)).AsTask().GetAwaiter().GetResult();
        public override bool CanRead => true;
        public override bool CanSeek => false;
        public override bool CanWrite => false;
        public override long Length => body.Length;
        public override long Position { get => _position; set => throw new NotSupportedException(); }
        public override void Flush() { }
        public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();
        public override void SetLength(long value) => throw new NotSupportedException();
        public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();
    }
}
