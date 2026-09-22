using System.Net;
using System.Text;
using LMSupply.Download;
using LMSupply.Exceptions;

namespace LMSupply.Core.Tests.Download;

/// <summary>
/// A release that moves where a model's files are read from must not make every install fetch the
/// model again. 0.63.0 moved a subfolder's files from the snapshot root into the subfolder and the new
/// path looked in exactly one place, so an existing cache — the file present at the root, byte for
/// byte — downloaded it once more (about 1 GB for the default embedder) and kept both copies. A file
/// wanted at one place and present at the other, with the length the repository lists, is now moved
/// there: no request, one copy.
/// </summary>
public sealed class HuggingFaceDownloaderAdoptionTests : IDisposable
{
    private const string Repo = "acme/adopted-model";
    private readonly string _cacheDir = Path.Combine(Path.GetTempPath(), "lmsupply-adopt-" + Guid.NewGuid().ToString("N"));

    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    public void Dispose()
    {
        if (Directory.Exists(_cacheDir))
        {
            Directory.Delete(_cacheDir, recursive: true);
        }
    }

    [Fact]
    public async Task ARootCopyFromBefore063_IsMovedIntoTheSubfolder_NotDownloadedAgain()
    {
        var snapshot = CacheManager.GetModelDirectory(_cacheDir, Repo);
        Directory.CreateDirectory(snapshot);
        await File.WriteAllTextAsync(Path.Combine(snapshot, "model.onnx"), "graph", Ct);      // 5 bytes, as the listing says
        await File.WriteAllTextAsync(Path.Combine(snapshot, "tokenizer.json"), "{}", Ct);
        var hub = new ListingHub(files: [("onnx/model.onnx", 5), ("tokenizer.json", 2)]);
        using var downloader = new HuggingFaceDownloader(_cacheDir, hub, localFilesOnly: false);

        var modelDir = await downloader.DownloadModelAsync(Repo, ["model.onnx", "tokenizer.json"], subfolder: "onnx", cancellationToken: Ct);

        Assert.Equal(Path.Combine(snapshot, "onnx"), modelDir);
        Assert.Equal("graph", await File.ReadAllTextAsync(Path.Combine(modelDir, "model.onnx"), Ct));
        Assert.Equal("{}", await File.ReadAllTextAsync(Path.Combine(modelDir, "tokenizer.json"), Ct));
        Assert.False(File.Exists(Path.Combine(snapshot, "model.onnx")), "moved, not copied — one copy, not two");
        Assert.DoesNotContain(hub.Paths, p => p.Contains("/resolve/"));
    }

    [Fact]
    public async Task ACopyOfTheWrongLength_IsNotAdopted()
    {
        var snapshot = CacheManager.GetModelDirectory(_cacheDir, Repo);
        Directory.CreateDirectory(snapshot);
        await File.WriteAllTextAsync(Path.Combine(snapshot, "model.onnx"), "truncated-graph", Ct);   // 15 bytes; the listing says 5
        var hub = new ListingHub(files: [("onnx/model.onnx", 5)], serveModel: true);
        using var downloader = new HuggingFaceDownloader(_cacheDir, hub, localFilesOnly: false);

        var modelDir = await downloader.DownloadModelAsync(Repo, ["model.onnx"], subfolder: "onnx", cancellationToken: Ct);

        Assert.Equal("graph", await File.ReadAllTextAsync(Path.Combine(modelDir, "model.onnx"), Ct));
        Assert.Contains(hub.Paths, p => p == $"/{Repo}/resolve/main/onnx/model.onnx");
        Assert.True(File.Exists(Path.Combine(snapshot, "model.onnx")), "a copy that does not match is left where it was, not deleted");
    }

    [Fact]
    public async Task ASubfolderCopy_IsMovedToTheRoot_WhenTheRootLayoutWantsIt()
    {
        // The other direction: a repository loaded by alias (files under onnx/) is then loaded by id
        // (root layout). The root wants tokenizer.json; onnx/ has it.
        var snapshot = CacheManager.GetModelDirectory(_cacheDir, Repo);
        Directory.CreateDirectory(Path.Combine(snapshot, "onnx"));
        await File.WriteAllTextAsync(Path.Combine(snapshot, "onnx", "tokenizer.json"), "{}", Ct);
        await File.WriteAllTextAsync(Path.Combine(snapshot, "onnx", "model.onnx"), "graph", Ct);
        var hub = new ListingHub(files: [("onnx/model.onnx", 5), ("tokenizer.json", 2)]);
        using var downloader = new HuggingFaceDownloader(_cacheDir, hub, localFilesOnly: false);

        var (localPath, _) = await downloader.DownloadWithDiscoveryAsync(Repo, cancellationToken: Ct);

        Assert.Equal(snapshot, localPath);
        Assert.Equal("{}", await File.ReadAllTextAsync(Path.Combine(snapshot, "tokenizer.json"), Ct));
        Assert.False(File.Exists(Path.Combine(snapshot, "onnx", "tokenizer.json")));
        Assert.DoesNotContain(hub.Paths, p => p.Contains("/resolve/"));
    }

    [Fact]
    public async Task ReadOnlyMode_DoesNotMove_AndStillReportsTheMiss()
    {
        var snapshot = CacheManager.GetModelDirectory(_cacheDir, Repo);
        Directory.CreateDirectory(snapshot);
        await File.WriteAllTextAsync(Path.Combine(snapshot, "model.onnx"), "graph", Ct);
        using var downloader = new HuggingFaceDownloader(_cacheDir, new ListingHub(files: []), localFilesOnly: true);

        await Assert.ThrowsAsync<ModelNotFoundException>(() => downloader.DownloadModelAsync(Repo, ["model.onnx"], subfolder: "onnx", cancellationToken: Ct));

        Assert.True(File.Exists(Path.Combine(snapshot, "model.onnx")), "an offline load reads a read-only cache and writes nothing");
    }

    /// <summary>Answers the tree listing with the given files and sizes; serves only <c>onnx/model.onnx</c> when asked to.</summary>
    private sealed class ListingHub((string Path, long Size)[] files, bool serveModel = false) : HttpMessageHandler
    {
        private readonly List<string> _paths = [];

        public IReadOnlyList<string> Paths
        {
            get
            {
                lock (_paths)
                {
                    return [.. _paths];
                }
            }
        }

        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            var path = Uri.UnescapeDataString(request.RequestUri!.AbsolutePath);
            lock (_paths)
            {
                _paths.Add(path);
            }

            if (path == $"/api/models/{Repo}/tree/main")
            {
                var root = files.Where(f => !f.Path.Contains('/')).Select(f => $$$"""{"path":"{{{f.Path}}}","type":"file","size":{{{f.Size}}}}""");
                var dirs = files.Where(f => f.Path.Contains('/')).Select(f => f.Path.Split('/')[0]).Distinct().Select(d => $$$"""{"path":"{{{d}}}","type":"directory"}""");
                return Ok("[" + string.Join(",", root.Concat(dirs)) + "]");
            }

            if (path.StartsWith($"/api/models/{Repo}/tree/main/", StringComparison.Ordinal))
            {
                var dir = path[$"/api/models/{Repo}/tree/main/".Length..];
                var inDir = files.Where(f => f.Path.StartsWith(dir + "/", StringComparison.Ordinal)).Select(f => $$$"""{"path":"{{{f.Path}}}","type":"file","size":{{{f.Size}}}}""");
                return Ok("[" + string.Join(",", inDir) + "]");
            }

            if (serveModel && path == $"/{Repo}/resolve/main/onnx/model.onnx")
            {
                return Ok("graph");
            }

            return Task.FromResult(new HttpResponseMessage(HttpStatusCode.NotFound));
        }

        private static Task<HttpResponseMessage> Ok(string body)
            => Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK) { Content = new ByteArrayContent(Encoding.UTF8.GetBytes(body)) });
    }
}
