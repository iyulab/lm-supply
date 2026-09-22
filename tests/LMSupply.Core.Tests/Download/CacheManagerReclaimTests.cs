using AwesomeAssertions;
using System.Net;
using LMSupply.Download;

namespace LMSupply.Core.Tests.Download;

/// <summary>
/// A release that moved where a model's files are read from (0.63.0) left the old copy at the snapshot
/// root next to the new one in the subfolder — byte for byte the same file, twice, for every model an
/// install had cached. <see cref="CacheManager.FindReclaimable"/> names exactly those root copies (same
/// name, same length, same SHA-256, and the subfolder twin is what a manifest lists) and nothing wider;
/// <see cref="CacheManager.Reclaim"/> deletes what it is handed after re-checking each entry.
/// </summary>
public sealed class CacheManagerReclaimTests : IDisposable
{
    private const string Repo = "acme/twice-cached";
    private readonly string _cacheDir = Path.Combine(Path.GetTempPath(), "lmsupply-reclaim-" + Guid.NewGuid().ToString("N"));

    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    public void Dispose()
    {
        if (Directory.Exists(_cacheDir))
            Directory.Delete(_cacheDir, recursive: true);
    }

    [Fact]
    public async Task TheMeasuredShape_RootCopyOfASubfolderFile_IsReportedWithItsBytes_AndReclaimed()
    {
        var snapshot = await SnapshotWithTwinAsync(rootBytes: "graph-bytes", subfolderBytes: "graph-bytes", manifestInSubfolder: true);

        var found = CacheManager.FindReclaimable(_cacheDir);

        found.Should().ContainSingle();
        var entry = found[0];
        entry.RepoId.Should().Be(Repo);
        entry.Path.Should().Be(Path.Combine(snapshot, "model.onnx"));
        entry.TwinPath.Should().Be(Path.Combine(snapshot, "onnx", "model.onnx"));
        entry.Size.Should().Be("graph-bytes".Length);
        entry.Reason.Should().Contain("onnx/model.onnx");

        var freed = CacheManager.Reclaim(_cacheDir, found);

        freed.Should().Be("graph-bytes".Length);
        File.Exists(Path.Combine(snapshot, "model.onnx")).Should().BeFalse("the root copy is the one that goes");
        File.Exists(Path.Combine(snapshot, "onnx", "model.onnx")).Should().BeTrue("the copy the manifest lists stays");
        CacheManager.FindReclaimable(_cacheDir).Should().BeEmpty("nothing is left to reclaim");
    }

    [Fact]
    public async Task AManifestAtTheRootListingTheSubfolderPath_AlsoCounts()
    {
        // The repository-id download path writes one manifest at the snapshot root with subfolder-prefixed paths.
        await SnapshotWithTwinAsync(rootBytes: "graph-bytes", subfolderBytes: "graph-bytes", manifestInSubfolder: false, manifestAtRoot: true);

        CacheManager.FindReclaimable(_cacheDir).Should().ContainSingle();
    }

    [Fact]
    public async Task ATwinWithTheSameLengthButDifferentBytes_IsNotReported()
    {
        await SnapshotWithTwinAsync(rootBytes: "graph-bytes", subfolderBytes: "graph-BYTES", manifestInSubfolder: true);

        CacheManager.FindReclaimable(_cacheDir).Should().BeEmpty("same name and length is not the same file");
    }

    [Fact]
    public async Task ATwinNoManifestLists_IsNotReported()
    {
        // Without a manifest naming the subfolder copy there is no evidence the loader reads it rather than the root one.
        await SnapshotWithTwinAsync(rootBytes: "graph-bytes", subfolderBytes: "graph-bytes", manifestInSubfolder: false, manifestAtRoot: false);

        CacheManager.FindReclaimable(_cacheDir).Should().BeEmpty();
    }

    [Fact]
    public async Task ATwinOfAnotherLength_AndARootFileWithNoTwin_AreNotReported()
    {
        var snapshot = await SnapshotWithTwinAsync(rootBytes: "graph-bytes-longer", subfolderBytes: "graph-bytes", manifestInSubfolder: true);
        await File.WriteAllTextAsync(Path.Combine(snapshot, "tokenizer.json"), "{}", Ct);   // only at the root — a live file

        CacheManager.FindReclaimable(_cacheDir).Should().BeEmpty();
    }

    [Fact]
    public async Task Reclaim_ReChecksEveryEntry_AndRefusesPathsOutsideTheCache()
    {
        var snapshot = await SnapshotWithTwinAsync(rootBytes: "graph-bytes", subfolderBytes: "graph-bytes", manifestInSubfolder: true);
        var found = CacheManager.FindReclaimable(_cacheDir);
        var outside = Path.Combine(Path.GetTempPath(), "lmsupply-reclaim-outside-" + Guid.NewGuid().ToString("N") + ".bin");
        await File.WriteAllTextAsync(outside, "not yours", Ct);
        try
        {
            // The twin vanished after the list was computed: the root copy is now the only one and must stay.
            File.Delete(Path.Combine(snapshot, "onnx", "model.onnx"));

            var freed = CacheManager.Reclaim(_cacheDir, [found[0], found[0] with { Path = outside, TwinPath = outside }]);

            freed.Should().Be(0);
            File.Exists(Path.Combine(snapshot, "model.onnx")).Should().BeTrue("a stale entry whose twin is gone is not deleted");
            File.Exists(outside).Should().BeTrue("an entry outside the cache directory is refused");
        }
        finally
        {
            File.Delete(outside);
        }
    }

    [Fact]
    public async Task AfterReclaim_TheModelStillLoadsFromTheCacheWithoutANetworkRequest()
    {
        var snapshot = await SnapshotWithTwinAsync(rootBytes: "graph-bytes", subfolderBytes: "graph-bytes", manifestInSubfolder: true);
        CacheManager.Reclaim(_cacheDir, CacheManager.FindReclaimable(_cacheDir));
        var network = new NoNetworkHandler();
        using var downloader = new HuggingFaceDownloader(_cacheDir, network, localFilesOnly: true);

        var modelDir = await downloader.DownloadModelAsync(Repo, ["model.onnx"], subfolder: "onnx", cancellationToken: Ct);

        modelDir.Should().Be(Path.Combine(snapshot, "onnx"));
        (await File.ReadAllTextAsync(Path.Combine(modelDir, "model.onnx"), Ct)).Should().Be("graph-bytes");
        network.Requests.Should().Be(0, "what was deleted was not a file the loader reads");
    }

    private async Task<string> SnapshotWithTwinAsync(string rootBytes, string subfolderBytes, bool manifestInSubfolder, bool manifestAtRoot = false)
    {
        var snapshot = CacheManager.GetModelDirectory(_cacheDir, Repo);
        Directory.CreateDirectory(Path.Combine(snapshot, "onnx"));
        await File.WriteAllTextAsync(Path.Combine(snapshot, "model.onnx"), rootBytes, Ct);
        await File.WriteAllTextAsync(Path.Combine(snapshot, "onnx", "model.onnx"), subfolderBytes, Ct);

        if (manifestInSubfolder)
        {
            await DownloadManifest.WriteAsync(Path.Combine(snapshot, "onnx"), new DownloadManifest
            {
                Version = DownloadManifest.VerifiedVersion,
                RepoId = Repo,
                Revision = "main",
                Files = [new ManifestFileEntry { Path = "model.onnx", Size = subfolderBytes.Length }]
            });
        }

        if (manifestAtRoot)
        {
            await DownloadManifest.WriteAsync(snapshot, new DownloadManifest
            {
                Version = DownloadManifest.VerifiedVersion,
                RepoId = Repo,
                Revision = "main",
                Files = [new ManifestFileEntry { Path = "onnx/model.onnx", Size = subfolderBytes.Length }]
            });
        }

        return snapshot;
    }

    private sealed class NoNetworkHandler : HttpMessageHandler
    {
        public int Requests { get; private set; }

        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            Requests++;
            return Task.FromResult(new HttpResponseMessage(HttpStatusCode.NotFound));
        }
    }
}
