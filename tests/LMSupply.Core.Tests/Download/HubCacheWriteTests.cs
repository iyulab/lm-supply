using System.Net;
using System.Security.Cryptography;
using System.Text;
using AwesomeAssertions;
using LMSupply.Download;

namespace LMSupply.Core.Tests.Download;

/// <summary>
/// Downloads write the Hugging Face hub cache layout — <c>blobs/{id}</c>, <c>snapshots/{commit}/</c> entries linked (or
/// moved) from the blobs, <c>refs/{revision}</c> naming the commit — so other Hugging Face tools find them. When the
/// revision cannot be resolved to a commit, the layout of earlier versions (<c>snapshots/{revision}</c>) is written
/// exactly as before. This library's own records stay out of the snapshots, and deleting a model removes only what this
/// library wrote.
/// </summary>
public sealed class HubCacheWriteTests : IDisposable
{
    private const string Repo = "acme/hub-write";
    private const string Commit = "89abcdef0123456789abcdef0123456789abcdef";
    private const string ConfigOid = "1111111111111111111111111111111111111111";
    private static readonly byte[] Model = Enumerable.Range(0, 4000).Select(i => (byte)(i % 239)).ToArray();
    private static readonly byte[] Config = "{\"a\":1}"u8.ToArray();
    private static readonly string ModelOid = Convert.ToHexStringLower(SHA256.HashData(Model));

    private readonly string _cacheDir = Path.Combine(Path.GetTempPath(), "lmsupply-hubwrite-" + Guid.NewGuid().ToString("N"));

    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    public void Dispose()
    {
        if (Directory.Exists(_cacheDir))
            Directory.Delete(_cacheDir, recursive: true);
    }

    private string RepoDir => Path.Combine(_cacheDir, "models--acme--hub-write");
    private string CommitSnapshot => Path.Combine(RepoDir, "snapshots", Commit);
    private string MainSnapshot => Path.Combine(RepoDir, "snapshots", "main");
    private string ModelBlob => Path.Combine(RepoDir, "blobs", ModelOid);
    private string ConfigBlob => Path.Combine(RepoDir, "blobs", ConfigOid);

    private static bool IsLink(string path) => new FileInfo(path).LinkTarget is not null;

    [Fact]
    public async Task ADownload_WithAResolvableCommit_WritesBlobsTheCommitSnapshotAndTheRef()
    {
        var hub = new Hub();
        using var downloader = new HuggingFaceDownloader(_cacheDir, hub);

        var dir = await downloader.DownloadModelAsync(Repo, ["model.onnx", "config.json"], cancellationToken: Ct);

        dir.Should().Be(CommitSnapshot);
        (await File.ReadAllBytesAsync(Path.Combine(dir, "model.onnx"), Ct)).Should().Equal(Model);
        (await File.ReadAllBytesAsync(Path.Combine(dir, "config.json"), Ct)).Should().Equal(Config);

        // Linked to its blob where links can be created; otherwise the blob was moved into the snapshot (one copy).
        var modelEntry = Path.Combine(dir, "model.onnx");
        if (IsLink(modelEntry))
        {
            File.Exists(ModelBlob).Should().BeTrue();
            new FileInfo(modelEntry).LinkTarget.Should().Be(Path.Combine("..", "..", "blobs", ModelOid));
            File.Exists(ConfigBlob).Should().BeTrue("a Git (non-LFS) file is named by its Git blob id");
        }
        else
        {
            File.Exists(ModelBlob).Should().BeFalse("without links the new blob is moved into the snapshot, not duplicated");
        }

        File.ReadAllText(Path.Combine(RepoDir, "refs", "main")).Should().Be(Commit, "the ref holds the commit id and nothing else");
        Directory.Exists(MainSnapshot).Should().BeFalse();
        CacheManager.GetSnapshotDirectories(_cacheDir, Repo).Should().Equal(CommitSnapshot);
        CacheManager.ModelFileExists(_cacheDir, Repo, "model.onnx").Should().BeTrue();
        hub.FileRequests.Should().OnlyContain(path => path.Contains($"/resolve/{Commit}/", StringComparison.Ordinal),
            "the files of a commit snapshot are fetched at that commit");
        Directory.EnumerateFiles(RepoDir, "*.part", SearchOption.AllDirectories).Should().BeEmpty();
    }

    [Fact]
    public async Task TheManifest_IsKeptOutsideTheSnapshots_AndMarksTheSnapshotAsOwn()
    {
        using var downloader = new HuggingFaceDownloader(_cacheDir, new Hub());

        var dir = await downloader.DownloadModelAsync(Repo, ["model.onnx", "config.json"], cancellationToken: Ct);

        Directory.EnumerateFiles(Path.Combine(RepoDir, "snapshots"), "*", SearchOption.AllDirectories)
            .Select(Path.GetFileName)
            .Should().BeEquivalentTo(["model.onnx", "config.json"], "a snapshot holds repository files only");
        File.Exists(Path.Combine(RepoDir, ".lmsupply", "manifests", Commit + ".json")).Should().BeTrue();

        var manifest = await DownloadManifest.ReadAsync(dir);
        manifest.Should().NotBeNull();
        manifest!.Version.Should().Be(DownloadManifest.VerifiedVersion);
        manifest.Files.Select(f => (f.Path, f.Size)).Should().BeEquivalentTo([("model.onnx", (long)Model.Length), ("config.json", (long)Config.Length)]);
        CacheManager.IsForeignSnapshot(_cacheDir, Repo, "main", dir).Should().BeFalse("this library downloaded into it");
    }

    [Fact]
    public async Task ASecondLoad_FindsTheCommitSnapshotThroughTheRef_WithoutARequest()
    {
        var hub = new Hub();
        using var downloader = new HuggingFaceDownloader(_cacheDir, hub);
        await downloader.DownloadModelAsync(Repo, ["model.onnx", "config.json"], cancellationToken: Ct);
        var before = hub.Requests;

        var dir = await downloader.DownloadModelAsync(Repo, ["model.onnx", "config.json"], cancellationToken: Ct);

        dir.Should().Be(CommitSnapshot);
        hub.Requests.Should().Be(before, "a warm load makes no request, the commit lookup included");
    }

    [Fact]
    public async Task ASubfolderDownload_WritesItsManifestUnderTheSubfolderName()
    {
        using var downloader = new HuggingFaceDownloader(_cacheDir, new Hub());

        var dir = await downloader.DownloadModelAsync(Repo, ["model.onnx"], subfolder: "onnx/int8", cancellationToken: Ct);

        dir.Should().Be(Path.Combine(CommitSnapshot, "onnx", "int8"));
        File.Exists(Path.Combine(RepoDir, ".lmsupply", "manifests", Commit + "__onnx_int8.json")).Should().BeTrue();
        (await DownloadManifest.ReadAsync(dir))!.Files.Should().ContainSingle(f => f.Path == "model.onnx");
        CacheManager.DeleteModel(_cacheDir, Repo).Should().BeTrue();
        Directory.Exists(RepoDir).Should().BeFalse("everything in it was this library's");
    }

    // Positive control: with no commit to be had, nothing about the layout changes.
    [Theory]
    [InlineData(HttpStatusCode.NotFound)]
    [InlineData(HttpStatusCode.ServiceUnavailable)]
    public async Task WhenTheRevisionCannotBeResolved_TheLayoutOfEarlierVersionsIsWritten(HttpStatusCode revisionStatus)
    {
        var hub = new Hub { RevisionStatus = revisionStatus };
        using var downloader = new HuggingFaceDownloader(_cacheDir, hub);

        var dir = await downloader.DownloadModelAsync(Repo, ["model.onnx", "config.json"], cancellationToken: Ct);

        dir.Should().Be(MainSnapshot);
        IsLink(Path.Combine(dir, "model.onnx")).Should().BeFalse();
        (await File.ReadAllBytesAsync(Path.Combine(dir, "model.onnx"), Ct)).Should().Equal(Model);
        File.Exists(Path.Combine(dir, ".lmsupply-manifest.json")).Should().BeTrue("a snapshot named after the revision keeps its manifest inside, as before");
        Directory.Exists(Path.Combine(RepoDir, "blobs")).Should().BeFalse();
        Directory.Exists(Path.Combine(RepoDir, "refs")).Should().BeFalse();
        Directory.Exists(Path.Combine(RepoDir, ".lmsupply")).Should().BeFalse();
        hub.FileRequests.Should().OnlyContain(path => path.Contains("/resolve/main/", StringComparison.Ordinal));
    }

    [Fact]
    public async Task WhenTheRevisionLookupThrows_TheLayoutOfEarlierVersionsIsWritten()
    {
        using var downloader = new HuggingFaceDownloader(_cacheDir, new Hub { RevisionThrows = true });

        var dir = await downloader.DownloadModelAsync(Repo, ["model.onnx"], cancellationToken: Ct);

        dir.Should().Be(MainSnapshot);
        Directory.Exists(Path.Combine(RepoDir, "blobs")).Should().BeFalse();
    }

    [Fact]
    public async Task ABlobAnotherToolDownloaded_IsReused_WithoutAFileRequest()
    {
        Directory.CreateDirectory(Path.GetDirectoryName(ModelBlob)!);
        await File.WriteAllBytesAsync(ModelBlob, Model, Ct);
        var hub = new Hub();
        using var downloader = new HuggingFaceDownloader(_cacheDir, hub);

        var dir = await downloader.DownloadModelAsync(Repo, ["model.onnx"], cancellationToken: Ct);

        (await File.ReadAllBytesAsync(Path.Combine(dir, "model.onnx"), Ct)).Should().Equal(Model);
        hub.FileRequests.Should().BeEmpty("the blob is already in the cache");
        File.Exists(ModelBlob).Should().BeTrue("a blob that was already there stays where others find it (linked, or copied)");
    }

    [Fact]
    public async Task ADiscoveryDownload_WritesTheHubLayoutToo()
    {
        var hub = new Hub();
        using var downloader = new HuggingFaceDownloader(_cacheDir, hub);

        var (dir, discovery) = await downloader.DownloadWithDiscoveryAsync(Repo, cancellationToken: Ct);

        dir.Should().Be(CommitSnapshot);
        var graph = discovery.GetAllFiles().Single(f => f.EndsWith(".onnx", StringComparison.Ordinal));
        (await File.ReadAllBytesAsync(Path.Combine(dir, graph.Replace('/', Path.DirectorySeparatorChar)), Ct)).Should().Equal(Model);
        File.ReadAllText(Path.Combine(RepoDir, "refs", "main")).Should().Be(Commit);
        File.Exists(Path.Combine(RepoDir, ".lmsupply", "manifests", Commit + ".json")).Should().BeTrue();
        File.Exists(Path.Combine(dir, ".lmsupply-manifest.json")).Should().BeFalse();
    }

    [Fact]
    public void AManifestInsideASnapshot_FromAnEarlierVersion_IsStillRead()
    {
        Directory.CreateDirectory(MainSnapshot);
        File.WriteAllBytes(Path.Combine(MainSnapshot, "model.onnx"), Model);
        File.WriteAllText(Path.Combine(MainSnapshot, ".lmsupply-manifest.json"),
            """{"version":2,"repoId":"acme/hub-write","files":[{"path":"model.onnx","size":4000}]}""");

        DownloadManifest.Read(MainSnapshot)!.Files.Should().ContainSingle(f => f.Path == "model.onnx" && f.Size == 4000);
        ModelDirectoryValidator.Validate(MainSnapshot).IsValid.Should().BeTrue();
    }

    [Fact]
    public void ACommitSnapshotWithAnEarlierVersionsManifestInside_IsOwn_AndOneWithoutIsForeign()
    {
        var refs = Path.Combine(RepoDir, "refs");
        Directory.CreateDirectory(refs);
        File.WriteAllText(Path.Combine(refs, "main"), Commit);
        Directory.CreateDirectory(CommitSnapshot);
        File.WriteAllBytes(Path.Combine(CommitSnapshot, "model.onnx"), Model);

        CacheManager.IsForeignSnapshot(_cacheDir, Repo, "main", CommitSnapshot).Should().BeTrue();

        File.WriteAllText(Path.Combine(CommitSnapshot, ".lmsupply-manifest.json"), """{"version":1,"files":[]}""");

        CacheManager.IsForeignSnapshot(_cacheDir, Repo, "main", CommitSnapshot).Should().BeFalse();
        CacheManager.IsForeignSnapshot(_cacheDir, Repo, "main", MainSnapshot).Should().BeFalse("a snapshot named after the revision is written by this library only");
    }

    [Fact]
    public async Task DeleteModel_RemovesTheOwnSnapshotItsBlobsAndRef_AndKeepsAnotherToolsSnapshotAndTheBlobItLinks()
    {
        using (var downloader = new HuggingFaceDownloader(_cacheDir, new Hub()))
            await downloader.DownloadModelAsync(Repo, ["model.onnx", "config.json"], cancellationToken: Ct);
        if (!IsLink(Path.Combine(CommitSnapshot, "config.json")))
            Assert.Skip("This host cannot create symbolic links, so no blob is shared between snapshots.");

        // Another tool's snapshot of another commit: a link to the shared config blob, and a blob of its own.
        const string otherCommit = "fedcba9876543210fedcba9876543210fedcba98";
        var other = Path.Combine(RepoDir, "snapshots", otherCommit);
        Directory.CreateDirectory(other);
        File.CreateSymbolicLink(Path.Combine(other, "config.json"), Path.Combine("..", "..", "blobs", ConfigOid));
        var otherBlob = Path.Combine(RepoDir, "blobs", new string('e', 64));
        File.WriteAllText(otherBlob, "weights");
        File.CreateSymbolicLink(Path.Combine(other, "weights.bin"), Path.Combine("..", "..", "blobs", new string('e', 64)));
        File.WriteAllText(Path.Combine(RepoDir, "refs", "v1"), otherCommit);

        CacheManager.DeleteModel(_cacheDir, Repo).Should().BeTrue();

        Directory.Exists(CommitSnapshot).Should().BeFalse();
        File.Exists(ModelBlob).Should().BeFalse("only the deleted snapshot linked to it");
        File.Exists(Path.Combine(RepoDir, "refs", "main")).Should().BeFalse();
        Directory.Exists(Path.Combine(RepoDir, ".lmsupply")).Should().BeFalse();

        File.Exists(ConfigBlob).Should().BeTrue("the other tool's snapshot still links to it");
        File.Exists(otherBlob).Should().BeTrue();
        File.ReadAllText(Path.Combine(other, "config.json")).Should().Be("{\"a\":1}");
        File.ReadAllText(Path.Combine(RepoDir, "refs", "v1")).Should().Be(otherCommit);
    }

    [Fact]
    public void DeleteModel_OfACacheHoldingOnlyAnotherToolsSnapshot_DeletesNothing()
    {
        var refs = Path.Combine(RepoDir, "refs");
        Directory.CreateDirectory(refs);
        File.WriteAllText(Path.Combine(refs, "main"), Commit);
        Directory.CreateDirectory(CommitSnapshot);
        File.WriteAllText(Path.Combine(CommitSnapshot, "config.json"), "{}");

        CacheManager.DeleteModel(_cacheDir, Repo).Should().BeFalse();

        File.Exists(Path.Combine(CommitSnapshot, "config.json")).Should().BeTrue();
        File.Exists(Path.Combine(refs, "main")).Should().BeTrue();
    }

    [Fact]
    public void DeleteModel_InACommitSnapshotSharedWithAnotherTool_DeletesOnlyTheListedFiles()
    {
        Directory.CreateDirectory(CommitSnapshot);
        File.WriteAllBytes(Path.Combine(CommitSnapshot, "model.onnx"), Model);
        File.WriteAllText(Path.Combine(CommitSnapshot, "README.md"), "added by another tool");
        var manifests = Path.Combine(RepoDir, ".lmsupply", "manifests");
        Directory.CreateDirectory(manifests);
        File.WriteAllText(Path.Combine(manifests, Commit + ".json"), """{"version":2,"files":[{"path":"model.onnx","size":4000}]}""");

        CacheManager.DeleteModel(_cacheDir, Repo).Should().BeTrue();

        File.Exists(Path.Combine(CommitSnapshot, "model.onnx")).Should().BeFalse();
        File.ReadAllText(Path.Combine(CommitSnapshot, "README.md")).Should().Be("added by another tool");
    }

    [Fact]
    public void LinkOrMove_WithoutLinks_MovesANewBlob_AndCopiesOneThatWasAlreadyThere()
    {
        var blob = Path.Combine(RepoDir, "blobs", ModelOid);
        Directory.CreateDirectory(Path.GetDirectoryName(blob)!);
        File.WriteAllBytes(blob, Model);
        static void NoLinks(string link, string target) => throw new IOException("A required privilege is not held by the client.");

        var pointer = Path.Combine(CommitSnapshot, "onnx", "model.onnx");
        HubCache.LinkOrMove(blob, pointer, newBlob: false, NoLinks).Should().BeFalse();
        File.ReadAllBytes(pointer).Should().Equal(Model);
        File.Exists(blob).Should().BeTrue("a blob that was already there is copied, not taken");

        var second = Path.Combine(CommitSnapshot, "model.onnx");
        HubCache.LinkOrMove(blob, second, newBlob: true, NoLinks).Should().BeFalse();
        File.ReadAllBytes(second).Should().Equal(Model);
        File.Exists(blob).Should().BeFalse("a blob this download fetched is moved, so there is one copy on disk");
    }

    [Fact]
    public void LinkOrMove_LinksRelativeToTheEntrysDepth()
    {
        var blob = Path.Combine(RepoDir, "blobs", ModelOid);
        Directory.CreateDirectory(Path.GetDirectoryName(blob)!);
        File.WriteAllBytes(blob, Model);
        var pointer = Path.Combine(CommitSnapshot, "onnx", "int8", "model.onnx");

        if (!HubCache.LinkOrMove(blob, pointer, newBlob: true))
            Assert.Skip("This host cannot create symbolic links.");

        new FileInfo(pointer).LinkTarget.Should().Be(Path.Combine("..", "..", "..", "..", "blobs", ModelOid));
        File.ReadAllBytes(pointer).Should().Equal(Model);
        CacheManager.GetTotalCacheSize(_cacheDir).Should().Be(Model.Length, "a link counts as its blob, once");
    }

    [Fact]
    public void BlobIds_AreTheLfsShaForLfsFiles_TheGitOidOtherwise_AndUnknownForAListingWithoutLfsEntries()
    {
        var gitOid = new string('2', 40);
        var lfsOid = new string('a', 64);
        var withLfs = new[]
        {
            new LMSupply.Core.Download.RepoFile { Path = "model.onnx", Type = "file", Size = 4000, Oid = gitOid, Lfs = new() { Oid = lfsOid, Size = 4000 } },
            new LMSupply.Core.Download.RepoFile { Path = "config.json", Type = "file", Size = 2, Oid = gitOid },
        };
        HubCache.BlobIdsOf(withLfs).Should().BeEquivalentTo(new Dictionary<string, string> { ["model.onnx"] = lfsOid, ["config.json"] = gitOid });

        // A listing cached before LFS entries were recorded cannot tell an LFS file from a Git one: no blob is named.
        var withoutLfs = new[] { new LMSupply.Core.Download.RepoFile { Path = "model.onnx", Type = "file", Size = 4000, Oid = gitOid } };
        HubCache.BlobIdsOf(withoutLfs).Should().BeEmpty();
    }

    [Fact]
    public void TheTreeListing_MapsTheLfsEntry()
    {
        var json = $$"""[{"type":"file","oid":"{{new string('2', 40)}}","size":9,"lfs":{"oid":"{{new string('a', 64)}}","size":9,"pointerSize":133},"path":"m.onnx"}]""";

        var files = System.Text.Json.JsonSerializer.Deserialize(json, LMSupply.Json.CoreJsonContext.Default.ListRepoFile)!;

        files.Should().ContainSingle().Which.BlobId.Should().Be(new string('a', 64));
    }

    /// <summary>A hub serving the tree, the revision and the two files; it records every request.</summary>
    private sealed class Hub : HttpMessageHandler
    {
        private readonly List<string> _fileRequests = [];
        private int _requests;

        public HttpStatusCode RevisionStatus { get; init; } = HttpStatusCode.OK;
        public bool RevisionThrows { get; init; }

        public int Requests => Volatile.Read(ref _requests);
        public IReadOnlyList<string> FileRequests { get { lock (_fileRequests) return [.. _fileRequests]; } }

        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            Interlocked.Increment(ref _requests);
            var path = Uri.UnescapeDataString(request.RequestUri!.AbsolutePath);

            if (path == $"/api/models/{Repo}/tree/main")
            {
                return Task.FromResult(Json($$$"""
                    [{"path":"model.onnx","type":"file","size":{{{Model.Length}}},"oid":"3333333333333333333333333333333333333333",
                      "lfs":{"oid":"{{{ModelOid}}}","size":{{{Model.Length}}},"pointerSize":134}},
                     {"path":"onnx/int8/model.onnx","type":"file","size":{{{Model.Length}}},"oid":"3333333333333333333333333333333333333333",
                      "lfs":{"oid":"{{{ModelOid}}}","size":{{{Model.Length}}},"pointerSize":134}},
                     {"path":"config.json","type":"file","size":{{{Config.Length}}},"oid":"{{{ConfigOid}}}"}]
                    """));
            }

            if (path == $"/api/models/{Repo}/revision/main")
            {
                if (RevisionThrows)
                    throw new HttpRequestException("No such host is known.");
                return Task.FromResult(RevisionStatus == HttpStatusCode.OK
                    ? Json($$"""{"id":"{{Repo}}","sha":"{{Commit}}"}""")
                    : new HttpResponseMessage(RevisionStatus));
            }

            foreach (var revision in new[] { "main", Commit })
            {
                var prefix = $"/{Repo}/resolve/{revision}/";
                if (!path.StartsWith(prefix, StringComparison.Ordinal))
                    continue;

                lock (_fileRequests) _fileRequests.Add(path);
                var file = path[prefix.Length..];
                byte[]? body = file switch
                {
                    "model.onnx" or "onnx/int8/model.onnx" => Model,
                    "config.json" => Config,
                    _ => null,
                };
                return Task.FromResult(body is null
                    ? new HttpResponseMessage(HttpStatusCode.NotFound)
                    : new HttpResponseMessage(HttpStatusCode.OK) { Content = new ByteArrayContent(body) });
            }

            return Task.FromResult(new HttpResponseMessage(HttpStatusCode.NotFound));
        }

        private static HttpResponseMessage Json(string body) =>
            new(HttpStatusCode.OK) { Content = new StringContent(body, Encoding.UTF8, "application/json") };
    }
}
