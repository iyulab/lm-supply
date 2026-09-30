using AwesomeAssertions;
using LMSupply.Download;
using LMSupply.Exceptions;

namespace LMSupply.Core.Tests.Download;

/// <summary>
/// The shared cache is also written by other Hugging Face tools, in the hub layout: <c>refs/main</c> names a commit and
/// <c>snapshots/{commit}/</c> holds its files (links into <c>blobs/</c>, or plain copies). Every lookup must find those
/// files — and keep finding the <c>snapshots/main/</c> directory this library writes — without modifying a snapshot it
/// does not own.
/// </summary>
public sealed class HubCacheLayoutTests : IDisposable
{
    private const string Repo = "acme/hub-model";
    private const string Commit = "0123456789abcdef0123456789abcdef01234567";

    private readonly string _cacheDir = Path.Combine(Path.GetTempPath(), "lmsupply-hub-" + Guid.NewGuid().ToString("N"));

    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    public void Dispose()
    {
        if (Directory.Exists(_cacheDir))
            Directory.Delete(_cacheDir, recursive: true);
    }

    private string RepoDir => Path.Combine(_cacheDir, "models--acme--hub-model");
    private string CommitSnapshot => Path.Combine(RepoDir, "snapshots", Commit);
    private string MainSnapshot => Path.Combine(RepoDir, "snapshots", "main");

    private void WriteRef(string revision, string content)
    {
        var path = Path.Combine(RepoDir, "refs", revision);
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        File.WriteAllText(path, content);
    }

    private static void WriteFile(string snapshot, string relativePath, string content)
    {
        var path = Path.Combine(snapshot, relativePath.Replace('/', Path.DirectorySeparatorChar));
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        File.WriteAllText(path, content);
    }

    // What a Python snapshot download leaves (with plain copies, as on Windows without link rights).
    private void SeedHubSnapshot()
    {
        WriteRef("main", Commit);
        WriteFile(CommitSnapshot, "onnx/model.onnx", "graph");
        WriteFile(CommitSnapshot, "tokenizer.json", "{}");
        WriteFile(CommitSnapshot, "config.json", "{}");
    }

    [Fact]
    public void HubSnapshot_FilesAreFound_ThroughTheRef()
    {
        SeedHubSnapshot();

        CacheManager.ModelFileExists(_cacheDir, Repo, "config.json").Should().BeTrue();
        CacheManager.ModelFileExists(_cacheDir, Repo, "onnx/model.onnx").Should().BeTrue();
        CacheManager.ModelFileExists(_cacheDir, Repo, "missing.json").Should().BeFalse();
        CacheManager.GetModelFilePath(_cacheDir, Repo, "config.json").Should().Be(Path.Combine(CommitSnapshot, "config.json"));
        CacheManager.GetSnapshotDirectories(_cacheDir, Repo).Should().Equal(CommitSnapshot);
    }

    [Fact]
    public void HubSnapshot_NothingIsMissing()
    {
        SeedHubSnapshot();

        CacheManager.GetMissingFiles(_cacheDir, Repo, ["model.onnx"], "onnx").Should().BeEmpty();
        CacheManager.GetMissingFiles(_cacheDir, Repo, ["tokenizer.json", "config.json"]).Should().BeEmpty();
        CacheManager.FindSnapshotDirectory(_cacheDir, Repo, ["model.onnx"], "onnx").Should().Be(Path.Combine(CommitSnapshot, "onnx"));
    }

    [Fact]
    public async Task HubSnapshot_ADownload_ReturnsItWithoutWritingAnything()
    {
        SeedHubSnapshot();
        using var downloader = new HuggingFaceDownloader(_cacheDir, localFilesOnly: false);

        // Online, but nothing to fetch: every requested file is in the hub snapshot, so no request is made.
        var dir = await downloader.DownloadModelAsync(Repo, ["tokenizer.json", "config.json"], cancellationToken: Ct);

        dir.Should().Be(CommitSnapshot);
        Directory.Exists(MainSnapshot).Should().BeFalse("a snapshot that holds the files is used where it is");
        Directory.EnumerateFiles(CommitSnapshot).Select(Path.GetFileName).Should().BeEquivalentTo(["tokenizer.json", "config.json"]);
    }

    [Fact]
    public async Task HubSnapshot_DiscoveryLoad_Offline_ListsTheSnapshotItself()
    {
        SeedHubSnapshot();
        using var downloader = new HuggingFaceDownloader(_cacheDir, localFilesOnly: true);

        var (dir, discovery) = await downloader.DownloadWithDiscoveryAsync(Repo, cancellationToken: Ct);

        dir.Should().Be(CommitSnapshot);
        discovery.GetAllFiles().Should().Contain("onnx/model.onnx");
        Directory.Exists(MainSnapshot).Should().BeFalse();
    }

    [Fact]
    public async Task HubSnapshot_ADownloadsPlan_IsTheSameFiles_SoTheLoadFetchesNothing()
    {
        SeedHubSnapshot();
        using var downloader = new HuggingFaceDownloader(_cacheDir, localFilesOnly: true);

        var plan = await downloader.PlanWithDiscoveryAsync(Repo, cancellationToken: Ct);

        plan.Files.Should().NotBeEmpty();
        CacheManager.GetMissingFiles(_cacheDir, Repo, plan.Files.Select(f => f.Path)).Should().BeEmpty();
    }

    // A file of another length than the listing is not the listed file: the hub snapshot is passed over, and it is
    // left exactly as it was — it belongs to another tool.
    [Fact]
    public async Task HubSnapshot_OfAnotherLength_IsNotUsed_AndNotTouched()
    {
        SeedHubSnapshot();
        const string listing = """
            [{"path":"onnx/model.onnx","type":"file","size":999},
             {"path":"tokenizer.json","type":"file","size":2},
             {"path":"config.json","type":"file","size":2}]
            """;
        var discoveryCache = Path.Combine(_cacheDir, ".discovery-cache");
        Directory.CreateDirectory(discoveryCache);
        await File.WriteAllTextAsync(Path.Combine(discoveryCache, "acme_hub-model_main.json"), listing, Ct);
        using var downloader = new HuggingFaceDownloader(_cacheDir, localFilesOnly: true);

        var load = () => downloader.DownloadWithDiscoveryAsync(Repo, cancellationToken: Ct);

        await load.Should().ThrowAsync<ModelNotFoundException>();
        File.ReadAllText(Path.Combine(CommitSnapshot, "onnx", "model.onnx")).Should().Be("graph");
    }

    [Fact]
    public void LibrarySnapshot_IsStillFound()
    {
        WriteFile(MainSnapshot, "config.json", "{}");

        CacheManager.ModelFileExists(_cacheDir, Repo, "config.json").Should().BeTrue();
        CacheManager.GetModelFilePath(_cacheDir, Repo, "config.json").Should().Be(Path.Combine(MainSnapshot, "config.json"));
        CacheManager.GetMissingFiles(_cacheDir, Repo, ["config.json"]).Should().BeEmpty();
        CacheManager.GetSnapshotDirectories(_cacheDir, Repo).Should().Equal(MainSnapshot);
    }

    [Fact]
    public async Task LibrarySnapshot_ADownload_StillReturnsIt()
    {
        WriteFile(MainSnapshot, "config.json", "{}");
        using var downloader = new HuggingFaceDownloader(_cacheDir, localFilesOnly: true);

        var dir = await downloader.DownloadModelAsync(Repo, ["config.json"], cancellationToken: Ct);

        dir.Should().Be(MainSnapshot);
    }

    // Both present: the snapshot the ref names comes first; a file only the library's snapshot has is still found.
    [Fact]
    public void BothSnapshots_TheRefWins_AndEitherCounts()
    {
        SeedHubSnapshot();
        WriteFile(MainSnapshot, "config.json", "{}");
        WriteFile(MainSnapshot, "extra.json", "{}");

        CacheManager.GetSnapshotDirectories(_cacheDir, Repo).Should().Equal(CommitSnapshot, MainSnapshot);
        CacheManager.GetModelFilePath(_cacheDir, Repo, "config.json").Should().Be(Path.Combine(CommitSnapshot, "config.json"));
        CacheManager.GetModelFilePath(_cacheDir, Repo, "extra.json").Should().Be(Path.Combine(MainSnapshot, "extra.json"));
        CacheManager.FindSnapshotDirectory(_cacheDir, Repo, ["config.json", "extra.json"]).Should().Be(MainSnapshot,
            "files a loader opens together come from one directory");
    }

    [Fact]
    public void ACommitRevision_NamesTheSnapshotOnce()
    {
        SeedHubSnapshot();

        CacheManager.GetSnapshotDirectories(_cacheDir, Repo, Commit).Should().Equal(CommitSnapshot);
        CacheManager.ModelFileExists(_cacheDir, Repo, "config.json", Commit).Should().BeTrue();
    }

    [Fact]
    public void ATagRef_IsFollowed()
    {
        SeedHubSnapshot();
        WriteRef("v1.0", Commit);

        CacheManager.ModelFileExists(_cacheDir, Repo, "config.json", "v1.0").Should().BeTrue();
    }

    [Theory]
    [InlineData("not a commit")]
    [InlineData("../../../elsewhere")]
    [InlineData("fedcba9876543210fedcba9876543210fedcba98")] // a commit whose snapshot is not in the cache
    public void ARefThatNamesNoSnapshot_IsIgnored(string content)
    {
        SeedHubSnapshot();
        WriteRef("main", content);

        CacheManager.ModelFileExists(_cacheDir, Repo, "config.json").Should().BeFalse();
        CacheManager.GetSnapshotDirectories(_cacheDir, Repo).Should().BeEmpty();
    }

    [Fact]
    public void ARevisionOutsideRefs_IsNotRead()
    {
        SeedHubSnapshot();

        CacheManager.GetSnapshotDirectories(_cacheDir, Repo, "../../models--other--repo/refs/main").Should().BeEmpty();
    }

    // The hub cache's snapshot entries are links into blobs/. A link's own length is not its content's: the
    // LFS-pointer test and every length comparison must see the blob.
    [Fact]
    public void ASymlinkedBlob_IsMeasuredAndReadThroughTheLink()
    {
        WriteRef("main", Commit);
        var content = new string('x', 4096);
        var blob = Path.Combine(RepoDir, "blobs", "a1b2c3");
        Directory.CreateDirectory(Path.GetDirectoryName(blob)!);
        File.WriteAllText(blob, content);
        var pointerBlob = Path.Combine(RepoDir, "blobs", "d4e5f6");
        File.WriteAllText(pointerBlob, "version https://git-lfs.github.com/spec/v1\noid sha256:0\nsize 1\n");

        Directory.CreateDirectory(CommitSnapshot);
        var link = Path.Combine(CommitSnapshot, "model.onnx");
        var pointerLink = Path.Combine(CommitSnapshot, "pointer.onnx");
        try
        {
            File.CreateSymbolicLink(link, Path.Combine("..", "..", "blobs", "a1b2c3"));
            File.CreateSymbolicLink(pointerLink, Path.Combine("..", "..", "blobs", "d4e5f6"));
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or PlatformNotSupportedException)
        {
            Assert.Skip($"This host cannot create symbolic links: {ex.Message}");
        }

        CacheManager.TryGetContentLength(link, out var length).Should().BeTrue();
        length.Should().Be(content.Length);
        CacheManager.IsLfsPointerFile(link).Should().BeFalse();
        CacheManager.IsLfsPointerFile(pointerLink).Should().BeTrue();
        CacheManager.ModelFileExists(_cacheDir, Repo, "model.onnx").Should().BeTrue();
        CacheManager.ModelFileExists(_cacheDir, Repo, "pointer.onnx").Should().BeFalse();
        ResumableFileDownload.IsUsableCachedFile(link, content.Length, readOnly: true).Should().BeTrue();
        ResumableFileDownload.IsCompleteFile(link, content.Length).Should().BeTrue();

        // A link whose blob is gone is not a cached file.
        File.Delete(blob);
        CacheManager.ModelFileExists(_cacheDir, Repo, "model.onnx").Should().BeFalse();
        CacheManager.TryGetContentLength(link, out _).Should().BeFalse();
    }
}
