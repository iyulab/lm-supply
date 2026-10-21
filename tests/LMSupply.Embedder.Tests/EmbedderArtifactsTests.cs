using AwesomeAssertions;
using LMSupply.Lifecycle;
using LMSupply.Llama.Server;
using Xunit;

namespace LMSupply.Embedder.Tests;

/// <summary>
/// <see cref="EmbedderArtifacts"/> and the GGUF branch of <see cref="LocalEmbedder.IsModelDownloaded"/>: a GGUF embedder is
/// downloaded when the file an offline load opens is cached, and available offline when llama-server is cached too; a delete
/// removes this library's snapshot and keeps another tool's.
/// Network-free — a seeded Hugging Face cache and a private llama-server cache.
/// </summary>
public sealed class EmbedderArtifactsTests : IDisposable
{
    private const string Repo = "example-org/tiny-embed-GGUF";
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "lmsupply-embartifacts-" + Guid.NewGuid().ToString("N"));
    private readonly string _hub;
    private readonly string _serverCache;

    public EmbedderArtifactsTests()
    {
        _hub = Path.Combine(_dir, "hub");
        _serverCache = Path.Combine(_dir, "llama-server");
        Directory.CreateDirectory(_hub);
        Directory.CreateDirectory(_serverCache);
    }

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch { /* best-effort temp cleanup */ }
        GC.SuppressFinalize(this);
    }

    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    private EmbedderArtifacts Artifacts() => new(new EmbedderOptions
    {
        Provider = ExecutionProvider.Cpu,
        CacheDirectory = _hub,
        ServerUpdateOptions = new LlamaServerUpdateOptions { CacheDirectory = _serverCache, AutoDownloadUpdates = false },
    });

    private void SeedGguf()
    {
        const string commit = "0123456789abcdef0123456789abcdef01234567";
        var repoDir = Path.Combine(_hub, "models--example-org--tiny-embed-GGUF");
        Directory.CreateDirectory(Path.Combine(repoDir, "refs"));
        File.WriteAllText(Path.Combine(repoDir, "refs", "main"), commit);
        var snapshot = Path.Combine(repoDir, "snapshots", commit);
        Directory.CreateDirectory(snapshot);
        File.WriteAllBytes(Path.Combine(snapshot, "tiny-embed.Q8_0.gguf"), "GGUF"u8.ToArray());
    }

    private void SeedServer()
    {
        var versionDir = Path.Combine(_serverCache, "b9898", "cpu");
        Directory.CreateDirectory(versionDir);
        File.WriteAllBytes(Path.Combine(versionDir, OperatingSystem.IsWindows() ? "llama-server.exe" : "llama-server"), "binary"u8.ToArray());
    }

    [Fact]
    public void ACachedGgufRepository_IsDownloaded_AndAMissingOneIsNot()
    {
        LocalEmbedder.IsModelDownloaded("gguf:" + Repo, _hub).Should().BeFalse();

        SeedGguf();

        LocalEmbedder.IsModelDownloaded("gguf:" + Repo, _hub).Should().BeTrue("the file an offline load opens is cached");
        LocalEmbedder.IsModelDownloaded(Repo, _hub).Should().BeTrue("a repository named -GGUF is a GGUF model too");
    }

    [Fact]
    public async Task AGgufEmbedder_IsAvailableOffline_OnlyWithItsServerCached()
    {
        SeedGguf();
        var artifacts = Artifacts();
        (await artifacts.IsAvailableOfflineAsync("gguf:" + Repo, Ct)).Should().BeFalse();

        SeedServer();

        (await artifacts.IsAvailableOfflineAsync("gguf:" + Repo, Ct)).Should().BeTrue();
        artifacts.Kind.Should().Be("embedder");
    }

    [Fact]
    public async Task AGgufEmbedder_WithItsFileAndServerCached_HasNothingLeftToDownload()
    {
        SeedGguf();
        SeedServer();
        // The repository listing the size is read from, as a load caches it: the cached file at its listed length
        var listing = Path.Combine(_hub, "models--example-org--tiny-embed-GGUF", ".lmsupply", "listings", "main.json");
        Directory.CreateDirectory(Path.GetDirectoryName(listing)!);
        File.WriteAllText(listing, """[{"path":"tiny-embed.Q8_0.gguf","type":"file","size":4}]""");

        var remaining = await Artifacts().GetRemainingDownloadAsync("gguf:" + Repo, Ct);

        remaining.Should().Be(new RemainingDownload(0, 0), "a CPU server needs no companion, so the cache answers alone");
        remaining.TotalBytes.Should().Be(0);
    }

    [Fact]
    public async Task Download_WithEverythingCached_KeepsItAvailableOffline()
    {
        SeedGguf();
        SeedServer();
        var artifacts = new EmbedderArtifacts(new EmbedderOptions
        {
            Provider = ExecutionProvider.Cpu,
            CacheDirectory = _hub,
            // Downloads disabled: the download must be satisfied from the cache, so this proves nothing is fetched.
            DisableAutoDownload = true,
            ServerUpdateOptions = new LlamaServerUpdateOptions { CacheDirectory = _serverCache, AutoDownloadUpdates = false },
        });

        await artifacts.DownloadAsync("gguf:" + Repo, cancellationToken: Ct);

        (await artifacts.IsAvailableOfflineAsync("gguf:" + Repo, Ct)).Should().BeTrue();
    }

    [Fact]
    public async Task Delete_RemovesThisLibrarysSnapshot_AndKeepsAnotherToolsOne()
    {
        // Another tool's snapshot (a commit with no manifest) stays; a snapshot named after a branch is this library's own.
        SeedGguf();
        var own = Path.Combine(_hub, "models--example-org--other-embed-GGUF", "snapshots", "main");
        Directory.CreateDirectory(own);
        File.WriteAllBytes(Path.Combine(own, "other-embed.Q8_0.gguf"), "GGUF"u8.ToArray());
        SeedServer();
        var artifacts = Artifacts();

        artifacts.Delete("gguf:example-org/other-embed-GGUF").Should().BeTrue();
        artifacts.Delete("gguf:" + Repo).Should().BeFalse("only another tool's files are there");

        (await artifacts.IsAvailableOfflineAsync("gguf:example-org/other-embed-GGUF", Ct)).Should().BeFalse();
        (await artifacts.IsAvailableOfflineAsync("gguf:" + Repo, Ct)).Should().BeTrue("another tool's snapshot is kept");
    }

    [Fact]
    public void Delete_OfALocalGgufFile_KeepsIt()
    {
        var file = Path.Combine(_dir, "local.gguf");
        File.WriteAllBytes(file, "GGUF"u8.ToArray());

        LocalEmbedder.DeleteModel(file, _hub).Should().BeFalse();

        File.Exists(file).Should().BeTrue();
    }
}
