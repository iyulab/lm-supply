using AwesomeAssertions;
using LMSupply.Llama.Server;
using Xunit;

namespace LMSupply.Reranker.Tests;

/// <summary>
/// <see cref="RerankerArtifacts"/> and <see cref="LocalReranker.DeleteModel"/>: a GGUF reranker is available offline when the
/// file an offline load opens and llama-server are both cached; a delete removes this library's snapshot, keeps another
/// tool's, and never touches a local file. Network-free — a seeded Hugging Face cache and a private llama-server cache.
/// </summary>
public sealed class RerankerArtifactsTests : IDisposable
{
    private const string Repo = "example-org/tiny-rerank-GGUF";
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "lmsupply-rrartifacts-" + Guid.NewGuid().ToString("N"));
    private readonly string _hub;
    private readonly string _serverCache;

    public RerankerArtifactsTests()
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

    private RerankerOptions Options(bool downloadsDisabled = false) => new()
    {
        Provider = ExecutionProvider.Cpu,
        CacheDirectory = _hub,
        DisableAutoDownload = downloadsDisabled,
        ServerUpdateOptions = new LlamaServerUpdateOptions { CacheDirectory = _serverCache, AutoDownloadUpdates = false },
    };

    // Another tool's layout: a snapshot under a commit, with refs/main pointing at it.
    private void SeedAnotherToolsGguf()
    {
        const string commit = "0123456789abcdef0123456789abcdef01234567";
        var repoDir = Path.Combine(_hub, "models--example-org--tiny-rerank-GGUF");
        Directory.CreateDirectory(Path.Combine(repoDir, "refs"));
        File.WriteAllText(Path.Combine(repoDir, "refs", "main"), commit);
        var snapshot = Path.Combine(repoDir, "snapshots", commit);
        Directory.CreateDirectory(snapshot);
        File.WriteAllBytes(Path.Combine(snapshot, $"tiny-rerank.{LocalReranker.DefaultGgufQuantization}.gguf"), "GGUF"u8.ToArray());
    }

    // This library's own layout: a snapshot named after the branch.
    private void SeedOwnGguf(string repoName)
    {
        var own = Path.Combine(_hub, $"models--example-org--{repoName}", "snapshots", "main");
        Directory.CreateDirectory(own);
        var fileName = $"{repoName.Replace("-GGUF", "")}.{LocalReranker.DefaultGgufQuantization}.gguf";
        File.WriteAllBytes(Path.Combine(own, fileName), "GGUF"u8.ToArray());
    }

    private void SeedServer()
    {
        var versionDir = Path.Combine(_serverCache, "b9898", "cpu");
        Directory.CreateDirectory(versionDir);
        File.WriteAllBytes(Path.Combine(versionDir, OperatingSystem.IsWindows() ? "llama-server.exe" : "llama-server"), "binary"u8.ToArray());
    }

    [Fact]
    public async Task AGgufReranker_IsAvailableOffline_OnlyWithItsFileAndServerCached()
    {
        var artifacts = new RerankerArtifacts(Options());
        artifacts.Kind.Should().Be("reranker");
        (await artifacts.IsAvailableOfflineAsync("gguf:" + Repo, Ct)).Should().BeFalse("nothing is cached");

        SeedAnotherToolsGguf();
        (await artifacts.IsAvailableOfflineAsync("gguf:" + Repo, Ct)).Should().BeFalse("llama-server is not cached");

        SeedServer();
        (await artifacts.IsAvailableOfflineAsync("gguf:" + Repo, Ct)).Should().BeTrue();
        (await artifacts.IsAvailableOfflineAsync(Repo, Ct)).Should().BeTrue("a repository named -GGUF is a GGUF model too");
    }

    [Fact]
    public async Task Download_WithEverythingCached_KeepsItAvailableOffline()
    {
        SeedAnotherToolsGguf();
        SeedServer();
        // Downloads disabled: the download must be satisfied from the cache, so this proves nothing is fetched.
        var artifacts = new RerankerArtifacts(Options(downloadsDisabled: true));

        await artifacts.DownloadAsync("gguf:" + Repo, cancellationToken: Ct);

        (await artifacts.IsAvailableOfflineAsync("gguf:" + Repo, Ct)).Should().BeTrue();
    }

    [Fact]
    public async Task Delete_RemovesThisLibrarysSnapshot_AndKeepsAnotherToolsOne()
    {
        SeedAnotherToolsGguf();
        SeedOwnGguf("other-rerank-GGUF");
        SeedServer();
        var artifacts = new RerankerArtifacts(Options());

        artifacts.Delete("gguf:example-org/other-rerank-GGUF").Should().BeTrue();
        artifacts.Delete("gguf:" + Repo).Should().BeFalse("only another tool's files are there");

        (await artifacts.IsAvailableOfflineAsync("gguf:example-org/other-rerank-GGUF", Ct)).Should().BeFalse();
        (await artifacts.IsAvailableOfflineAsync("gguf:" + Repo, Ct)).Should().BeTrue("another tool's snapshot is kept");
    }

    [Fact]
    public void Delete_OfALocalGgufFile_OrAnUncachedAlias_DeletesNothing()
    {
        var file = Path.Combine(_dir, "local.gguf");
        File.WriteAllBytes(file, "GGUF"u8.ToArray());

        LocalReranker.DeleteModel(file, new RerankerOptions { CacheDirectory = _hub }).Should().BeFalse();
        LocalReranker.DeleteModel("default", new RerankerOptions { CacheDirectory = _hub }).Should().BeFalse("nothing of the default model is cached");

        File.Exists(file).Should().BeTrue();
    }
}
