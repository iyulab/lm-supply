using AwesomeAssertions;
using LMSupply.Lifecycle;
using LMSupply.Llama.Server;

namespace LMSupply.Decider.Tests;

/// <summary>
/// <see cref="DeciderArtifacts"/> and the lifecycle of <see cref="LocalDecider"/>: the size, availability, download and
/// delete of the GGUF file a load picks from a repository of several quantizations, and of llama-server. Network-free —
/// a seeded listing and snapshot in a private Hugging Face cache, and a private llama-server cache.
/// </summary>
public sealed class DeciderArtifactsTests : IDisposable
{
    private const string Repo = "example-org/tiny-decider-GGUF";
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "lmsupply-decider-artifacts-" + Guid.NewGuid().ToString("N"));
    private string Hub => Path.Combine(_dir, "hub");
    private string ServerCache => Path.Combine(_dir, "llama-server");
    private string RepoDir => Path.Combine(Hub, "models--example-org--tiny-decider-GGUF");

    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch { /* best-effort temp cleanup */ }
    }

    private DeciderOptions Options(bool offline = false, string? quantization = null) => new()
    {
        Provider = ExecutionProvider.Cpu,
        CacheDirectory = Hub,
        DisableAutoDownload = offline,
        QuantizationHint = quantization,
        ServerUpdateOptions = new LlamaServerUpdateOptions { CacheDirectory = ServerCache, AutoDownloadUpdates = false },
    };

    private void SeedListing()
    {
        var listing = Path.Combine(RepoDir, ".lmsupply", "listings", "main.json");
        Directory.CreateDirectory(Path.GetDirectoryName(listing)!);
        File.WriteAllText(listing,
            """[{"path":"tiny-decider.Q4_K_M.gguf","type":"file","size":1000},{"path":"tiny-decider.Q8_0.gguf","type":"file","size":2000}]""");
    }

    private void SeedModel(int length = 1000)
    {
        var snapshot = Path.Combine(RepoDir, "snapshots", "main");
        Directory.CreateDirectory(snapshot);
        File.WriteAllBytes(Path.Combine(snapshot, "tiny-decider.Q4_K_M.gguf"), new byte[length]);
    }

    private void SeedServer()
    {
        var versionDir = Path.Combine(ServerCache, "b9898", "cpu");
        Directory.CreateDirectory(versionDir);
        File.WriteAllBytes(Path.Combine(versionDir, OperatingSystem.IsWindows() ? "llama-server.exe" : "llama-server"), "binary"u8.ToArray());
    }

    [Fact]
    public async Task The_size_is_the_quantization_the_load_picks_and_what_is_left_of_it()
    {
        SeedListing();

        (await LocalDecider.GetDownloadSizeBytesAsync(Repo, Options(), Ct)).Should().Be(1000, "Q4_K_M is the default");
        (await LocalDecider.GetDownloadSizeBytesAsync("gguf:" + Repo, Options(quantization: "Q8_0"), Ct)).Should().Be(2000);
        (await LocalDecider.GetRemainingDownloadBytesAsync(Repo, Options(), Ct)).Should().Be(1000);
        (await LocalDecider.IsModelDownloadedAsync(Repo, Options(), cancellationToken: TestContext.Current.CancellationToken)).Should().BeFalse();

        SeedModel(length: 400);
        (await LocalDecider.GetRemainingDownloadBytesAsync(Repo, Options(), Ct)).Should().Be(1000, "a partial file counts in full");

        SeedModel();
        (await LocalDecider.GetRemainingDownloadBytesAsync(Repo, Options(), Ct)).Should().Be(0);
        (await LocalDecider.IsModelDownloadedAsync(Repo, Options(), cancellationToken: TestContext.Current.CancellationToken)).Should().BeTrue();
    }

    [Fact]
    public async Task A_cached_model_is_available_offline_only_with_llama_server()
    {
        SeedListing();
        SeedModel();
        var artifacts = new DeciderArtifacts(Options());
        artifacts.Kind.Should().Be("decider");

        (await artifacts.IsAvailableOfflineAsync(Repo, Ct)).Should().BeFalse("llama-server is not cached");

        SeedServer();
        (await artifacts.IsAvailableOfflineAsync(Repo, Ct)).Should().BeTrue();

        // Downloads disabled: the download must be satisfied from the caches, so this proves nothing is fetched
        await new DeciderArtifacts(Options(offline: true)).DownloadAsync(Repo, cancellationToken: Ct);
    }

    [Fact]
    public async Task A_local_file_downloads_nothing_and_is_never_deleted()
    {
        var local = Path.Combine(_dir, "local.gguf");
        Directory.CreateDirectory(_dir);
        File.WriteAllBytes(local, [1, 2, 3]);

        (await LocalDecider.GetDownloadSizeBytesAsync(local, Options(), Ct)).Should().Be(0);
        (await LocalDecider.IsModelDownloadedAsync(local, Options(), cancellationToken: TestContext.Current.CancellationToken)).Should().BeTrue();
        LocalDecider.DeleteModel(local, Options()).Should().BeFalse();
        File.Exists(local).Should().BeTrue();
    }

    [Fact]
    public async Task Delete_removes_this_librarys_files_of_the_repository()
    {
        SeedListing();
        SeedModel();

        new DeciderArtifacts(Options()).Delete(Repo).Should().BeTrue();
        (await LocalDecider.IsModelDownloadedAsync(Repo, Options(), cancellationToken: TestContext.Current.CancellationToken)).Should().BeFalse();
        LocalDecider.DeleteModel(Repo, Options()).Should().BeFalse("nothing is left to delete");
    }

    [Fact]
    public async Task A_name_that_is_neither_a_file_nor_a_repository_is_not_downloaded()
    {
        (await LocalDecider.IsModelDownloadedAsync("not-a-model", Options(), cancellationToken: TestContext.Current.CancellationToken)).Should().BeFalse();
        Directory.Exists(Hub).Should().BeFalse("the check reads the cache only and writes nothing");
    }
}
