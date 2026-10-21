using AwesomeAssertions;
using LMSupply.Lifecycle;
using LMSupply.Testing;

namespace LMSupply.Detector.Tests;

/// <summary>
/// <see cref="DetectorArtifacts"/> and the lifecycle of <see cref="LocalDetector"/>: the size, availability, download and
/// delete of the file a load picks — here one of several models published in one repository — and of the ONNX Runtime it
/// runs on. Network-free: a seeded listing and snapshot in a private cache, and an application runtime directory.
/// </summary>
public sealed class DetectorArtifactsTests : IDisposable
{
    private const string Repo = "xnorpx/rt-detr2-onnx";
    private readonly string _cache = Path.Combine(Path.GetTempPath(), "lmsupply-detector-artifacts-" + Guid.NewGuid().ToString("N"));

    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    public void Dispose()
    {
        try { Directory.Delete(_cache + "-runtime", recursive: true); } catch { /* best-effort temp cleanup */ }
        try { Directory.Delete(_cache, recursive: true); } catch { /* best-effort temp cleanup */ }
    }

    private DetectorOptions Options() => new() { CacheDirectory = _cache, Provider = ExecutionProvider.Cpu };

    private string RepoDir => Path.Combine(_cache, "models--xnorpx--rt-detr2-onnx");

    // The repository listing a load caches: the default model's file and a larger sibling it does not take
    private void SeedListing()
    {
        var listing = Path.Combine(RepoDir, ".lmsupply", "listings", "main.json");
        Directory.CreateDirectory(Path.GetDirectoryName(listing)!);
        File.WriteAllText(listing, """[{"path":"rt-detrv2-s.onnx","type":"file","size":1000},{"path":"rt-detrv2-m.onnx","type":"file","size":2000}]""");
    }

    private void SeedModel(int length = 1000)
    {
        var snapshot = Path.Combine(RepoDir, "snapshots", "main");
        Directory.CreateDirectory(snapshot);
        File.WriteAllBytes(Path.Combine(snapshot, "rt-detrv2-s.onnx"), new byte[length]);
    }

    [Fact]
    public async Task The_size_is_the_file_the_load_picks_and_what_is_left_of_it()
    {
        SeedListing();

        (await LocalDetector.GetDownloadSizeBytesAsync("default", Options(), Ct)).Should().Be(1000, "the sibling model is not the default's");
        (await LocalDetector.GetRemainingDownloadBytesAsync("default", Options(), Ct)).Should().Be(1000);
        (await LocalDetector.IsModelDownloadedAsync("default", Options(), Ct)).Should().BeFalse();

        SeedModel(length: 400);
        (await LocalDetector.GetRemainingDownloadBytesAsync("default", Options(), Ct)).Should().Be(1000, "a partial file counts in full");

        SeedModel();
        (await LocalDetector.GetRemainingDownloadBytesAsync("default", Options(), Ct)).Should().Be(0);
        (await LocalDetector.IsModelDownloadedAsync("default", Options(), Ct)).Should().BeTrue();
    }

    [Fact]
    public async Task A_cached_model_is_available_offline_only_with_its_runtime()
    {
        SeedListing();
        SeedModel();
        var artifacts = new DetectorArtifacts(Options());
        artifacts.Kind.Should().Be("detector");

        using (BundledRuntime.Use(_cache + "-runtime", present: false))
        {
            (await artifacts.IsAvailableOfflineAsync("default", Ct)).Should().BeFalse();
            (await artifacts.GetRemainingDownloadAsync("default", Ct)).Should().Be(new RemainingDownload(0, null));
        }

        using (BundledRuntime.Use(_cache + "-runtime"))
        {
            (await artifacts.IsAvailableOfflineAsync("default", Ct)).Should().BeTrue();
            (await artifacts.GetRemainingDownloadAsync("default", Ct)).TotalBytes.Should().Be(0);

            // Downloads disabled: the download must be satisfied from the cache, so this proves nothing is fetched
            await new DetectorArtifacts(new DetectorOptions { CacheDirectory = _cache, Provider = ExecutionProvider.Cpu, DisableAutoDownload = true })
                .DownloadAsync("default", cancellationToken: Ct);
        }
    }

    [Fact]
    public async Task A_model_never_listed_into_this_cache_is_not_downloaded()
    {
        (await LocalDetector.IsModelDownloadedAsync("default", Options(), Ct)).Should().BeFalse();
        Directory.Exists(_cache).Should().BeFalse("the check reads the cache only and writes nothing");
    }

    [Fact]
    public void Delete_removes_this_librarys_snapshot_and_keeps_a_local_file()
    {
        SeedListing();
        SeedModel();
        var artifacts = new DetectorArtifacts(Options());

        artifacts.Delete("default").Should().BeTrue();
        artifacts.Delete("default").Should().BeFalse("nothing is left to delete");

        var local = Path.Combine(_cache + "-runtime", "local.onnx");
        Directory.CreateDirectory(Path.GetDirectoryName(local)!);
        File.WriteAllBytes(local, [1]);
        LocalDetector.DeleteModel(local, Options()).Should().BeFalse();
        File.Exists(local).Should().BeTrue();
    }
}
