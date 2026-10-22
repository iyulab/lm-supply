using AwesomeAssertions;
using LMSupply.Lifecycle;
using LMSupply.Testing;

namespace LMSupply.ImageGenerator.Tests;

/// <summary>
/// <see cref="ImageGeneratorArtifacts"/> and the lifecycle of <see cref="LocalImageGenerator"/>: the size, availability,
/// download and delete of the pipeline's files — its ONNX models and the config and tokenizer files beside them — and of the
/// ONNX Runtime it runs on. Network-free: a seeded listing and snapshot in a private cache, and an application runtime
/// directory.
/// </summary>
public sealed class ImageGeneratorArtifactsTests : IDisposable
{
    private const string Repo = "TheyCallMeHex/LCM-Dreamshaper-V7-ONNX";
    private readonly string _cache = Path.Combine(Path.GetTempPath(), "lmsupply-imagegen-artifacts-" + Guid.NewGuid().ToString("N"));

    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    public void Dispose()
    {
        try { Directory.Delete(_cache + "-runtime", recursive: true); } catch { /* best-effort temp cleanup */ }
        try { Directory.Delete(_cache, recursive: true); } catch { /* best-effort temp cleanup */ }
    }

    private ImageGeneratorOptions Options(bool offline = false) =>
        new() { CacheDirectory = _cache, Provider = ExecutionProvider.Cpu, DisableAutoDownload = offline };

    private string RepoDir => Path.Combine(_cache, "models--" + Repo.Replace("/", "--"));

    private const long Models = 5_000 + 1_000 + 800;
    private const long Supporting = 30 + 20 + 5 + 5;

    private static readonly (string Path, long Size)[] Listing =
    [
        ("unet/model.onnx", 5_000),
        ("text_encoder/model.onnx", 1_000),
        ("vae_decoder/model.onnx", 800),
        ("tokenizer/vocab.json", 30),
        ("tokenizer/merges.txt", 20),
        ("model_index.json", 5),
        ("scheduler/scheduler_config.json", 5),
    ];

    private void SeedListing()
    {
        var path = Path.Combine(RepoDir, ".lmsupply", "listings", "main.json");
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        File.WriteAllText(path, "[" + string.Join(",", Listing.Select(f => $$"""{"path":"{{f.Path}}","type":"file","size":{{f.Size}}}""")) + "]");
    }

    private void SeedSnapshot(long? unetLength = null)
    {
        foreach (var (file, size) in Listing)
        {
            var path = Path.Combine(RepoDir, "snapshots", "main", file);
            Directory.CreateDirectory(Path.GetDirectoryName(path)!);
            File.WriteAllBytes(path, new byte[file == "unet/model.onnx" ? unetLength ?? size : size]);
        }
    }

    [Fact]
    public async Task The_size_is_the_pipelines_files_and_what_is_left_of_them()
    {
        SeedListing();

        var size = await LocalImageGenerator.GetDownloadSizeBytesAsync("default", Options(), Ct);
        size.Should().BeInRange(Models, Models + Supporting, "the three models are the bulk of what a load fetches");
        (await LocalImageGenerator.GetRemainingDownloadBytesAsync("default", Options(), Ct)).Should().Be(size);
        (await LocalImageGenerator.IsModelDownloadedAsync("default", Options(), Ct)).Should().BeFalse();

        SeedSnapshot(unetLength: 100);
        (await LocalImageGenerator.GetRemainingDownloadBytesAsync("default", Options(), Ct)).Should().Be(5_000, "a partial file counts in full");

        SeedSnapshot();
        (await LocalImageGenerator.GetRemainingDownloadBytesAsync("default", Options(), Ct)).Should().Be(0);
        (await LocalImageGenerator.IsModelDownloadedAsync("default", Options(), Ct)).Should().BeTrue();
    }

    [Fact]
    public async Task A_cached_pipeline_is_available_offline_only_with_its_runtime()
    {
        SeedListing();
        SeedSnapshot();
        var artifacts = new ImageGeneratorArtifacts(Options());
        artifacts.Kind.Should().Be("imagegenerator");

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
            await new ImageGeneratorArtifacts(Options(offline: true)).DownloadAsync("default", cancellationToken: Ct);
        }
    }

    [Fact]
    public async Task A_pipeline_never_listed_into_this_cache_is_not_downloaded()
    {
        (await LocalImageGenerator.IsModelDownloadedAsync("default", Options(), Ct)).Should().BeFalse();
        Directory.Exists(_cache).Should().BeFalse("the check reads the cache only and writes nothing");
    }

    [Fact]
    public async Task Delete_removes_this_librarys_snapshot_and_keeps_a_local_pipeline()
    {
        SeedListing();
        SeedSnapshot();
        var artifacts = new ImageGeneratorArtifacts(Options());

        artifacts.Delete("default").Should().BeTrue();
        artifacts.Delete("default").Should().BeFalse("nothing is left to delete");

        var local = Path.Combine(_cache + "-runtime", "my-pipeline");
        Directory.CreateDirectory(local);
        (await LocalImageGenerator.IsModelDownloadedAsync(local, Options(), Ct)).Should().BeTrue();
        LocalImageGenerator.DeleteModel(local, Options()).Should().BeFalse();
        Directory.Exists(local).Should().BeTrue();
    }
}
