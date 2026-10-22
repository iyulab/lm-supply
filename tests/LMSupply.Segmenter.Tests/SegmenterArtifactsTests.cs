using AwesomeAssertions;
using LMSupply.Lifecycle;
using LMSupply.Testing;

namespace LMSupply.Segmenter.Tests;

/// <summary>
/// <see cref="SegmenterArtifacts"/> and the lifecycle of <see cref="LocalSegmenter"/>, for both kinds of load: a semantic
/// segmenter takes the one file it picks from its repository, an interactive one its encoder, decoder and configuration.
/// Network-free: seeded listings and snapshots in a private cache, and an application runtime directory.
/// </summary>
public sealed class SegmenterArtifactsTests : IDisposable
{
    private readonly string _cache = Path.Combine(Path.GetTempPath(), "lmsupply-segmenter-artifacts-" + Guid.NewGuid().ToString("N"));

    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    public void Dispose()
    {
        try { Directory.Delete(_cache, recursive: true); } catch { /* best-effort temp cleanup */ }
    }

    private SegmenterOptions Options() => new() { CacheDirectory = _cache, Provider = ExecutionProvider.Cpu };

    private string RepoDir(string repo) => Path.Combine(_cache, "models--" + repo.Replace("/", "--"));

    private void SeedListing(string repo, params (string Path, int Size)[] files)
    {
        var listing = Path.Combine(RepoDir(repo), ".lmsupply", "listings", "main.json");
        Directory.CreateDirectory(Path.GetDirectoryName(listing)!);
        File.WriteAllText(listing, "[" + string.Join(",", files.Select(f => $$"""{"path":"{{f.Path}}","type":"file","size":{{f.Size}}}""")) + "]");
    }

    private void SeedFile(string repo, string path, int length)
    {
        var snapshot = Path.Combine(RepoDir(repo), "snapshots", "main");
        Directory.CreateDirectory(snapshot);
        File.WriteAllBytes(Path.Combine(snapshot, path), new byte[length]);
    }

    [Fact]
    public async Task A_semantic_segmenter_counts_the_file_its_load_picks()
    {
        const string repo = "optimum/segformer-b0-finetuned-ade-512-512";
        SeedListing(repo, ("model.onnx", 1000), ("model_quantized.onnx", 500));

        (await LocalSegmenter.GetDownloadSizeBytesAsync("default", Options(), Ct)).Should().Be(1000, "the load asks for model.onnx");
        (await LocalSegmenter.IsModelDownloadedAsync("default", Options(), Ct)).Should().BeFalse();

        SeedFile(repo, "model.onnx", 1000);
        (await LocalSegmenter.GetRemainingDownloadBytesAsync("default", Options(), Ct)).Should().Be(0);
        (await LocalSegmenter.IsModelDownloadedAsync("default", Options(), Ct)).Should().BeTrue();
    }

    [Fact]
    public async Task An_interactive_segmenter_counts_its_encoder_decoder_and_configuration()
    {
        const string repo = "Acly/MobileSAM";
        SeedListing(repo, ("mobile_sam_image_encoder.onnx", 3000), ("sam_mask_decoder_multi.onnx", 2000), ("config.json", 10), ("sam_mask_decoder_single.onnx", 1500));

        (await LocalSegmenter.GetDownloadSizeBytesAsync("interactive", Options(), Ct)).Should().Be(5010, "the other decoder is not the load's");

        SeedFile(repo, "mobile_sam_image_encoder.onnx", 3000);
        (await LocalSegmenter.GetRemainingDownloadBytesAsync("interactive", Options(), Ct)).Should().Be(2010);

        SeedFile(repo, "sam_mask_decoder_multi.onnx", 2000);
        SeedFile(repo, "config.json", 10);
        (await LocalSegmenter.IsModelDownloadedAsync("interactive", Options(), Ct)).Should().BeTrue();
    }

    [Fact]
    public async Task A_cached_model_is_available_offline_only_with_its_runtime_and_delete_keeps_the_runtime()
    {
        const string repo = "optimum/segformer-b0-finetuned-ade-512-512";
        SeedListing(repo, ("model.onnx", 1000));
        SeedFile(repo, "model.onnx", 1000);
        var artifacts = new SegmenterArtifacts(Options());
        artifacts.Kind.Should().Be("segmenter");

        using (BundledRuntime.Absent())
        {
            (await artifacts.IsAvailableOfflineAsync("default", Ct)).Should().BeFalse();
            (await artifacts.GetRemainingDownloadAsync("default", Ct)).Should().Be(new RemainingDownload(0, null));
        }

        using (BundledRuntime.Use())
        {
            (await artifacts.IsAvailableOfflineAsync("default", Ct)).Should().BeTrue();
            (await artifacts.GetRemainingDownloadAsync("default", Ct)).TotalBytes.Should().Be(0);
            await new SegmenterArtifacts(new SegmenterOptions { CacheDirectory = _cache, Provider = ExecutionProvider.Cpu, DisableAutoDownload = true })
                .DownloadAsync("default", cancellationToken: Ct);

            artifacts.Delete("default").Should().BeTrue();
            (await artifacts.IsAvailableOfflineAsync("default", Ct)).Should().BeFalse("the model's file is gone; the runtime stays");
        }
    }
}
