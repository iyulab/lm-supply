using System.Text.Json;
using AwesomeAssertions;
using LMSupply.Lifecycle;
using LMSupply.Testing;

namespace LMSupply.Translator.Tests;

/// <summary>
/// <see cref="TranslatorArtifacts"/> and the lifecycle of <see cref="LocalTranslator"/>: the size, availability, download and
/// delete of the encoder, decoder and tokenizer files a load picks — here from a repository that also publishes a
/// quantized encoder and an unmerged decoder — and of the ONNX Runtime it runs on. Network-free: a seeded listing and
/// snapshot in a private cache, and an application runtime directory.
/// </summary>
public sealed class TranslatorArtifactsTests : IDisposable
{
    private const string Repo = "onnx-community/opus-mt-ko-en";
    private readonly string _cache = Path.Combine(Path.GetTempPath(), "lmsupply-translator-artifacts-" + Guid.NewGuid().ToString("N"));

    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    public void Dispose()
    {
        try { Directory.Delete(_cache + "-runtime", recursive: true); } catch { /* best-effort temp cleanup */ }
        try { Directory.Delete(_cache, recursive: true); } catch { /* best-effort temp cleanup */ }
    }

    private TranslatorOptions Options(bool offline = false) =>
        new() { CacheDirectory = _cache, Provider = ExecutionProvider.Cpu, QuantizationHint = "fp32", DisableAutoDownload = offline };

    private string RepoDir => Path.Combine(_cache, "models--" + Repo.Replace("/", "--"));

    private const long Encoder = 1_000;
    private const long MergedDecoder = 2_000;
    private const long UnmergedDecoder = 1_900;
    private const long Supporting = 10 + 20 + 30 + 40 + 5;

    private static readonly (string Path, long Size)[] Listing =
    [
        ("onnx/encoder_model.onnx", Encoder),
        ("onnx/encoder_model_quantized.onnx", 300),
        ("onnx/decoder_model_merged.onnx", MergedDecoder),
        ("onnx/decoder_model.onnx", UnmergedDecoder),
        ("config.json", 10),
        ("source.spm", 20),
        ("target.spm", 30),
        ("vocab.json", 40),
        ("tokenizer_config.json", 5),
    ];

    private void SeedListing()
    {
        var path = Path.Combine(RepoDir, ".lmsupply", "listings", "main.json");
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        File.WriteAllText(path, JsonSerializer.Serialize(Listing.Select(f => new { path = f.Path, type = "file", size = f.Size })));
    }

    // Every listed file in this library's snapshot; a partial file where asked
    private void SeedSnapshot(long? encoderLength = null)
    {
        var snapshot = Path.Combine(RepoDir, "snapshots", "main");
        foreach (var (file, size) in Listing)
        {
            var path = Path.Combine(snapshot, file);
            Directory.CreateDirectory(Path.GetDirectoryName(path)!);
            File.WriteAllBytes(path, new byte[file == "onnx/encoder_model.onnx" ? encoderLength ?? size : size]);
        }
    }

    [Fact]
    public async Task The_size_is_the_files_the_load_picks_and_what_is_left_of_them()
    {
        SeedListing();

        var size = await LocalTranslator.GetDownloadSizeBytesAsync("default", Options(), Ct);
        size.Should().BeInRange(Encoder + MergedDecoder, Encoder + MergedDecoder + Supporting,
            "the quantized encoder and the unmerged decoder are not what the load takes");
        (await LocalTranslator.GetRemainingDownloadBytesAsync("default", Options(), Ct)).Should().Be(size);
        (await LocalTranslator.IsModelDownloadedAsync("default", Options(), Ct)).Should().BeFalse();

        SeedSnapshot(encoderLength: 400);
        (await LocalTranslator.GetRemainingDownloadBytesAsync("default", Options(), Ct)).Should().Be(Encoder, "a partial file counts in full");
        (await LocalTranslator.IsModelDownloadedAsync("default", Options(), Ct)).Should().BeFalse();

        SeedSnapshot();
        (await LocalTranslator.GetRemainingDownloadBytesAsync("default", Options(), Ct)).Should().Be(0);
        (await LocalTranslator.IsModelDownloadedAsync("default", Options(), Ct)).Should().BeTrue();
    }

    [Fact]
    public async Task A_variant_qualifier_is_honoured_like_the_load()
    {
        SeedListing();
        var options = new TranslatorOptions { CacheDirectory = _cache, Provider = ExecutionProvider.Cpu };

        (await LocalTranslator.GetDownloadSizeBytesAsync("default:fp32", options, Ct))
            .Should().Be(await LocalTranslator.GetDownloadSizeBytesAsync("default", Options(), Ct));
        options.QuantizationHint.Should().BeNull("the caller's options are not modified");
    }

    [Fact]
    public async Task A_cached_model_is_available_offline_only_with_its_runtime()
    {
        SeedListing();
        SeedSnapshot();
        var artifacts = new TranslatorArtifacts(Options());
        artifacts.Kind.Should().Be("translator");

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
            await new TranslatorArtifacts(Options(offline: true)).DownloadAsync("default", cancellationToken: Ct);
        }
    }

    [Fact]
    public async Task A_model_never_listed_into_this_cache_is_not_downloaded()
    {
        (await LocalTranslator.IsModelDownloadedAsync("default", Options(), Ct)).Should().BeFalse();
        Directory.Exists(_cache).Should().BeFalse("the check reads the cache only and writes nothing");
    }

    [Fact]
    public void Delete_removes_this_librarys_snapshot_and_keeps_a_local_model()
    {
        SeedListing();
        SeedSnapshot();
        var artifacts = new TranslatorArtifacts(Options());

        artifacts.Delete("default").Should().BeTrue();
        artifacts.Delete("default").Should().BeFalse("nothing is left to delete");

        var local = Path.Combine(_cache + "-runtime", "local-model");
        Directory.CreateDirectory(local);
        LocalTranslator.DeleteModel(local, Options()).Should().BeFalse();
        Directory.Exists(local).Should().BeTrue();
    }
}
