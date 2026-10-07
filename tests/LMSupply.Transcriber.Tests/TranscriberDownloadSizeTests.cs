using System.Text.Json;
using AwesomeAssertions;
using LMSupply.Transcriber.Models;

namespace LMSupply.Transcriber.Tests;

/// <summary>
/// The size a consent screen shows is the size the load downloads — the quantization the load picks, not the registry's
/// full-precision figure. The repository listing is seeded into the cache (fresh, so no request is made); the load reads
/// the same listing, so these facts hold for what it would fetch.
/// </summary>
public sealed class TranscriberDownloadSizeTests : IDisposable
{
    private readonly string _cache = Path.Combine(Path.GetTempPath(), "lmsupply-stt-size-" + Guid.NewGuid().ToString("N"));

    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    public void Dispose()
    {
        if (Directory.Exists(_cache))
            Directory.Delete(_cache, recursive: true);
    }

    private const long Int8Pair = 23_000_000 + 52_000_000;
    private const long Fp32Pair = 83_000_000 + 208_000_000;
    private const long Shared = 2_000 + 1_000 + 2_000_000 + 300;

    private void SeedWhisperBaseListing()
    {
        var files = new (string Path, long Size)[]
        {
            ("onnx/encoder_model.onnx", 83_000_000),
            ("onnx/encoder_model_int8.onnx", 23_000_000),
            ("onnx/decoder_model_merged.onnx", 208_000_000),
            ("onnx/decoder_model_merged_int8.onnx", 52_000_000),
            ("config.json", 2_000),
            ("generation_config.json", 1_000),
            ("tokenizer.json", 2_000_000),
            ("preprocessor_config.json", 300),
        };
        // The listing cache: models--{org}--{name}/.lmsupply/listings/{revision}.json
        var path = Path.Combine(_cache, "models--" + DefaultModels.WhisperBase.Id.Replace("/", "--"), ".lmsupply", "listings", "main.json");
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        File.WriteAllText(
            path,
            JsonSerializer.Serialize(files.Select(f => new { path = f.Path, type = "file", size = f.Size })));
    }

    [Theory]
    [InlineData("int8", Int8Pair)]
    [InlineData("fp32", Fp32Pair)]
    public async Task Size_FollowsTheQuantizationTheLoadPicks(string hint, long pair)
    {
        SeedWhisperBaseListing();

        var size = await LocalTranscriber.GetDownloadSizeBytesAsync(
            "default", new TranscriberOptions { CacheDirectory = _cache, QuantizationHint = hint }, Ct);

        size.Should().BeGreaterThanOrEqualTo(pair).And.BeLessThanOrEqualTo(pair + Shared);
        if (hint == "int8")
            size.Should().BeLessThan(DefaultModels.WhisperBase.SizeBytes / 2, "the int8 download is a fraction of the registry's full-precision size");
    }

    [Fact]
    public async Task AVariantQualifier_IsHonouredLikeTheLoad()
    {
        SeedWhisperBaseListing();
        var options = new TranscriberOptions { CacheDirectory = _cache };

        var viaQualifier = await LocalTranscriber.GetDownloadSizeBytesAsync("default:int8", options, Ct);
        var viaHint = await LocalTranscriber.GetDownloadSizeBytesAsync(
            "default", new TranscriberOptions { CacheDirectory = _cache, QuantizationHint = "int8" }, Ct);

        viaQualifier.Should().Be(viaHint);
        options.ModelId.Should().Be("default", "the caller's options are not modified");
        options.QuantizationHint.Should().BeNull();
    }

    [Fact]
    public async Task PreloadDiarization_AddsThePair()
    {
        SeedWhisperBaseListing();

        var without = await LocalTranscriber.GetDownloadSizeBytesAsync(
            "default", new TranscriberOptions { CacheDirectory = _cache, QuantizationHint = "int8" }, Ct);
        var with = await LocalTranscriber.GetDownloadSizeBytesAsync(
            "default", new TranscriberOptions { CacheDirectory = _cache, QuantizationHint = "int8", PreloadDiarization = true }, Ct);

        (with - without).Should().Be(LocalTranscriber.DiarizationDownloadSizeBytes);
    }

    [Fact]
    public async Task AModelOnLocalDisk_DownloadsNothing()
    {
        Directory.CreateDirectory(_cache);

        (await LocalTranscriber.GetDownloadSizeBytesAsync(_cache, new TranscriberOptions(), Ct)).Should().Be(0);
        (await LocalTranscriber.GetDownloadSizeBytesAsync(_cache, new TranscriberOptions { PreloadDiarization = true }, Ct))
            .Should().Be(LocalTranscriber.DiarizationDownloadSizeBytes);
    }

    // The real repository, against a real load into an empty cache — the measurement the consumer made by hand.
    [Fact]
    [Trait("Category", "Integration")]
    public async Task DefaultSize_EqualsWhatADefaultLoadWritesToAnEmptyCache()
    {
        var options = new TranscriberOptions { CacheDirectory = _cache };

        var size = await LocalTranscriber.GetDownloadSizeBytesAsync(options, Ct);
        await using (await LocalTranscriber.LoadAsync(options, cancellationToken: Ct)) { }

        var repoDir = Directory.GetDirectories(_cache, "models--onnx-community--whisper-base").Single();
        var written = Directory.GetFiles(Path.Combine(repoDir, "snapshots"), "*", SearchOption.AllDirectories)
            .Where(f => !Path.GetFileName(f).StartsWith('.'))
            .Sum(f => new FileInfo(f).Length);
        size.Should().Be(written);
    }
}
