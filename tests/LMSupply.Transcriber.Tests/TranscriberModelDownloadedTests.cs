using System.Text.Json;
using AwesomeAssertions;
using LMSupply.Download;
using LMSupply.Transcriber.Models;

namespace LMSupply.Transcriber.Tests;

/// <summary>
/// <see cref="LocalTranscriber.IsModelDownloadedAsync(string, TranscriberOptions?, CancellationToken)"/> answers for the files
/// the load picks, at the lengths the repository lists — never from the mere presence of the repository directory. The
/// listing is seeded into the cache with small sizes and the files are written at those sizes; nothing reaches the network.
/// </summary>
public sealed class TranscriberModelDownloadedTests : IDisposable
{
    private readonly string _cache = Path.Combine(Path.GetTempPath(), "lmsupply-stt-cached-" + Guid.NewGuid().ToString("N"));

    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    public void Dispose()
    {
        if (Directory.Exists(_cache))
            Directory.Delete(_cache, recursive: true);
    }

    private static readonly (string Path, long Size)[] Listing =
    [
        ("onnx/encoder_model.onnx", 8_300),
        ("onnx/encoder_model_int8.onnx", 2_300),
        ("onnx/decoder_model_merged.onnx", 20_800),
        ("onnx/decoder_model_merged_int8.onnx", 5_200),
        ("config.json", 200),
        ("generation_config.json", 100),
        ("tokenizer.json", 2_000),
        ("preprocessor_config.json", 30),
    ];

    private string SnapshotDir => CacheManager.GetModelDirectory(_cache, DefaultModels.WhisperBase.Id);

    private void SeedListing()
    {
        var dir = Path.Combine(_cache, ".discovery-cache");
        Directory.CreateDirectory(dir);
        File.WriteAllText(
            Path.Combine(dir, DefaultModels.WhisperBase.Id.Replace('/', '_') + "_main.json"),
            JsonSerializer.Serialize(Listing.Select(f => new { path = f.Path, type = "file", size = f.Size })));
    }

    private void WriteFile(string repoPath, long length) => WriteFile(SnapshotDir, repoPath, length);

    private static void WriteFile(string snapshotDir, string repoPath, long length)
    {
        var path = Path.Combine(snapshotDir, repoPath.Replace('/', Path.DirectorySeparatorChar));
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        File.WriteAllBytes(path, new byte[length]);
    }

    private void WriteAllListedFiles()
    {
        foreach (var (path, size) in Listing)
            WriteFile(path, size);
    }

    private TranscriberOptions Options(string? hint = null) => new() { CacheDirectory = _cache, QuantizationHint = hint };

    [Fact]
    public async Task AnEmptyCache_IsNotDownloaded_WithoutARequest()
    {
        (await LocalTranscriber.IsModelDownloadedAsync("default", Options("int8"), Ct)).Should().BeFalse();

        Directory.Exists(Path.Combine(_cache, ".discovery-cache")).Should().BeFalse(
            "a listing request would have cached its answer; the check reads the cache only");
    }

    // A consumer asks at start-up, days after the download: the listing's one-day freshness must not turn it into "no".
    [Fact]
    public async Task AListingOlderThanItsFreshness_StillAnswers()
    {
        SeedListing();
        WriteAllListedFiles();
        foreach (var listing in Directory.GetFiles(Path.Combine(_cache, ".discovery-cache")))
            File.SetLastWriteTimeUtc(listing, DateTime.UtcNow.AddDays(-30));

        (await LocalTranscriber.IsModelDownloadedAsync("default", Options("int8"), Ct)).Should().BeTrue();
    }

    // The consumer's repro: a load that failed after the listing left the repository directory behind with no model files.
    [Fact]
    public async Task ARepositoryDirectoryWithoutItsFiles_IsNotDownloaded()
    {
        SeedListing();
        Directory.CreateDirectory(SnapshotDir);

        (await LocalTranscriber.IsModelDownloadedAsync("default", Options("int8"), Ct)).Should().BeFalse();
    }

    [Fact]
    public async Task EveryListedFileAtItsLength_IsDownloaded()
    {
        SeedListing();
        WriteAllListedFiles();

        (await LocalTranscriber.IsModelDownloadedAsync("default", Options("int8"), Ct)).Should().BeTrue();
        (await LocalTranscriber.IsModelDownloadedAsync("default", Options("fp32"), Ct)).Should().BeTrue();
    }

    [Fact]
    public async Task TheAnswer_FollowsTheQuantizationTheLoadPicks()
    {
        SeedListing();
        WriteAllListedFiles();
        File.Delete(Path.Combine(SnapshotDir, "onnx", "encoder_model_int8.onnx"));

        (await LocalTranscriber.IsModelDownloadedAsync("default", Options("int8"), Ct)).Should().BeFalse();
        (await LocalTranscriber.IsModelDownloadedAsync("default:int8", Options(), Ct)).Should().BeFalse("the qualifier resolves like the load's");
        (await LocalTranscriber.IsModelDownloadedAsync("default", Options("fp32"), Ct)).Should().BeTrue();
    }

    [Fact]
    public async Task APartialFile_IsNotDownloaded()
    {
        SeedListing();
        WriteAllListedFiles();
        WriteFile("onnx/decoder_model_merged_int8.onnx", 1_000);

        (await LocalTranscriber.IsModelDownloadedAsync("default", Options("int8"), Ct)).Should().BeFalse();
    }

    [Fact]
    public async Task AModelOnLocalDisk_IsDownloaded()
    {
        Directory.CreateDirectory(_cache);

        (await LocalTranscriber.IsModelDownloadedAsync(_cache, new TranscriberOptions(), Ct)).Should().BeTrue();
    }

    [Fact]
    public async Task TheCallersOptions_AreNotModified()
    {
        SeedListing();
        var options = Options();

        await LocalTranscriber.IsModelDownloadedAsync("default:int8", options, Ct);

        options.ModelId.Should().Be("default");
        options.QuantizationHint.Should().BeNull();
        options.DisableAutoDownload.Should().BeFalse();
    }

    private const string Commit = "3c1a0f7e9b2d4c6a8e0f1b3d5c7a9e1f2b4d6c8a";

    private string RepoDir => Path.Combine(_cache, "models--" + DefaultModels.WhisperBase.Id.Replace("/", "--"));

    // What another Hugging Face tool's snapshot download leaves: refs/main names the commit, snapshots/{commit} holds the files.
    private string SeedHubSnapshot()
    {
        var snapshot = Path.Combine(RepoDir, "snapshots", Commit);
        Directory.CreateDirectory(Path.Combine(RepoDir, "refs"));
        File.WriteAllText(Path.Combine(RepoDir, "refs", "main"), Commit);
        foreach (var (path, size) in Listing)
            WriteFile(snapshot, path, size);
        return snapshot;
    }

    // The shared cache holds the model in the hub layout: listed by an earlier size query, files downloaded by another tool.
    [Fact]
    public async Task AHubSnapshot_IsDownloaded_AndNothingIsWritten()
    {
        SeedListing();
        SeedHubSnapshot();

        (await LocalTranscriber.IsModelDownloadedAsync("default", Options("int8"), Ct)).Should().BeTrue();
        (await LocalTranscriber.IsModelDownloadedAsync("default", Options("fp32"), Ct)).Should().BeTrue();
        Directory.Exists(SnapshotDir).Should().BeFalse("the check reads the cache and writes nothing");
    }

    // No listing was ever cached: the hub snapshot lists itself, so the check still answers without a request.
    [Fact]
    public async Task AHubSnapshot_WithoutACachedListing_IsDownloaded()
    {
        SeedHubSnapshot();

        (await LocalTranscriber.IsModelDownloadedAsync("default", Options(), Ct)).Should().BeTrue();
        Directory.Exists(Path.Combine(_cache, ".discovery-cache")).Should().BeFalse();
    }

    [Fact]
    public async Task AHubSnapshot_WithAPartialFile_IsNotDownloaded()
    {
        SeedListing();
        var snapshot = SeedHubSnapshot();
        WriteFile(snapshot, "onnx/decoder_model_merged_int8.onnx", 1_000);

        (await LocalTranscriber.IsModelDownloadedAsync("default", Options("int8"), Ct)).Should().BeFalse();
    }

    // The load takes the same files from the hub snapshot: offline, it must not need anything else.
    [Fact]
    public async Task AHubSnapshot_ADownloadOfTheLoadsFiles_FetchesNothing()
    {
        SeedListing();
        var snapshot = SeedHubSnapshot();
        using var downloader = new HuggingFaceDownloader(_cache, localFilesOnly: true);

        var (dir, _) = await downloader.DownloadWithDiscoveryAsync(DefaultModels.WhisperBase.Id, cancellationToken: Ct);

        dir.Should().Be(snapshot);
        Directory.Exists(SnapshotDir).Should().BeFalse();
    }
}
