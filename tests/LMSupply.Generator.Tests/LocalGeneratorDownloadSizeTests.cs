using AwesomeAssertions;
using LMSupply.Exceptions;
using LMSupply.Generator.Gguf;

namespace LMSupply.Generator.Tests;

/// <summary>
/// <see cref="LocalGenerator.GetDownloadSizeBytesAsync"/> is the size of the files the load picks on this host, at the
/// lengths the repository lists. The listing is seeded into the cache (fresh, so no request is made); the download reads
/// the same listing, so these facts hold for what it would fetch. <c>gguf:qwen3-fast</c> fits any host.
/// </summary>
public sealed class LocalGeneratorDownloadSizeTests : IDisposable
{
    private const string Alias = "gguf:qwen3-fast";
    private readonly string _cacheDir = Path.Combine(Path.GetTempPath(), "lmsupply-gsize-" + Guid.NewGuid().ToString("N"));

    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    public void Dispose()
    {
        if (Directory.Exists(_cacheDir))
            Directory.Delete(_cacheDir, recursive: true);
    }

    private GeneratorOptions Options() => new() { CacheDirectory = _cacheDir };

    private void SeedListing(string repoId, params (string Path, long Size)[] files)
    {
        var dir = Path.Combine(_cacheDir, ".discovery-cache");
        Directory.CreateDirectory(dir);
        File.WriteAllText(
            Path.Combine(dir, repoId.Replace('/', '_') + "_main.json"),
            "[" + string.Join(",", files.Select(f => $$"""{"path":"{{f.Path}}","type":"file","size":{{f.Size}}}""")) + "]");
    }

    [Fact]
    public async Task RegistryAlias_IsTheSizeOfTheFileTheLoadPicks()
    {
        var info = GgufModelRegistry.Resolve(Alias)!;
        SeedListing(info.RepoId,
            (info.DefaultFile, 1_100_000_000),
            (info.DefaultFile.Replace("Q4_K_M", "Q8_0"), 2_200_000_000),
            ("README.md", 5_000));

        (await LocalGenerator.GetDownloadSizeBytesAsync(Alias, Options(), Ct)).Should().Be(1_100_000_000,
            "the alias's default fits any host, so the load fetches it — not the other quantization");
    }

    [Fact]
    public async Task SplitRegistryModel_IsEveryShard()
    {
        const string alias = "gguf:xlarge";
        var info = GgufModelRegistry.Resolve(alias)!;
        var shards = GgufModelDownloader.GenerateShardFilenames(info.DefaultFile, info.ShardCount!.Value);
        SeedListing(info.RepoId, [.. shards.Select((s, i) => (s, 1_000_000L * (i + 1)))]);

        var expected = shards.Select((_, i) => 1_000_000L * (i + 1)).Sum();
        (await LocalGenerator.GetDownloadSizeBytesAsync(alias, Options(), Ct)).Should().Be(expected);
    }

    [Fact]
    public async Task Auto_IsTheModelTheSelectionPicks()
    {
        var options = Options();
        var picked = LocalGenerator.SelectAutoModel(options).ModelId;
        var info = GgufModelRegistry.Resolve(picked, options.SelectionProvider, options.MaxContextLength, options.AutoSelectionGoal)!;
        // The pick depends on this host; a large host can pick a split model, whose every shard is counted.
        var files = info.ShardCount is > 1
            ? GgufModelDownloader.GenerateShardFilenames(info.DefaultFile, info.ShardCount.Value)
            : [info.DefaultFile];
        SeedListing(info.RepoId, [.. files.Select(f => (f, 3_300_000_000L))]);
        var expected = 3_300_000_000L * files.Count;

        (await LocalGenerator.GetDownloadSizeBytesAsync("auto", options, Ct)).Should().Be(expected);
        (await LocalGenerator.GetDownloadSizeBytesAsync("default", options, Ct)).Should().Be(expected);
    }

    [Fact]
    public async Task Auto_FollowsThePreferredModel()
    {
        var info = GgufModelRegistry.Resolve(Alias)!;
        SeedListing(info.RepoId, (info.DefaultFile, 1_100_000_000));

        var options = new GeneratorOptions { CacheDirectory = _cacheDir, PreferredAutoModelId = Alias };
        (await LocalGenerator.GetDownloadSizeBytesAsync("auto", options, Ct)).Should().Be(1_100_000_000);
    }

    [Fact]
    public async Task RawGgufRepository_IsTheFileTheLoadPicks()
    {
        const string repo = "someone/Some-Model-GGUF";
        SeedListing(repo, ("Some-Model-Q4_K_M.gguf", 700_000_000), ("Some-Model-imatrix.gguf", 4_000_000));

        (await LocalGenerator.GetDownloadSizeBytesAsync(repo, Options(), Ct)).Should().Be(700_000_000,
            "the importance matrix is calibration data, not a model the load takes");
    }

    [Fact]
    public async Task ALocalPath_DownloadsNothing()
    {
        Directory.CreateDirectory(_cacheDir);
        var path = Path.Combine(_cacheDir, "local.gguf");
        File.WriteAllBytes(path, "GGUF-test-bytes"u8.ToArray());

        (await LocalGenerator.GetDownloadSizeBytesAsync(path, Options(), Ct)).Should().Be(0);
    }

    [Fact]
    public async Task DownloadsDisabled_WithoutACachedListing_Throws()
    {
        var options = new GeneratorOptions { CacheDirectory = _cacheDir, DisableAutoDownload = true };

        await FluentActions.Awaiting(() => LocalGenerator.GetDownloadSizeBytesAsync(Alias, options, Ct))
            .Should().ThrowAsync<ModelNotFoundException>();
    }

    [Fact]
    public async Task DownloadsDisabled_AnswersFromTheCachedListing()
    {
        var info = GgufModelRegistry.Resolve(Alias)!;
        SeedListing(info.RepoId, (info.DefaultFile, 1_100_000_000));
        var options = new GeneratorOptions { CacheDirectory = _cacheDir, DisableAutoDownload = true };

        (await LocalGenerator.GetDownloadSizeBytesAsync(Alias, options, Ct)).Should().Be(1_100_000_000);
    }

    [Fact]
    public async Task AListingWithoutTheFilesLength_Throws()
    {
        var info = GgufModelRegistry.Resolve(Alias)!;
        SeedListing(info.RepoId, (info.DefaultFile, 0));

        await FluentActions.Awaiting(() => LocalGenerator.GetDownloadSizeBytesAsync(Alias, Options(), Ct))
            .Should().ThrowAsync<ModelDownloadException>();
    }

    [Fact]
    public async Task TheCallersOptions_AreNotModified()
    {
        Directory.CreateDirectory(_cacheDir);
        var path = Path.Combine(_cacheDir, "local.gguf");
        File.WriteAllBytes(path, "GGUF-test-bytes"u8.ToArray());
        var options = new GeneratorOptions { CacheDirectory = _cacheDir, PreferredAutoModelId = path };

        (await LocalGenerator.GetDownloadSizeBytesAsync("default:int4", options, Ct)).Should().Be(0);

        options.QuantizationHint.Should().BeNull("the qualifier is applied to a copy");
    }
}
