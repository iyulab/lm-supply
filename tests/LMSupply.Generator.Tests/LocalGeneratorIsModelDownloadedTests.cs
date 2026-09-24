using AwesomeAssertions;
using LMSupply.Download;
using LMSupply.Generator.Internal.Llama;

namespace LMSupply.Generator.Tests;

/// <summary>
/// <see cref="LocalGenerator.IsModelDownloaded"/> answers a consent gate: <c>true</c> only when the load
/// with the same id and options opens cached files. Each test builds its own cache directory, so the
/// answer never depends on what this machine has downloaded. Sizes do not matter to the probe (the
/// files here are a few bytes); the one registry model used, <c>gguf:qwen3-fast</c>, fits any host.
/// </summary>
public sealed class LocalGeneratorIsModelDownloadedTests : IDisposable
{
    private const string Alias = "gguf:qwen3-fast";
    private readonly string _cacheDir = Path.Combine(Path.GetTempPath(), "lmsupply-isdl-" + Guid.NewGuid().ToString("N"));

    public void Dispose()
    {
        if (Directory.Exists(_cacheDir))
            Directory.Delete(_cacheDir, recursive: true);
    }

    private GeneratorOptions Options() => new() { CacheDirectory = _cacheDir };

    private string Snapshot(string repoId)
        => Path.Combine(_cacheDir, "models--" + repoId.Replace("/", "--"), "snapshots", "main");

    private void Put(string repoId, string file)
    {
        var path = Path.Combine(Snapshot(repoId), file.Replace('/', Path.DirectorySeparatorChar));
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        File.WriteAllBytes(path, "GGUF-test-bytes"u8.ToArray());
    }

    [Fact]
    public void RegistryAlias_NothingCached_IsFalse()
    {
        LocalGenerator.IsModelDownloaded(Alias, Options()).Should().BeFalse();
    }

    [Fact]
    public void RegistryAlias_DefaultFileCached_IsTrue()
    {
        var info = GgufModelRegistry.Resolve(Alias)!;
        Put(info.RepoId, info.DefaultFile);

        LocalGenerator.IsModelDownloaded(Alias, Options()).Should().BeTrue();
    }

    [Fact]
    public void RegistryAlias_OnlyAnotherQuantOfTheSameRepoCached_IsFalse()
    {
        // The repository is "cached", but not the file this alias loads.
        var info = GgufModelRegistry.Resolve(Alias)!;
        Put(info.RepoId, info.DefaultFile.Replace("Q4_K_M", "Q8_0"));

        LocalGenerator.IsModelDownloaded(Alias, Options()).Should().BeFalse(
            "the alias's default fits any host, so a load fetches it instead of using the other quantization");
    }

    [Fact]
    public void RegistryAlias_CompanionFileOnly_IsFalse()
    {
        var info = GgufModelRegistry.Resolve(Alias)!;
        Put(info.RepoId, "mmproj-F16.gguf");

        LocalGenerator.IsModelDownloaded(Alias, Options()).Should().BeFalse();
    }

    [Fact]
    public void RegistryAlias_LfsPointerInPlaceOfTheFile_IsFalse()
    {
        var info = GgufModelRegistry.Resolve(Alias)!;
        var path = Path.Combine(Snapshot(info.RepoId), info.DefaultFile);
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        File.WriteAllText(path, "version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 123\n");

        LocalGenerator.IsModelDownloaded(Alias, Options()).Should().BeFalse();
    }

    [Fact]
    public void SplitRegistryModel_CountsOnlyWhenEveryShardIsCached()
    {
        const string alias = "gguf:xlarge";
        var info = GgufModelRegistry.Resolve(alias)!;
        var shards = GgufModelDownloader.GenerateShardFilenames(info.DefaultFile, info.ShardCount!.Value);

        Put(info.RepoId, shards[0]);
        Put(info.RepoId, shards[1]);
        LocalGenerator.IsModelDownloaded(alias, Options()).Should().BeFalse("one shard is still missing");

        Put(info.RepoId, shards[2]);
        LocalGenerator.IsModelDownloaded(alias, Options()).Should().BeTrue();
    }

    [Fact]
    public void UserAlias_FollowsItsTarget()
    {
        var info = GgufModelRegistry.Resolve(Alias)!;
        Put(info.RepoId, info.DefaultFile);
        var name = "isdl-" + Guid.NewGuid().ToString("N");
        GeneratorModelRegistry.Default.RegisterAlias(name, Alias);

        LocalGenerator.IsModelDownloaded(name, Options()).Should().BeTrue();
    }

    [Fact]
    public void RawGgufRepository_AnyCachedGguf_IsTrue()
    {
        const string repo = "someone/Some-Model-GGUF";
        LocalGenerator.IsModelDownloaded(repo, Options()).Should().BeFalse();

        Put(repo, "Some-Model-Q5_K_M.gguf");
        LocalGenerator.IsModelDownloaded(repo, Options()).Should().BeTrue(
            "a load of a raw repository takes a cached GGUF before it lists the repository");
    }

    [Fact]
    public void RawGgufRepository_OnlyImportanceMatrixCached_IsFalse()
    {
        // Quantizers publish *-imatrix.gguf next to their quants: calibration data, not a model.
        const string repo = "someone/Other-Model-GGUF";
        Put(repo, "Other-Model-imatrix.gguf");

        LocalGenerator.IsModelDownloaded(repo, Options()).Should().BeFalse();
    }

    [Fact]
    public void UnregisteredGgufPrefix_IsFalse()
    {
        LocalGenerator.IsModelDownloaded("gguf:not-a-registered-alias", Options()).Should().BeFalse();
    }

    [Fact]
    public void LocalPath_IsTrueWhenItExists()
    {
        var path = Path.Combine(_cacheDir, "local.gguf");
        LocalGenerator.IsModelDownloaded(path, Options()).Should().BeFalse();

        Directory.CreateDirectory(_cacheDir);
        File.WriteAllBytes(path, [1, 2, 3]);
        LocalGenerator.IsModelDownloaded(path, Options()).Should().BeTrue();
    }

    [Fact]
    public async Task OnnxRepository_CountsOnlyAfterACompletedDownload()
    {
        const string repo = "someone/Some-Model-onnx";
        var snapshot = Snapshot(repo);
        Directory.CreateDirectory(snapshot);
        await File.WriteAllBytesAsync(Path.Combine(snapshot, "model.onnx"), [1, 2, 3, 4], TestContext.Current.CancellationToken);

        LocalGenerator.IsModelDownloaded(repo, Options()).Should().BeFalse(
            "files without the manifest a completed download writes do not show the model is whole");

        await DownloadManifest.WriteAsync(snapshot, new DownloadManifest
        {
            Version = DownloadManifest.VerifiedVersion,
            RepoId = repo,
            Files = [new ManifestFileEntry { Path = "model.onnx", Size = 4 }, new ManifestFileEntry { Path = "genai_config.json", Size = 2 }],
        });
        LocalGenerator.IsModelDownloaded(repo, Options()).Should().BeFalse("a listed file is missing");

        await File.WriteAllBytesAsync(Path.Combine(snapshot, "genai_config.json"), [(byte)'{', (byte)'}'], TestContext.Current.CancellationToken);
        LocalGenerator.IsModelDownloaded(repo, Options()).Should().BeTrue();

        await File.WriteAllBytesAsync(Path.Combine(snapshot, "model.onnx"), [1, 2], TestContext.Current.CancellationToken);
        LocalGenerator.IsModelDownloaded(repo, Options()).Should().BeFalse("a file shorter than the listed length is not the file");
    }
}
