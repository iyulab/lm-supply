using AwesomeAssertions;
using LMSupply.Download;
using LMSupply.Generator.Gguf;

namespace LMSupply.Generator.Tests;

/// <summary>
/// <see cref="LocalGenerator.IsModelDownloadedAsync"/> answers a consent gate: <c>true</c> only when the load
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
    public async Task RegistryAlias_NothingCached_IsFalse()
    {
        (await LocalGenerator.IsModelDownloadedAsync(Alias, Options(), cancellationToken: TestContext.Current.CancellationToken)).Should().BeFalse();
    }

    [Fact]
    public async Task RegistryAlias_DefaultFileCached_IsTrue()
    {
        var info = GgufModelRegistry.Resolve(Alias)!;
        Put(info.RepoId, info.DefaultFile);

        (await LocalGenerator.IsModelDownloadedAsync(Alias, Options(), cancellationToken: TestContext.Current.CancellationToken)).Should().BeTrue();
    }

    // Downloaded by another Hugging Face tool: refs/main names the commit, snapshots/{commit} holds the file.
    [Fact]
    public async Task RegistryAlias_DefaultFileInAHubSnapshot_IsTrue()
    {
        const string commit = "89abcdef0123456789abcdef0123456789abcdef";
        var info = GgufModelRegistry.Resolve(Alias)!;
        var repoDir = Path.Combine(_cacheDir, "models--" + info.RepoId.Replace("/", "--"));
        Directory.CreateDirectory(Path.Combine(repoDir, "refs"));
        File.WriteAllText(Path.Combine(repoDir, "refs", "main"), commit);
        var path = Path.Combine(repoDir, "snapshots", commit, info.DefaultFile);
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        File.WriteAllBytes(path, "GGUF-test-bytes"u8.ToArray());

        (await LocalGenerator.IsModelDownloadedAsync(Alias, Options(), cancellationToken: TestContext.Current.CancellationToken)).Should().BeTrue();
        Directory.Exists(Snapshot(info.RepoId)).Should().BeFalse();
    }

    [Fact]
    public async Task RegistryAlias_OnlyAnotherQuantOfTheSameRepoCached_IsFalse()
    {
        // The repository is "cached", but not the file this alias loads.
        var info = GgufModelRegistry.Resolve(Alias)!;
        Put(info.RepoId, info.DefaultFile.Replace("Q4_K_M", "Q8_0"));

        (await LocalGenerator.IsModelDownloadedAsync(Alias, Options(), cancellationToken: TestContext.Current.CancellationToken)).Should().BeFalse(
            "the alias's default fits any host, so a load fetches it instead of using the other quantization");
    }

    [Fact]
    public async Task RegistryAlias_CompanionFileOnly_IsFalse()
    {
        var info = GgufModelRegistry.Resolve(Alias)!;
        Put(info.RepoId, "mmproj-F16.gguf");

        (await LocalGenerator.IsModelDownloadedAsync(Alias, Options(), cancellationToken: TestContext.Current.CancellationToken)).Should().BeFalse();
    }

    [Fact]
    public async Task RegistryAlias_LfsPointerInPlaceOfTheFile_IsFalse()
    {
        var info = GgufModelRegistry.Resolve(Alias)!;
        var path = Path.Combine(Snapshot(info.RepoId), info.DefaultFile);
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        File.WriteAllText(path, "version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 123\n");

        (await LocalGenerator.IsModelDownloadedAsync(Alias, Options(), cancellationToken: TestContext.Current.CancellationToken)).Should().BeFalse();
    }

    [Fact]
    public async Task SplitRegistryModel_CountsOnlyWhenEveryShardIsCached()
    {
        const string alias = "gguf:xlarge";
        var info = GgufModelRegistry.Resolve(alias)!;
        var shards = GgufModelDownloader.GenerateShardFilenames(info.DefaultFile, info.ShardCount!.Value);

        Put(info.RepoId, shards[0]);
        Put(info.RepoId, shards[1]);
        (await LocalGenerator.IsModelDownloadedAsync(alias, Options(), cancellationToken: TestContext.Current.CancellationToken)).Should().BeFalse("one shard is still missing");

        Put(info.RepoId, shards[2]);
        (await LocalGenerator.IsModelDownloadedAsync(alias, Options(), cancellationToken: TestContext.Current.CancellationToken)).Should().BeTrue();
    }

    [Fact]
    public async Task UserAlias_FollowsItsTarget()
    {
        var info = GgufModelRegistry.Resolve(Alias)!;
        Put(info.RepoId, info.DefaultFile);
        var name = "isdl-" + Guid.NewGuid().ToString("N");
        GeneratorModelRegistry.Default.RegisterAlias(name, Alias);

        (await LocalGenerator.IsModelDownloadedAsync(name, Options(), cancellationToken: TestContext.Current.CancellationToken)).Should().BeTrue();
    }

    [Fact]
    public async Task RawGgufRepository_AnyCachedGguf_IsTrue()
    {
        const string repo = "someone/Some-Model-GGUF";
        (await LocalGenerator.IsModelDownloadedAsync(repo, Options(), cancellationToken: TestContext.Current.CancellationToken)).Should().BeFalse();

        Put(repo, "Some-Model-Q5_K_M.gguf");
        (await LocalGenerator.IsModelDownloadedAsync(repo, Options(), cancellationToken: TestContext.Current.CancellationToken)).Should().BeTrue(
            "a load of a raw repository takes a cached GGUF before it lists the repository");
    }

    [Fact]
    public async Task RawGgufRepository_OnlyImportanceMatrixCached_IsFalse()
    {
        // Quantizers publish *-imatrix.gguf next to their quants: calibration data, not a model.
        const string repo = "someone/Other-Model-GGUF";
        Put(repo, "Other-Model-imatrix.gguf");

        (await LocalGenerator.IsModelDownloadedAsync(repo, Options(), cancellationToken: TestContext.Current.CancellationToken)).Should().BeFalse();
    }

    [Fact]
    public async Task UnregisteredGgufPrefix_IsFalse()
    {
        (await LocalGenerator.IsModelDownloadedAsync("gguf:not-a-registered-alias", Options(), cancellationToken: TestContext.Current.CancellationToken)).Should().BeFalse();
    }

    [Fact]
    public async Task LocalPath_IsTrueWhenItExists()
    {
        var path = Path.Combine(_cacheDir, "local.gguf");
        (await LocalGenerator.IsModelDownloadedAsync(path, Options(), cancellationToken: TestContext.Current.CancellationToken)).Should().BeFalse();

        Directory.CreateDirectory(_cacheDir);
        File.WriteAllBytes(path, [1, 2, 3]);
        (await LocalGenerator.IsModelDownloadedAsync(path, Options(), cancellationToken: TestContext.Current.CancellationToken)).Should().BeTrue();
    }

    [Fact]
    public async Task OnnxRepository_CountsOnlyAfterACompletedDownload()
    {
        const string repo = "someone/Some-Model-onnx";
        var snapshot = Snapshot(repo);
        Directory.CreateDirectory(snapshot);
        await File.WriteAllBytesAsync(Path.Combine(snapshot, "model.onnx"), [1, 2, 3, 4], TestContext.Current.CancellationToken);

        (await LocalGenerator.IsModelDownloadedAsync(repo, Options(), cancellationToken: TestContext.Current.CancellationToken)).Should().BeFalse(
            "files without the manifest a completed download writes do not show the model is whole");

        await DownloadManifest.WriteAsync(snapshot, new DownloadManifest
        {
            Version = DownloadManifest.VerifiedVersion,
            RepoId = repo,
            Files = [new ManifestFileEntry { Path = "model.onnx", Size = 4 }, new ManifestFileEntry { Path = "genai_config.json", Size = 2 }],
        }, cancellationToken: TestContext.Current.CancellationToken);
        (await LocalGenerator.IsModelDownloadedAsync(repo, Options(), cancellationToken: TestContext.Current.CancellationToken)).Should().BeFalse("a listed file is missing");

        await File.WriteAllBytesAsync(Path.Combine(snapshot, "genai_config.json"), [(byte)'{', (byte)'}'], TestContext.Current.CancellationToken);
        (await LocalGenerator.IsModelDownloadedAsync(repo, Options(), cancellationToken: TestContext.Current.CancellationToken)).Should().BeTrue();

        await File.WriteAllBytesAsync(Path.Combine(snapshot, "model.onnx"), [1, 2], TestContext.Current.CancellationToken);
        (await LocalGenerator.IsModelDownloadedAsync(repo, Options(), cancellationToken: TestContext.Current.CancellationToken)).Should().BeFalse("a file shorter than the listed length is not the file");
    }
}
