using AwesomeAssertions;
using LMSupply.Download;
using LMSupply.Embedder.Utils;
using LMSupply.Exceptions;

namespace LMSupply.Embedder.Tests;

/// <summary>
/// <see cref="LocalEmbedder.GetDownloadSizeBytesAsync"/> is the size of the files the load picks, at the lengths the
/// repository lists, for a consent screen shown before anything is fetched. The listing is seeded into the cache (fresh,
/// so no request is made); the download reads the same listing, so these facts hold for what it would fetch.
/// </summary>
public sealed class LocalEmbedderDownloadSizeTests : IDisposable
{
    private readonly string _cacheDir = Path.Combine(Path.GetTempPath(), "lmsupply-esize-" + Guid.NewGuid().ToString("N"));

    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    public void Dispose()
    {
        if (Directory.Exists(_cacheDir))
            Directory.Delete(_cacheDir, recursive: true);
    }

    private EmbedderOptions Options() => new() { CacheDirectory = _cacheDir };

    private void SeedListing(string repoId, params (string Path, long Size)[] files)
    {
        var dir = Path.Combine(_cacheDir, ".discovery-cache");
        Directory.CreateDirectory(dir);
        File.WriteAllText(
            Path.Combine(dir, repoId.Replace('/', '_') + "_main.json"),
            "[" + string.Join(",", files.Select(f => $$"""{"path":"{{f.Path}}","type":"file","size":{{f.Size}}}""")) + "]");
    }

    [Fact]
    public async Task CatalogAlias_IsTheModelFileAndItsTokenizer_NotOtherVariants()
    {
        EmbedderModelRegistry.Default.TryResolveCatalog("default", out var info, out _).Should().BeTrue();
        var model = string.IsNullOrEmpty(info!.Subfolder) ? "model.onnx" : $"{info.Subfolder}/model.onnx";
        var int8 = string.IsNullOrEmpty(info.Subfolder) ? "model_int8.onnx" : $"{info.Subfolder}/model_int8.onnx";
        SeedListing(info.RepoId,
            (model, 90_000_000),
            (int8, 30_000_000),
            ("tokenizer.json", 700_000),
            ("README.md", 5_000));

        (await LocalEmbedder.GetDownloadSizeBytesAsync("default", Options(), Ct)).Should().Be(90_700_000,
            "the alias loads model.onnx and its tokenizer; the int8 file beside it and the README are not fetched");
    }

    [Fact]
    public async Task RemainingBytes_AreTheFilesTheCacheDoesNotHoldAtTheListedLength()
    {
        EmbedderModelRegistry.Default.TryResolveCatalog("default", out var info, out _).Should().BeTrue();
        var model = string.IsNullOrEmpty(info!.Subfolder) ? "model.onnx" : $"{info.Subfolder}/model.onnx";
        SeedListing(info.RepoId, (model, 9_000), ("tokenizer.json", 700));
        var snapshot = CacheManager.GetModelDirectory(_cacheDir, info.RepoId);

        (await LocalEmbedder.GetRemainingDownloadBytesAsync("default", Options(), Ct)).Should().Be(9_700, "nothing is cached");

        Place(snapshot, "tokenizer.json", 700);
        (await LocalEmbedder.GetRemainingDownloadBytesAsync("default", Options(), Ct)).Should().Be(9_000);

        Place(snapshot, model, 10);
        (await LocalEmbedder.GetRemainingDownloadBytesAsync("default", Options(), Ct)).Should().Be(9_000,
            "a file at another length than the listing's is fetched again");

        Place(snapshot, model, 9_000);
        (await LocalEmbedder.GetRemainingDownloadBytesAsync("default", Options(), Ct)).Should().Be(0);
        (await LocalEmbedder.GetDownloadSizeBytesAsync("default", Options(), Ct)).Should().Be(9_700, "the total is unchanged");
    }

    [Fact]
    public async Task RemainingBytes_CountAGgufFileInTheTreeEarlierVersionsWrote_AsCached()
    {
        const string repo = "example-org/tiny-embed-GGUF";
        SeedListing(repo, ("tiny-embed.Q4_K_M.gguf", 4_000));
        (await LocalEmbedder.GetRemainingDownloadBytesAsync("gguf:" + repo, Options(), Ct)).Should().Be(4_000);

        Place(Path.Combine(_cacheDir, "gguf-embeddings", "example-org_tiny-embed-GGUF"), "tiny-embed.Q4_K_M.gguf", 4_000);

        (await LocalEmbedder.GetRemainingDownloadBytesAsync("gguf:" + repo, Options(), Ct)).Should().Be(0,
            "the load opens that file without downloading");
    }

    private static void Place(string directory, string repoPath, int length)
    {
        var path = Path.Combine(directory, repoPath.Replace('/', Path.DirectorySeparatorChar));
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        File.WriteAllBytes(path, new byte[length]);
    }

    [Fact]
    public async Task GgufRepository_IsTheQuantizationTheLoadPicks()
    {
        const string repo = "example-org/tiny-embed-GGUF";
        SeedListing(repo,
            ("tiny-embed.Q4_K_M.gguf", 80_000_000),
            ("tiny-embed.Q8_0.gguf", 150_000_000),
            ("README.md", 5_000));

        (await LocalEmbedder.GetDownloadSizeBytesAsync("gguf:" + repo, Options(), Ct)).Should().Be(80_000_000,
            "the embedder prefers Q4_K_M when it fits, so the load fetches it and not the Q8_0 file");
    }

    [Fact]
    public async Task GgufRepository_Offline_IsTheCachedFileTheLoadOpens()
    {
        const string repo = "example-org/tiny-embed-GGUF";
        var dir = Path.Combine(_cacheDir, "gguf-embeddings", "example-org_tiny-embed-GGUF");
        Directory.CreateDirectory(dir);
        File.WriteAllBytes(Path.Combine(dir, "tiny-embed.Q4_K_M.gguf"), new byte[1234]);

        var options = Options();
        options.DisableAutoDownload = true;

        (await LocalEmbedder.GetDownloadSizeBytesAsync("gguf:" + repo, options, Ct)).Should().Be(1234);
    }

    [Fact]
    public async Task GgufRepository_Offline_NothingCached_Throws()
    {
        var options = Options();
        options.DisableAutoDownload = true;

        var act = () => LocalEmbedder.GetDownloadSizeBytesAsync("gguf:example-org/absent-GGUF", options, Ct);

        await act.Should().ThrowAsync<ModelNotFoundException>();
    }

    [Fact]
    public async Task LocalFiles_DownloadNothing()
    {
        Directory.CreateDirectory(_cacheDir);
        var onnx = Path.Combine(_cacheDir, "model.onnx");
        var gguf = Path.Combine(_cacheDir, "model.gguf");
        File.WriteAllBytes(onnx, [1]);
        File.WriteAllBytes(gguf, [1]);

        (await LocalEmbedder.GetDownloadSizeBytesAsync(onnx, Options(), Ct)).Should().Be(0);
        (await LocalEmbedder.GetDownloadSizeBytesAsync(gguf, Options(), Ct)).Should().Be(0);
    }

    [Fact]
    public async Task UnknownModel_Throws_AsTheLoadDoes()
    {
        var act = () => LocalEmbedder.GetDownloadSizeBytesAsync("no-such-alias", Options(), Ct);

        await act.Should().ThrowAsync<ModelNotFoundException>();
    }

    [Fact]
    public async Task CallersOptions_AreNotModified()
    {
        EmbedderModelRegistry.Default.TryResolveCatalog("default", out var info, out _).Should().BeTrue();
        var model = string.IsNullOrEmpty(info!.Subfolder) ? "model.onnx" : $"{info.Subfolder}/model.onnx";
        SeedListing(info.RepoId, (model, 90_000_000), ("tokenizer.json", 700_000));
        var options = Options();

        await LocalEmbedder.GetDownloadSizeBytesAsync("default:fp16", options, Ct);

        options.QuantizationHint.Should().BeNull("the size query reads a copy; a ':variant' qualifier is not written back");
    }
}
