using AwesomeAssertions;
using LMSupply.Exceptions;

namespace LMSupply.Reranker.Tests;

/// <summary>
/// <see cref="LocalReranker.GetDownloadSizeBytesAsync"/> is the size of the files a load fetches, at the lengths the
/// repository lists, for a consent screen shown before anything is fetched. The listing is seeded into the cache (fresh,
/// so no request is made); the download reads the same listing.
/// </summary>
public sealed class LocalRerankerDownloadSizeTests : IDisposable
{
    private readonly string _cacheDir = Path.Combine(Path.GetTempPath(), "lmsupply-rsize-" + Guid.NewGuid().ToString("N"));

    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    public void Dispose()
    {
        if (Directory.Exists(_cacheDir))
            Directory.Delete(_cacheDir, recursive: true);
    }

    private RerankerOptions Options(bool offline = false) => new() { CacheDirectory = _cacheDir, DisableAutoDownload = offline };

    private void SeedListing(string repoId, params (string Path, long Size)[] files)
    {
        var dir = Path.Combine(_cacheDir, ".discovery-cache");
        Directory.CreateDirectory(dir);
        File.WriteAllText(
            Path.Combine(dir, repoId.Replace('/', '_') + "_main.json"),
            "[" + string.Join(",", files.Select(f => $$"""{"path":"{{f.Path}}","type":"file","size":{{f.Size}}}""")) + "]");
    }

    [Fact]
    public async Task RegisteredAlias_IsTheGraphAndTheTokenizerFiles_NotOtherVariants()
    {
        var info = LocalReranker.Registry.Resolve("default");
        SeedListing(info.Id,
            (info.OnnxFile, 90_000_000),
            ("onnx/model_quantized.onnx", 23_000_000),
            ("tokenizer.json", 700_000),
            ("vocab.txt", 230_000),
            ("README.md", 5_000));

        (await LocalReranker.GetDownloadSizeBytesAsync("default", Options(), Ct)).Should().Be(90_930_000,
            "the load fetches the registered graph and its tokenizer files; the quantized graph and the README are not fetched");
    }

    [Fact]
    public async Task GgufRepository_IsTheQuantizationTheLoadPicks()
    {
        const string repo = "example-org/tiny-rerank-GGUF";
        SeedListing(repo,
            ("tiny-rerank.Q4_K_M.gguf", 80_000_000),
            ("tiny-rerank.Q8_0.gguf", 150_000_000));

        (await LocalReranker.GetDownloadSizeBytesAsync("gguf:" + repo, Options(), Ct)).Should().Be(80_000_000);
        var q8 = Options();
        q8.QuantizationHint = "Q8_0";
        (await LocalReranker.GetDownloadSizeBytesAsync("gguf:" + repo, q8, Ct)).Should().Be(150_000_000,
            "the quantization hint names the file the load takes");
    }

    [Fact]
    public async Task GgufRepository_Offline_IsTheCachedFileTheLoadOpens()
    {
        var dir = Path.Combine(_cacheDir, "gguf-rerankers", "example-org_tiny-rerank-GGUF");
        Directory.CreateDirectory(dir);
        File.WriteAllBytes(Path.Combine(dir, "tiny-rerank.Q4_K_M.gguf"), new byte[1234]);

        (await LocalReranker.GetDownloadSizeBytesAsync("gguf:example-org/tiny-rerank-GGUF", Options(offline: true), Ct)).Should().Be(1234);
    }

    [Fact]
    public async Task Offline_NeverListed_Throws()
    {
        var act = () => LocalReranker.GetDownloadSizeBytesAsync("default", Options(offline: true), Ct);

        await act.Should().ThrowAsync<ModelNotFoundException>();
    }

    [Fact]
    public async Task LocalPath_DownloadsNothing()
    {
        Directory.CreateDirectory(_cacheDir);
        (await LocalReranker.GetDownloadSizeBytesAsync(_cacheDir, Options(), Ct)).Should().Be(0);
    }
}
