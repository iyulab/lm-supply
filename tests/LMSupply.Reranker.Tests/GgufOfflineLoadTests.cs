using AwesomeAssertions;
using LMSupply.Exceptions;

namespace LMSupply.Reranker.Tests;

/// <summary>
/// The reranker's ONNX path has honoured <see cref="RerankerOptions.DisableAutoDownload"/> since the option
/// existed, through <c>ModelManager</c>; its GGUF path built its own downloader and ignored the flag — the same
/// shape as the segmenter's ONNX path before 0.64.0. Since 0.65.0 both paths refuse the same way.
/// </summary>
public sealed class GgufOfflineLoadTests : IDisposable
{
    private readonly string _cacheDir = Path.Combine(Path.GetTempPath(), "lmsupply-reranker-gguf-cache-" + Guid.NewGuid().ToString("N"));

    public void Dispose()
    {
        if (Directory.Exists(_cacheDir))
            Directory.Delete(_cacheDir, recursive: true);
    }

    [Fact]
    public async Task DisableAutoDownload_WithAGgufRepoNotCached_FailsTheLoadInsteadOfListingTheRepository()
    {
        var options = new RerankerOptions { CacheDirectory = _cacheDir, DisableAutoDownload = true };

        var load = () => LocalReranker.LoadAsync("gguf:example/some-reranker-GGUF", options, cancellationToken: TestContext.Current.CancellationToken);

        await load.Should().ThrowAsync<ModelNotFoundException>().WithMessage("*downloads are disabled*");
        Directory.Exists(_cacheDir).Should().BeFalse("an offline miss downloads nothing and writes nothing to the cache");
    }
}
