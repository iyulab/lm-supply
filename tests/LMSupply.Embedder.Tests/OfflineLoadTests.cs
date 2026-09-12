using AwesomeAssertions;
using LMSupply.Exceptions;

namespace LMSupply.Embedder.Tests;

/// <summary>
/// <see cref="EmbedderOptions.DisableAutoDownload"/> makes a load fail-closed on every download path —
/// registry alias, repository id, GGUF repository — reading the cache and writing nothing. Until 0.65.0
/// the option did not exist on this module, so a consumer behind a download-consent boundary could only
/// guard the embedder with its own cache probe.
/// </summary>
public sealed class OfflineLoadTests : IDisposable
{
    private readonly string _cacheDir = Path.Combine(Path.GetTempPath(), "lmsupply-embedder-cache-" + Guid.NewGuid().ToString("N"));

    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    public void Dispose()
    {
        if (Directory.Exists(_cacheDir))
            Directory.Delete(_cacheDir, recursive: true);
    }

    [Fact]
    public void DisableAutoDownload_DefaultsToFalse() =>
        new EmbedderOptions().DisableAutoDownload.Should().BeFalse();

    [Theory]
    [InlineData("default")]
    [InlineData("sentence-transformers/all-MiniLM-L6-v2")]
    [InlineData("gguf:nomic-ai/nomic-embed-text-v1.5-GGUF")]
    public async Task DisableAutoDownload_WithTheModelNotCached_FailsTheLoadInsteadOfDownloading(string modelId)
    {
        var options = new EmbedderOptions { CacheDirectory = _cacheDir, DisableAutoDownload = true };

        var load = () => LocalEmbedder.LoadAsync(modelId, options, cancellationToken: Ct);

        await load.Should().ThrowAsync<ModelNotFoundException>().WithMessage("*downloads are disabled*");
        Directory.Exists(_cacheDir).Should().BeFalse("an offline miss downloads nothing and writes nothing to the cache");
    }

    // The explicit download entry point honours the same flag: with it set it can only confirm a cached model.
    [Fact]
    public async Task DownloadModelAsync_WithDownloadsDisabled_FailsOnAMissInsteadOfDownloading()
    {
        var options = new EmbedderOptions { CacheDirectory = _cacheDir, DisableAutoDownload = true };

        var download = () => LocalEmbedder.DownloadModelAsync("default", options, cancellationToken: Ct);

        await download.Should().ThrowAsync<ModelNotFoundException>().WithMessage("*downloads are disabled*");
        Directory.Exists(_cacheDir).Should().BeFalse();
    }
}
