using AwesomeAssertions;
using LMSupply.Download;
using LMSupply.Exceptions;

namespace LMSupply.Captioner.Tests;

/// <summary>
/// <see cref="CaptionerOptions.DisableAutoDownload"/> makes a load fail-closed: the cache is read, never
/// written, and a file that is not there ends the load instead of starting a download. A consumer behind a
/// download-consent boundary relies on the loader for this — its own cache probe can only reconstruct the
/// loader's file list from the outside.
/// </summary>
public sealed class OfflineLoadTests : IDisposable
{
    private readonly string _cacheDir = Path.Combine(Path.GetTempPath(), "lmsupply-captioner-cache-" + Guid.NewGuid().ToString("N"));

    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    public void Dispose()
    {
        if (Directory.Exists(_cacheDir))
            Directory.Delete(_cacheDir, recursive: true);
    }

    [Fact]
    public async Task DisableAutoDownload_WithTheModelNotCached_FailsTheLoadInsteadOfDownloading()
    {
        var options = new CaptionerOptions { CacheDirectory = _cacheDir, DisableAutoDownload = true };

        var load = () => LocalCaptioner.LoadAsync("default", options, cancellationToken: Ct);

        await load.Should().ThrowAsync<ModelNotFoundException>().WithMessage("*downloads are disabled*");
        Directory.Exists(_cacheDir).Should().BeFalse("an offline miss downloads nothing and writes nothing to the cache");
    }

    // The same rule on the repo-id path, which resolves files by discovery rather than by the registry list.
    [Fact]
    public async Task DisableAutoDownload_WithAnUncachedRepoId_FailsTheLoadInsteadOfDownloading()
    {
        var options = new CaptionerOptions { CacheDirectory = _cacheDir, DisableAutoDownload = true };

        var load = () => LocalCaptioner.LoadAsync("Xenova/vit-gpt2-image-captioning", options, cancellationToken: Ct);

        await load.Should().ThrowAsync<ModelNotFoundException>();
        Directory.Exists(_cacheDir).Should().BeFalse("an offline miss downloads nothing and writes nothing to the cache");
    }

    // With every file the loader asks for present, an offline load gets past the download stage: the
    // failure it then hits is the placeholder content, not a refused download. This is what ties the
    // offline check to the loader's own file list — the seam a consumer's external probe cannot see.
    [Fact]
    public async Task DisableAutoDownload_WithTheModelCached_PassesTheDownloadStage_AndWritesNothing()
    {
        var model = LocalCaptioner.Registry.Resolve("default");
        var modelDir = Path.Combine(CacheManager.GetModelDirectory(_cacheDir, model.RepoId), model.Subfolder ?? string.Empty);
        Directory.CreateDirectory(modelDir);
        foreach (var file in new[] { model.EncoderFile, model.DecoderFile, "config.json", "vocab.json", "merges.txt", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json" })
            await File.WriteAllTextAsync(Path.Combine(modelDir, file), "placeholder", Ct);
        var before = Snapshot();

        var options = new CaptionerOptions { CacheDirectory = _cacheDir, DisableAutoDownload = true };
        var load = () => LocalCaptioner.LoadAsync("default", options, cancellationToken: Ct);

        var failure = await load.Should().ThrowAsync<Exception>("placeholder files are not a model");
        failure.Which.Message.Should().NotContain("downloads are disabled", "every file the loader needs is in the cache");
        Snapshot().Should().Equal(before, "an offline load reads the cache and writes nothing to it");
    }

    private List<string> Snapshot() => Directory.EnumerateFiles(_cacheDir, "*", SearchOption.AllDirectories)
        .Select(f => $"{Path.GetRelativePath(_cacheDir, f)} {new FileInfo(f).Length} {new FileInfo(f).LastWriteTimeUtc.Ticks}")
        .Order(StringComparer.Ordinal)
        .ToList();
}
