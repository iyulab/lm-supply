using AwesomeAssertions;
using LMSupply.Exceptions;
using LMSupply.Generator.Internal.Llama;

namespace LMSupply.Generator.Tests;

/// <summary>
/// <see cref="GeneratorOptions.DisableAutoDownload"/> makes a load fail-closed on the GGUF paths (registry
/// alias, repository id) and on the ONNX download path: the cache is read, never written, no repository is
/// listed, and the refusal comes before any runtime binary (llama-server, GenAI) would be fetched.
/// </summary>
public sealed class OfflineLoadTests : IDisposable
{
    private readonly string _cacheDir = Path.Combine(Path.GetTempPath(), "lmsupply-generator-cache-" + Guid.NewGuid().ToString("N"));

    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    public void Dispose()
    {
        if (Directory.Exists(_cacheDir))
            Directory.Delete(_cacheDir, recursive: true);
    }

    [Fact]
    public void DisableAutoDownload_DefaultsToFalse() =>
        new GeneratorOptions().DisableAutoDownload.Should().BeFalse();

    [Fact]
    public async Task DisableAutoDownload_WithAGgufAliasNotCached_FailsTheLoadInsteadOfDownloading()
    {
        var alias = GgufModelRegistry.GetAliases()[0];
        var options = new GeneratorOptions { CacheDirectory = _cacheDir, DisableAutoDownload = true };

        var load = () => LocalGenerator.LoadAsync(alias, options, cancellationToken: Ct);

        await load.Should().ThrowAsync<ModelNotFoundException>().WithMessage("*downloads are disabled*");
        Directory.Exists(_cacheDir).Should().BeFalse("an offline miss downloads nothing and writes nothing to the cache");
    }

    [Fact]
    public async Task DisableAutoDownload_WithAGgufRepoIdNotCached_FailsTheLoadInsteadOfListingTheRepository()
    {
        var options = new GeneratorOptions { CacheDirectory = _cacheDir, DisableAutoDownload = true };

        var load = () => LocalGenerator.LoadAsync("bartowski/Qwen2.5-0.5B-Instruct-GGUF", options, cancellationToken: Ct);

        await load.Should().ThrowAsync<ModelNotFoundException>().WithMessage("*downloads are disabled*");
        Directory.Exists(_cacheDir).Should().BeFalse();
    }

    // The ONNX download half (shared by LoadAsync and DownloadModelAsync) refuses the same way.
    [Fact]
    public async Task DisableAutoDownload_WithAnOnnxRepoIdNotCached_FailsTheDownloadInsteadOfDiscovering()
    {
        var options = new GeneratorOptions { CacheDirectory = _cacheDir, DisableAutoDownload = true };

        var download = () => LocalGenerator.DownloadModelAsync("microsoft/Phi-3.5-mini-instruct-onnx", options, cancellationToken: Ct);

        await download.Should().ThrowAsync<ModelNotFoundException>();
        Directory.Exists(_cacheDir).Should().BeFalse();
    }

    // A cached GGUF file is what an offline load opens — the file, not a repository listing, decides.
    [Fact]
    public async Task DisableAutoDownload_WithAGgufFileCached_ResolvesThatFileWithoutListingTheRepository()
    {
        const string repoId = "example/some-model-GGUF";
        using var downloader = new GgufModelDownloader(_cacheDir, localFilesOnly: true);
        var seeded = Path.Combine(_cacheDir, "models--example--some-model-GGUF", "snapshots", "main", "some-model-Q4_K_M.gguf");
        Directory.CreateDirectory(Path.GetDirectoryName(seeded)!);
        await File.WriteAllTextAsync(seeded, "placeholder", Ct);

        var resolved = await downloader.DownloadAsync(repoId, cancellationToken: Ct);

        resolved.Should().Be(seeded);
    }
}
