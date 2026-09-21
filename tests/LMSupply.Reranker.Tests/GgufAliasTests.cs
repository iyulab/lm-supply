using AwesomeAssertions;
using LMSupply.Exceptions;

namespace LMSupply.Reranker.Tests;

/// <summary>
/// <c>multilingual-fast</c> names the GGUF route by intent, the way the other aliases name ONNX models. It has
/// to behave as an alias on every entry point — load, download probe, pre-download — and it must not change
/// what <c>multilingual</c> or <c>auto</c> mean: a caller that did not ask for llama-server never gets one.
/// </summary>
public sealed class GgufAliasTests : IDisposable
{
    private const string Alias = "multilingual-fast";
    private const string TargetRepo = "gpustack/bge-reranker-v2-m3-GGUF";

    private readonly string _cacheDir = Path.Combine(Path.GetTempPath(), "lmsupply-reranker-gguf-alias-" + Guid.NewGuid().ToString("N"));

    public void Dispose()
    {
        if (Directory.Exists(_cacheDir))
            Directory.Delete(_cacheDir, recursive: true);
    }

    private void Place(string repo, string fileName)
    {
        var dir = Path.Combine(_cacheDir, "gguf-rerankers", repo.Replace('/', '_'));
        Directory.CreateDirectory(dir);
        File.WriteAllText(Path.Combine(dir, fileName), "GGUF-not-a-real-model");
    }

    [Fact]
    public void TheAlias_IsListed_AndPointsAtARepositoryNotAtAnOnnxEntry()
    {
        LocalReranker.GetAvailableModels().Should().Contain(Alias);
        LocalReranker.GgufAliases[Alias].Should().Be("gguf:" + TargetRepo);
    }

    [Fact]
    public void TheOnnxAliases_KeepTheirMeaning()
    {
        LocalReranker.GgufAliases.Keys.Should().NotContain(["multilingual", "auto", "default", "quality", "large", "fast"]);
        LocalReranker.Registry.Resolve("multilingual").Id.Should().Be("onnx-community/bge-reranker-v2-m3-ONNX");
        LocalReranker.Registry.Resolve("multilingual").OnnxFile.Should().NotBeNullOrEmpty();
    }

    [Fact]
    public void TheProbe_FollowsTheAlias()
    {
        LocalReranker.IsModelDownloaded(Alias, _cacheDir).Should().BeFalse();

        Place(TargetRepo, "bge-reranker-v2-m3-Q4_K_M.gguf");

        LocalReranker.IsModelDownloaded(Alias, _cacheDir).Should().BeTrue();
        LocalReranker.IsModelDownloaded(Alias.ToUpperInvariant(), _cacheDir).Should().BeTrue("aliases are case-insensitive");
        LocalReranker.IsModelDownloaded(Alias + ":Q8_0", _cacheDir).Should().BeTrue("a quantization qualifier does not change which model this is");
    }

    [Fact]
    public async Task AnOfflineLoad_OfTheUncachedAlias_IsRefusedByTheGgufRoute()
    {
        var options = new RerankerOptions { CacheDirectory = _cacheDir, DisableAutoDownload = true };

        var load = () => LocalReranker.LoadAsync(Alias, options, cancellationToken: TestContext.Current.CancellationToken);
        var download = () => LocalReranker.DownloadModelAsync(Alias, new RerankerOptions { CacheDirectory = _cacheDir, DisableAutoDownload = true },
            cancellationToken: TestContext.Current.CancellationToken);

        // The wording is the GGUF downloader's: the alias reached it rather than the ONNX registry.
        (await load.Should().ThrowAsync<ModelNotFoundException>()).WithMessage($"*{TargetRepo}*downloads are disabled*");
        (await download.Should().ThrowAsync<ModelNotFoundException>()).WithMessage($"*{TargetRepo}*downloads are disabled*");
    }

    [Fact]
    public void AUserAlias_OfTheSameName_Wins()
    {
        const string elsewhere = "example/my-own-reranker-GGUF";
        Place(elsewhere, "mine-Q4_K_M.gguf");

        LocalReranker.Registry.RegisterAlias(Alias, "gguf:" + elsewhere);
        try
        {
            LocalReranker.IsModelDownloaded(Alias, _cacheDir).Should().BeTrue("the caller's alias points at a cached model");
        }
        finally
        {
            LocalReranker.Registry.RemoveAlias(Alias);
        }

        LocalReranker.IsModelDownloaded(Alias, _cacheDir).Should().BeFalse("with the user alias gone the built-in target is not cached here");
    }

    [Fact]
    public void AUserAlias_MayPointAtTheBuiltInAlias()
    {
        Place(TargetRepo, "bge-reranker-v2-m3-Q4_K_M.gguf");

        LocalReranker.Registry.RegisterAlias("my-reranker", Alias);
        try
        {
            LocalReranker.IsModelDownloaded("my-reranker", _cacheDir).Should().BeTrue();
        }
        finally
        {
            LocalReranker.Registry.RemoveAlias("my-reranker");
        }
    }
}
