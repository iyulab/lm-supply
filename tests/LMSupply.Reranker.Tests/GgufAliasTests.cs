using AwesomeAssertions;
using LMSupply.Exceptions;

namespace LMSupply.Reranker.Tests;

[CollectionDefinition(nameof(BuiltInAliasShadowSerialGroup), DisableParallelization = true)]
public sealed class BuiltInAliasShadowSerialGroup;

/// <summary>
/// <c>multilingual-fast</c> names the GGUF route by intent, the way the other aliases name ONNX models. It has
/// to behave as an alias on every entry point — load, download probe, pre-download — and it must not change
/// what <c>multilingual</c> or <c>auto</c> mean: a caller that did not ask for llama-server never gets one.
/// </summary>
/// <remarks>
/// One test here registers a user alias named <c>multilingual-fast</c> on the process-wide registry, shadowing the
/// built-in alias while it runs; a test in another class that resolved the built-in alias at that moment got the
/// user's target instead. The class runs without parallel neighbours.
/// </remarks>
[Collection(nameof(BuiltInAliasShadowSerialGroup))]
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
        LocalReranker.GgufAliases[Alias].Target.Should().Be("gguf:" + TargetRepo);
    }

    [Fact]
    public void TheOnnxAliases_KeepTheirMeaning()
    {
        LocalReranker.GgufAliases.Keys.Should().NotContain(["multilingual", "auto", "default", "quality", "large", "fast"]);
        LocalReranker.Registry.Resolve("multilingual").Id.Should().Be("onnx-community/bge-reranker-v2-m3-ONNX");
        LocalReranker.Registry.Resolve("multilingual").OnnxFile.Should().NotBeNullOrEmpty();
    }

    [Fact]
    public async Task TheProbe_FollowsTheAlias()
    {
        (await LocalReranker.IsModelDownloadedAsync(Alias, new RerankerOptions { CacheDirectory = _cacheDir }, cancellationToken: TestContext.Current.CancellationToken)).Should().BeFalse();

        Place(TargetRepo, "bge-reranker-v2-m3-Q4_K_M.gguf");

        (await LocalReranker.IsModelDownloadedAsync(Alias, new RerankerOptions { CacheDirectory = _cacheDir }, cancellationToken: TestContext.Current.CancellationToken)).Should().BeTrue();
        (await LocalReranker.IsModelDownloadedAsync(Alias.ToUpperInvariant(), new RerankerOptions { CacheDirectory = _cacheDir }, cancellationToken: TestContext.Current.CancellationToken)).Should().BeTrue("aliases are case-insensitive");
        (await LocalReranker.IsModelDownloadedAsync(Alias + ":Q8_0", new RerankerOptions { CacheDirectory = _cacheDir }, cancellationToken: TestContext.Current.CancellationToken)).Should().BeTrue("a quantization qualifier does not change which model this is");
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
    public async Task AUserAlias_OfTheSameName_Wins()
    {
        const string elsewhere = "example/my-own-reranker-GGUF";
        Place(elsewhere, "mine-Q4_K_M.gguf");

        LocalReranker.Registry.RegisterAlias(Alias, "gguf:" + elsewhere);
        try
        {
            (await LocalReranker.IsModelDownloadedAsync(Alias, new RerankerOptions { CacheDirectory = _cacheDir }, cancellationToken: TestContext.Current.CancellationToken)).Should().BeTrue("the caller's alias points at a cached model");
        }
        finally
        {
            LocalReranker.Registry.RemoveAlias(Alias);
        }

        (await LocalReranker.IsModelDownloadedAsync(Alias, new RerankerOptions { CacheDirectory = _cacheDir }, cancellationToken: TestContext.Current.CancellationToken)).Should().BeFalse("with the user alias gone the built-in target is not cached here");
    }

    [Fact]
    public async Task AUserAlias_MayPointAtTheBuiltInAlias()
    {
        Place(TargetRepo, "bge-reranker-v2-m3-Q4_K_M.gguf");

        LocalReranker.Registry.RegisterAlias("my-reranker", Alias);
        try
        {
            (await LocalReranker.IsModelDownloadedAsync("my-reranker", new RerankerOptions { CacheDirectory = _cacheDir }, cancellationToken: TestContext.Current.CancellationToken)).Should().BeTrue();
        }
        finally
        {
            LocalReranker.Registry.RemoveAlias("my-reranker");
        }
    }
}
