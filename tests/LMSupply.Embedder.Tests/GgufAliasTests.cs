using AwesomeAssertions;
using LMSupply.Embedder.Utils;

namespace LMSupply.Embedder.Tests;

/// <summary>
/// <c>embeddinggemma-2</c> names the GGUF route by intent, the way the other aliases name ONNX models. It behaves as an alias on
/// the entry points a caller reaches before a load (list, describe, probe), carries the prompts the model is used with, and does
/// not change what <c>default</c> or any ONNX alias means: a caller that did not ask for llama-server never gets one.
/// </summary>
public sealed class GgufAliasTests : IDisposable
{
    private const string Alias = "embeddinggemma-2";
    private const string TargetRepo = "ggml-org/embeddinggemma-2-GGUF";

    private readonly string _cacheDir = Path.Combine(Path.GetTempPath(), "lmsupply-embedder-gguf-alias-" + Guid.NewGuid().ToString("N"));

    public void Dispose()
    {
        if (Directory.Exists(_cacheDir))
            Directory.Delete(_cacheDir, recursive: true);
    }

    [Fact]
    public void TheAlias_IsListed_AndPointsAtARepository()
    {
        LocalEmbedder.GetAvailableModels().Should().Contain(Alias);
        LocalEmbedder.GgufAliases[Alias].Target.Should().Be("gguf:" + TargetRepo);
    }

    [Fact]
    public void TheOnnxAliases_KeepTheirMeaning() =>
        LocalEmbedder.GgufAliases.Keys.Should().NotContain(["default", "fast", "quality", "large", "auto"]);

    [Fact]
    public void TheModel_IsUsedWithItsTaskPrompts() =>
        LocalEmbedder.ResolveGgufPrefixes(TargetRepo).Should().Be(
            new PromptPrefixes("task: search result | query: ", "title: none | text: ", null));

    [Fact]
    public void Describe_NamesTheGgufBuildAndTheLicenceOfTheModel()
    {
        var description = LocalEmbedder.Describe(Alias);

        description.Backend.Should().Be(ModelBackend.Gguf);
        description.ResolvedId.Should().Be(TargetRepo);
        description.License.Should().Be("Apache-2.0");
        description.DisplayName.Should().Be("EmbeddingGemma 2 (GGUF)");
    }

    [Fact]
    public async Task TheProbe_FollowsTheAlias()
    {
        var options = new EmbedderOptions { CacheDirectory = _cacheDir };
        (await LocalEmbedder.IsModelDownloadedAsync(Alias, options, cancellationToken: TestContext.Current.CancellationToken)).Should().BeFalse();

        var dir = Path.Combine(_cacheDir, GgufDownloader.LegacyTreeName, TargetRepo.Replace('/', '_'));
        Directory.CreateDirectory(dir);
        File.WriteAllText(Path.Combine(dir, "embeddinggemma-2-Q8_0.gguf"), "GGUF-not-a-real-model");

        (await LocalEmbedder.IsModelDownloadedAsync(Alias, options, cancellationToken: TestContext.Current.CancellationToken)).Should().BeTrue();
    }
}
