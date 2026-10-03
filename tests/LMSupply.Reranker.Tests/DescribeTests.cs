using AwesomeAssertions;

namespace LMSupply.Reranker.Tests;

/// <summary>
/// <see cref="LocalReranker.Describe"/> names and licenses what a load of an id would open — the GGUF route included,
/// which has no registry entry — so a consent screen states it from the library rather than from text of its own.
/// </summary>
public sealed class DescribeTests
{
    private const string GgufRepo = "gpustack/bge-reranker-v2-m3-GGUF";

    [Fact]
    public void TheBuiltInGgufAlias_IsDescribedWithANameAndTheSourceModelsLicence()
    {
        var d = LocalReranker.Describe("multilingual-fast");

        d.RequestedId.Should().Be("multilingual-fast");
        d.ResolvedId.Should().Be(GgufRepo);
        d.Backend.Should().Be(ModelBackend.Gguf);
        d.DisplayName.Should().Be("BGE Reranker v2 M3 (GGUF Q4_K_M)");
        d.License.Should().Be("Apache-2.0", "the quantized build carries bge-reranker-v2-m3's licence");
        d.CatalogEntry.Should().BeNull("a GGUF build has no ONNX registry entry");
    }

    [Fact]
    public void TheGgufRepository_ByItsOwnId_IsDescribedTheSame()
    {
        var byAlias = LocalReranker.Describe("multilingual-fast");
        var byRepo = LocalReranker.Describe("gguf:" + GgufRepo);

        (byRepo.ResolvedId, byRepo.Backend, byRepo.DisplayName, byRepo.License)
            .Should().Be((byAlias.ResolvedId, byAlias.Backend, byAlias.DisplayName, byAlias.License));
    }

    [Theory]
    [InlineData("default")]
    [InlineData("default:fp16")] // a variant qualifier does not change which model this is
    [InlineData("multilingual")]
    [InlineData("quality")]
    public void AnOnnxAlias_IsDescribedFromItsRegistryEntry(string alias)
    {
        var entry = LocalReranker.Registry.Resolve(alias.Split(':')[0]);

        var d = LocalReranker.Describe(alias);

        d.Backend.Should().Be(ModelBackend.Onnx);
        d.ResolvedId.Should().Be(entry.Id);
        d.DisplayName.Should().Be(entry.DisplayName);
        d.License.Should().Be(entry.License).And.NotBeNullOrEmpty();
        d.CatalogEntry.Should().Be(entry);
    }

    [Fact]
    public void AUserAlias_IsFollowed_ToTheGgufRoute()
    {
        LocalReranker.Registry.RegisterAlias("describe-my-reranker", "multilingual-fast");
        try
        {
            var d = LocalReranker.Describe("describe-my-reranker");

            d.Backend.Should().Be(ModelBackend.Gguf);
            d.License.Should().Be("Apache-2.0");
            d.RequestedId.Should().Be("describe-my-reranker");
        }
        finally
        {
            LocalReranker.Registry.RemoveAlias("describe-my-reranker");
        }
    }

    [Fact]
    public void Auto_IsDescribedAsWhatItBecomesOnThisHost()
    {
        // Per host, like the download size: the GGUF build where a llama-server binary is cached on a Medium tier, an
        // ONNX model otherwise. Either way the answer names a model and a licence.
        var d = LocalReranker.Describe("auto");

        d.License.Should().NotBeNullOrEmpty();
        d.DisplayName.Should().NotBeNullOrEmpty();
        (d.Backend == ModelBackend.Gguf).Should().Be(d.ResolvedId == GgufRepo);
    }

    [Fact]
    public void AnUncataloguedRepository_IsDescribedByItsOwnName_WithoutALicence()
    {
        var d = LocalReranker.Describe("someone/some-reranker");

        d.Backend.Should().Be(ModelBackend.Onnx);
        d.ResolvedId.Should().Be("someone/some-reranker");
        d.DisplayName.Should().Be("some-reranker");
        d.License.Should().BeNull("no licence is invented for a model the catalog does not know");
        d.CatalogEntry.Should().BeNull();
    }
}
