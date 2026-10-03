using AwesomeAssertions;

namespace LMSupply.Embedder.Tests;

/// <summary>
/// <see cref="LocalEmbedder.Describe"/> names and licenses what a load of an id would open, a GGUF build included — it
/// reports the licence of the model it was converted from when the catalog knows that model.
/// </summary>
public sealed class DescribeTests
{
    [Fact]
    public void ACatalogedAlias_IsDescribedFromItsRegistryEntry()
    {
        var entry = LocalEmbedder.Registry.Resolve("default");

        var d = LocalEmbedder.Describe("default");

        d.Backend.Should().Be(ModelBackend.Onnx);
        d.ResolvedId.Should().Be(entry.RepoId);
        d.License.Should().Be(entry.License).And.NotBeNullOrEmpty();
        d.DisplayName.Should().Be(entry.RepoId.Split('/')[^1]);
        d.CatalogEntry.Should().Be(entry);
    }

    [Fact]
    public void AGgufBuildOfACatalogedModel_CarriesThatModelsLicence()
    {
        var source = LocalEmbedder.Registry.Resolve("nomic-ai/nomic-embed-text-v1.5");

        var d = LocalEmbedder.Describe("gguf:nomic-ai/nomic-embed-text-v1.5-GGUF");

        d.Backend.Should().Be(ModelBackend.Gguf);
        d.ResolvedId.Should().Be("nomic-ai/nomic-embed-text-v1.5-GGUF");
        d.DisplayName.Should().Be("nomic-embed-text-v1.5 (GGUF)");
        d.License.Should().Be(source.License).And.NotBeNullOrEmpty();
        d.CatalogEntry.Should().BeNull();
    }

    [Fact]
    public void AUserAlias_ToAGgufRepository_IsFollowed()
    {
        LocalEmbedder.Registry.RegisterAlias("describe-my-embedder", "gguf:nomic-ai/nomic-embed-text-v1.5-GGUF");
        try
        {
            LocalEmbedder.Describe("describe-my-embedder").Backend.Should().Be(ModelBackend.Gguf);
        }
        finally
        {
            LocalEmbedder.Registry.RemoveAlias("describe-my-embedder");
        }
    }

    [Theory]
    [InlineData("gguf:someone/unknown-embedder-GGUF", ModelBackend.Gguf, "unknown-embedder-GGUF")]
    [InlineData("someone/unknown-embedder", ModelBackend.Onnx, "unknown-embedder")]
    public void AnUncataloguedModel_IsDescribedByItsOwnName_WithoutALicence(string id, ModelBackend backend, string name)
    {
        var d = LocalEmbedder.Describe(id);

        d.Backend.Should().Be(backend);
        d.DisplayName.Should().Be(name);
        d.License.Should().BeNull();
    }
}
