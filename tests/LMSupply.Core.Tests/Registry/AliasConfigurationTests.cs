using System.Text.Json;
using AwesomeAssertions;

namespace LMSupply.Core.Tests.Registry;

// Reads the env-sensitive DefaultFilePath — must serialize with every class that
// mutates LMSUPPLY_* env vars (see the collection rule in LMSupplyCachePathsTests).
[Collection("lmsupply-cache-env")]
public class AliasConfigurationTests
{
    [Fact]
    public void ParseAliasJson_ValidJson_ReturnsDomainAliases()
    {
        var json = """
        {
            "embedder": {
                "my-embed": "sentence-transformers/all-MiniLM-L6-v2"
            },
            "reranker": {
                "my-rerank": "BAAI/bge-reranker-base"
            }
        }
        """;

        var result = AliasConfiguration.ParseAliasJson(json);

        result.Should().ContainKey("embedder");
        result["embedder"].Should().ContainKey("my-embed");
        result["embedder"]["my-embed"].Should().Be("sentence-transformers/all-MiniLM-L6-v2");
        result["reranker"]["my-rerank"].Should().Be("BAAI/bge-reranker-base");
    }

    [Fact]
    public void ParseAliasJson_EmptyJson_ReturnsEmpty()
    {
        var result = AliasConfiguration.ParseAliasJson("{}");
        result.Should().BeEmpty();
    }

    [Fact]
    public void ParseAliasJson_InvalidJson_ThrowsJsonException()
    {
        var act = () => AliasConfiguration.ParseAliasJson("not json");
        act.Should().Throw<JsonException>();
    }

    [Fact]
    public void ParseAliasJson_UnknownDomain_IsIncluded()
    {
        var json = """{ "unknowndomain": { "a": "b" } }""";
        var result = AliasConfiguration.ParseAliasJson(json);
        result.Should().ContainKey("unknowndomain");
    }

    [Fact]
    public void DefaultFilePath_ReturnsLmsupplyPath()
    {
        var path = AliasConfiguration.DefaultFilePath;
        path.Should().Contain(".lmsupply");
        path.Should().EndWith("aliases.json");
    }

    [Fact]
    public void LoadFromFile_NonExistentFile_ReturnsEmpty()
    {
        var result = AliasConfiguration.LoadFromFile("/nonexistent/path/aliases.json");
        result.Should().BeEmpty();
    }
}
