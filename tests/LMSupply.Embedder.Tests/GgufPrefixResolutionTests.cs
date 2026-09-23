using AwesomeAssertions;

namespace LMSupply.Embedder.Tests;

/// <summary>
/// A GGUF embedding repository takes the query/passage prefixes of the model it was converted from. Before, the GGUF
/// path reported no prefixes, so <c>EmbedQueryAsync</c>/<c>EmbedPassageAsync</c> on e.g.
/// <c>nomic-ai/nomic-embed-text-v1.5-GGUF</c> (the README's own GGUF example) embedded bare text while the ONNX
/// build of the same model prefixed <c>search_query: </c> / <c>search_document: </c>.
/// </summary>
public class GgufPrefixResolutionTests
{
    [Theory]
    [InlineData("nomic-ai/nomic-embed-text-v1.5-GGUF")]
    [InlineData("nomic-ai/nomic-embed-text-v1.5-gguf")]
    [InlineData("nomic-ai/nomic-embed-text-v1.5_GGUF")]
    public void A_GGUF_repository_of_a_catalog_model_takes_its_prefixes(string repo)
    {
        var (query, passage) = LocalEmbedder.ResolveGgufPrefixes(repo);

        query.Should().Be("search_query: ");
        passage.Should().Be("search_document: ");
    }

    [Theory]
    [InlineData("someone/unknown-embedder-GGUF")]   // not in the catalog
    [InlineData("nomic-ai/nomic-embed-text-v1.5")]  // not a GGUF repository name
    [InlineData("model.gguf")]                        // a local file name carries no repository
    public void Anything_else_has_no_prefixes(string repoOrPath)
        => LocalEmbedder.ResolveGgufPrefixes(repoOrPath).Should().Be(((string?)null, (string?)null));
}
