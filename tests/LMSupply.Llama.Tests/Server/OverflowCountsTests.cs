using LMSupply.Llama.Server;

namespace LMSupply.Llama.Tests.Server;

/// <summary>
/// A server-side overflow carries what the server says about it, so <c>ContextLengthExceededException.TokenCount</c>
/// tells a caller by how much to shorten the request.
/// </summary>
public class OverflowCountsTests
{
    [Fact]
    public void ExceedContextSizeError_ReadsThePromptTokensAndTheServerContext()
    {
        const string body = """
            {"error":{"code":400,"message":"the request exceeds the available context size, try increasing it","type":"exceed_context_size_error","n_prompt_tokens":5321,"n_ctx":4096}}
            """;

        Assert.Equal((5321, 4096), LlamaServerClient.ReadOverflowCounts(body));
    }

    [Theory]
    [InlineData("the prompt is too long")]
    [InlineData("""{"error":{"message":"context length exceeded"}}""")]
    [InlineData("""{"error":"context length exceeded"}""")]
    public void WithoutCounts_ReadsNothing(string body) =>
        Assert.Equal(((int?)null, (int?)null), LlamaServerClient.ReadOverflowCounts(body));
}
