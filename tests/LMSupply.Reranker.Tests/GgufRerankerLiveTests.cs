using AwesomeAssertions;

namespace LMSupply.Reranker.Tests;

/// <summary>
/// The score contract against a real llama-server and a real quantized cross-encoder. Runs only where the
/// model is already in the cache — it never downloads — so it is a local check, not a CI one. It is what
/// would notice llama-server starting to normalize on its side (scores would bunch up between 0.5 and 0.73).
/// </summary>
[Trait("Category", "Integration")]
[Trait("Category", "LocalOnly")]
[Trait("Category", "GGUF")]
public sealed class GgufRerankerLiveTests
{
    private const string Model = "gguf:gpustack/bge-reranker-v2-m3-GGUF";

    [Fact]
    public async Task ARealGgufReranker_ScoresOnTheDocumentedScale()
    {
        if (!LocalReranker.IsModelDownloaded(Model))
        {
            Assert.Skip($"{Model} is not in the local cache; this test never downloads.");
        }

        var ct = TestContext.Current.CancellationToken;
        await using var reranker = await LocalReranker.LoadAsync(
            Model, new RerankerOptions { DisableAutoDownload = true }, cancellationToken: ct);

        string[] documents =
        [
            "Paris is the capital and most populous city of France.",
            "The mitochondrion is the organelle that produces most of a cell's ATP.",
            "Lyon is a city in France known for its cuisine.",
        ];

        var ranked = await reranker.RerankAsync("What is the capital of France?", documents, cancellationToken: ct);
        var scores = await reranker.ScoreAsync("What is the capital of France?", documents, ct);

        ranked[0].OriginalIndex.Should().Be(0);
        ranked.Should().OnlyContain(r => r.Score > 0f && r.Score < 1f);
        ranked[0].Score.Should().BeGreaterThan(0.9f, "the answer is in the first document");
        scores[1].Should().BeLessThan(0.05f, "the second document is about cell biology");
        scores[0].Should().BeApproximately(ranked[0].Score, 1e-4f, "both entry points are on one scale");
    }
}
