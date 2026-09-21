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
    public async Task TheBuiltInAlias_LoadsTheSameModel()
    {
        if (!LocalReranker.IsModelDownloaded("multilingual-fast"))
        {
            Assert.Skip("multilingual-fast is not in the local cache; this test never downloads.");
        }

        var ct = TestContext.Current.CancellationToken;
        await using var reranker = await LocalReranker.LoadAsync(
            "multilingual-fast", new RerankerOptions { DisableAutoDownload = true }, cancellationToken: ct);

        // Korean on purpose: the alias exists for the corpora the English-only defaults rank badly.
        string[] documents =
        [
            "경조사 휴가는 배우자 사망 시 5일, 부모 사망 시 5일을 부여한다.",
            "사내 식당의 점심 메뉴는 매주 월요일에 게시된다.",
        ];

        var ranked = await reranker.RerankAsync("부모님이 돌아가시면 며칠 쉴 수 있나요?", documents, cancellationToken: ct);

        ranked[0].OriginalIndex.Should().Be(0);
        ranked.Should().OnlyContain(r => r.Score > 0f && r.Score < 1f);
        ranked[0].Score.Should().BeGreaterThan(ranked[1].Score * 10f, "the unrelated passage is not a near miss");
    }

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
