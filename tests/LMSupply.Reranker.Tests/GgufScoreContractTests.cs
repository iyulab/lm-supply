using AwesomeAssertions;
using LMSupply.Llama.Server;
using LMSupply.Reranker.Core;
using LMSupply.Reranker.Inference;

namespace LMSupply.Reranker.Tests;

/// <summary>
/// <see cref="RankedResult.Score"/> is documented as a relevance between 0 and 1. The ONNX cross-encoder
/// path has always delivered that; the GGUF path handed back llama-server's raw logit, so a threshold a
/// caller calibrated on one backend silently changed meaning on the other — and a downstream "drop
/// everything under 0" default started dropping most candidates. These facts pin the GGUF side to the
/// same scale, from logits shaped like the ones a real rank-pooling server returns.
/// </summary>
public sealed class GgufScoreContractTests
{
    private static readonly string[] Documents = ["answer", "near miss", "off topic"];

    // Server order is not document order, and most logits are negative — as measured on a real model.
    private static readonly RerankResult[] ServerRows =
    [
        new() { Index = 2, RelevanceScore = -6.1f },
        new() { Index = 0, RelevanceScore = 2.813f },
        new() { Index = 1, RelevanceScore = -3.2f },
    ];

    [Fact]
    public void RankedScores_AreOnTheDocumentedScale_AndKeepTheLogitRanking()
    {
        var ranked = LlamaServerRerankerModel.ToRankedResults(ServerRows, Documents, topK: null);

        ranked.Select(r => r.OriginalIndex).Should().Equal(0, 1, 2);
        ranked.Select(r => r.Document).Should().Equal("answer", "near miss", "off topic");
        ranked.Should().OnlyContain(r => r.Score > 0f && r.Score < 1f);

        ranked[0].Score.Should().BeApproximately(ScoreNormalizer.Sigmoid(2.813f), 1e-6f);
        ranked[0].Score.Should().BeGreaterThan(0.9f, "a confident match reads like the ONNX path's confident match");
        ranked[2].Score.Should().BeLessThan(0.01f);
    }

    [Fact]
    public void ANegativeLogit_IsStillAPositiveScore()
    {
        // The downstream default this broke: a filter of "score >= 0" is a no-op on the documented scale
        // and removes every negative-logit candidate on the raw one.
        var ranked = LlamaServerRerankerModel.ToRankedResults(ServerRows, Documents, topK: null);

        ranked.Count(r => r.Score >= 0f).Should().Be(ServerRows.Length);
    }

    [Fact]
    public void TopK_CutsAfterRanking()
    {
        var ranked = LlamaServerRerankerModel.ToRankedResults(ServerRows, Documents, topK: 1);

        ranked.Should().ContainSingle().Which.OriginalIndex.Should().Be(0);
    }

    [Fact]
    public void Scores_ComeBackInDocumentOrder_OnTheSameScale()
    {
        var scores = LlamaServerRerankerModel.ToScores(ServerRows, Documents.Length);

        scores.Should().HaveCount(3);
        scores[0].Should().BeApproximately(ScoreNormalizer.Sigmoid(2.813f), 1e-6f);
        scores[1].Should().BeApproximately(ScoreNormalizer.Sigmoid(-3.2f), 1e-6f);
        scores[2].Should().BeApproximately(ScoreNormalizer.Sigmoid(-6.1f), 1e-6f);
    }

    [Fact]
    public void TheDeadModelSignature_IsReadFromTheLogits()
    {
        RerankResult[] dead =
        [
            new() { Index = 0, RelevanceScore = 1e-6f },
            new() { Index = 1, RelevanceScore = -3e-5f },
        ];

        LlamaServerRerankerModel.AllLogitsNearZero(dead).Should().BeTrue();
        LlamaServerRerankerModel.AllLogitsNearZero(ServerRows).Should().BeFalse();
        LlamaServerRerankerModel.AllLogitsNearZero([]).Should().BeFalse();

        // After the sigmoid the same model reads 0.5 everywhere — which is why the test is not on the scores.
        LlamaServerRerankerModel.ToScores(dead, 2).Should().OnlyContain(s => MathF.Abs(s - 0.5f) < 1e-3f);
    }
}
