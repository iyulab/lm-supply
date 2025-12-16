using FluentAssertions;
using LMSupply.Reranker.Core;

namespace LMSupply.Reranker.Tests;

public class ScoreNormalizerTests
{
    [Theory]
    [InlineData(0f, 0.5f)]
    [InlineData(1f, 0.7310586f)]
    [InlineData(-1f, 0.26894143f)]
    [InlineData(10f, 0.9999546f)]
    [InlineData(-10f, 0.0000454f)]
    public void Sigmoid_ShouldReturnCorrectProbability(float logit, float expected)
    {
        var result = ScoreNormalizer.Sigmoid(logit);
        result.Should().BeApproximately(expected, 0.0001f);
    }

    [Fact]
    public void Sigmoid_VeryLargePositive_ShouldReturn1()
    {
        var result = ScoreNormalizer.Sigmoid(100f);
        result.Should().Be(1f);
    }

    [Fact]
    public void Sigmoid_VeryLargeNegative_ShouldReturn0()
    {
        var result = ScoreNormalizer.Sigmoid(-100f);
        result.Should().Be(0f);
    }

    [Fact]
    public void SigmoidArray_ShouldNormalizeAllValues()
    {
        var logits = new float[] { -1f, 0f, 1f };
        var results = ScoreNormalizer.Sigmoid(logits);

        results.Should().HaveCount(3);
        results[0].Should().BeApproximately(0.269f, 0.001f);
        results[1].Should().BeApproximately(0.5f, 0.001f);
        results[2].Should().BeApproximately(0.731f, 0.001f);
    }

    [Fact]
    public void SigmoidInPlace_ShouldModifyArray()
    {
        var logits = new float[] { -1f, 0f, 1f };
        ScoreNormalizer.SigmoidInPlace(logits);

        logits[0].Should().BeApproximately(0.269f, 0.001f);
        logits[1].Should().BeApproximately(0.5f, 0.001f);
        logits[2].Should().BeApproximately(0.731f, 0.001f);
    }

    [Fact]
    public void SoftmaxPositive_ShouldReturnPositiveClassProbability()
    {
        var result = ScoreNormalizer.SoftmaxPositive(0f, 1f);
        result.Should().BeApproximately(0.731f, 0.001f);

        var result2 = ScoreNormalizer.SoftmaxPositive(1f, 0f);
        result2.Should().BeApproximately(0.269f, 0.001f);

        var result3 = ScoreNormalizer.SoftmaxPositive(0f, 0f);
        result3.Should().BeApproximately(0.5f, 0.001f);
    }

    [Fact]
    public void MinMaxNormalize_EmptyArray_ShouldReturnEmpty()
    {
        var result = ScoreNormalizer.MinMaxNormalize([]);
        result.Should().BeEmpty();
    }

    [Fact]
    public void MinMaxNormalize_SingleValue_ShouldReturn1()
    {
        var result = ScoreNormalizer.MinMaxNormalize([5f]);
        result.Should().Equal([1f]);
    }

    [Fact]
    public void MinMaxNormalize_ShouldNormalizeTo01Range()
    {
        var scores = new float[] { 0f, 50f, 100f };
        var result = ScoreNormalizer.MinMaxNormalize(scores);

        result[0].Should().Be(0f);
        result[1].Should().Be(0.5f);
        result[2].Should().Be(1f);
    }

    [Fact]
    public void MinMaxNormalize_AllSameValues_ShouldReturn05()
    {
        var scores = new float[] { 5f, 5f, 5f };
        var result = ScoreNormalizer.MinMaxNormalize(scores);

        result.Should().AllBeEquivalentTo(0.5f);
    }
}
