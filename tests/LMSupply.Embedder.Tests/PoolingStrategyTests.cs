using FluentAssertions;
using LMSupply.Embedder.Pooling;

namespace LMSupply.Embedder.Tests;

public class PoolingStrategyTests
{
    private const int HiddenDim = 4;
    private const int SeqLength = 3;

    [Fact]
    public void MeanPooling_CalculatesCorrectAverage()
    {
        var strategy = new MeanPoolingStrategy();

        // Token embeddings: 3 tokens x 4 dims
        var tokenEmbeddings = new float[]
        {
            1, 2, 3, 4,     // Token 0
            5, 6, 7, 8,     // Token 1
            9, 10, 11, 12   // Token 2
        };
        var attentionMask = new long[] { 1, 1, 1 };
        var result = new float[HiddenDim];

        strategy.Pool(tokenEmbeddings, attentionMask, result, SeqLength, HiddenDim);

        // Expected: (1+5+9)/3, (2+6+10)/3, (3+7+11)/3, (4+8+12)/3
        result[0].Should().BeApproximately(5.0f, 0.00001f);
        result[1].Should().BeApproximately(6.0f, 0.00001f);
        result[2].Should().BeApproximately(7.0f, 0.00001f);
        result[3].Should().BeApproximately(8.0f, 0.00001f);
    }

    [Fact]
    public void MeanPooling_IgnoresPaddingTokens()
    {
        var strategy = new MeanPoolingStrategy();

        var tokenEmbeddings = new float[]
        {
            1, 2, 3, 4,     // Token 0 (real)
            5, 6, 7, 8,     // Token 1 (real)
            100, 100, 100, 100  // Token 2 (padding - should be ignored)
        };
        var attentionMask = new long[] { 1, 1, 0 };
        var result = new float[HiddenDim];

        strategy.Pool(tokenEmbeddings, attentionMask, result, SeqLength, HiddenDim);

        // Expected: (1+5)/2, (2+6)/2, (3+7)/2, (4+8)/2
        result[0].Should().BeApproximately(3.0f, 0.00001f);
        result[1].Should().BeApproximately(4.0f, 0.00001f);
        result[2].Should().BeApproximately(5.0f, 0.00001f);
        result[3].Should().BeApproximately(6.0f, 0.00001f);
    }

    [Fact]
    public void MeanPooling_AllPadding_ReturnsNearZero()
    {
        var strategy = new MeanPoolingStrategy();

        var tokenEmbeddings = new float[]
        {
            1, 2, 3, 4,
            5, 6, 7, 8,
            9, 10, 11, 12
        };
        var attentionMask = new long[] { 0, 0, 0 };
        var result = new float[HiddenDim];

        strategy.Pool(tokenEmbeddings, attentionMask, result, SeqLength, HiddenDim);

        // When all tokens are padding, result should be near zero (divided by epsilon)
        result.Should().AllSatisfy(v => v.Should().BeApproximately(0.0f, 0.0001f));
    }

    [Fact]
    public void ClsPooling_ReturnsFirstTokenEmbedding()
    {
        var strategy = new ClsPoolingStrategy();

        var tokenEmbeddings = new float[]
        {
            1, 2, 3, 4,     // CLS token
            5, 6, 7, 8,
            9, 10, 11, 12
        };
        var attentionMask = new long[] { 1, 1, 1 };
        var result = new float[HiddenDim];

        strategy.Pool(tokenEmbeddings, attentionMask, result, SeqLength, HiddenDim);

        result[0].Should().Be(1.0f);
        result[1].Should().Be(2.0f);
        result[2].Should().Be(3.0f);
        result[3].Should().Be(4.0f);
    }

    [Fact]
    public void MaxPooling_ReturnsMaxPerDimension()
    {
        var strategy = new MaxPoolingStrategy();

        var tokenEmbeddings = new float[]
        {
            1, 10, 3, 4,    // Token 0
            5, 2, 11, 8,    // Token 1
            9, 6, 7, 12     // Token 2
        };
        var attentionMask = new long[] { 1, 1, 1 };
        var result = new float[HiddenDim];

        strategy.Pool(tokenEmbeddings, attentionMask, result, SeqLength, HiddenDim);

        result[0].Should().Be(9.0f);   // max(1,5,9)
        result[1].Should().Be(10.0f);  // max(10,2,6)
        result[2].Should().Be(11.0f);  // max(3,11,7)
        result[3].Should().Be(12.0f);  // max(4,8,12)
    }

    [Fact]
    public void MaxPooling_IgnoresPaddingTokens()
    {
        var strategy = new MaxPoolingStrategy();

        var tokenEmbeddings = new float[]
        {
            1, 2, 3, 4,
            5, 6, 7, 8,
            100, 100, 100, 100  // Padding
        };
        var attentionMask = new long[] { 1, 1, 0 };
        var result = new float[HiddenDim];

        strategy.Pool(tokenEmbeddings, attentionMask, result, SeqLength, HiddenDim);

        result[0].Should().Be(5.0f);
        result[1].Should().Be(6.0f);
        result[2].Should().Be(7.0f);
        result[3].Should().Be(8.0f);
    }

    [Fact]
    public void PoolingFactory_CreatesMeanStrategy()
    {
        var strategy = PoolingFactory.Create(PoolingMode.Mean);
        strategy.Should().BeOfType<MeanPoolingStrategy>();
    }

    [Fact]
    public void PoolingFactory_CreatesClsStrategy()
    {
        var strategy = PoolingFactory.Create(PoolingMode.Cls);
        strategy.Should().BeOfType<ClsPoolingStrategy>();
    }

    [Fact]
    public void PoolingFactory_CreatesMaxStrategy()
    {
        var strategy = PoolingFactory.Create(PoolingMode.Max);
        strategy.Should().BeOfType<MaxPoolingStrategy>();
    }

    [Fact]
    public void MeanPooling_SingleToken()
    {
        var strategy = new MeanPoolingStrategy();
        var tokenEmbeddings = new float[] { 1, 2, 3, 4 };
        var attentionMask = new long[] { 1 };
        var result = new float[HiddenDim];

        strategy.Pool(tokenEmbeddings, attentionMask, result, 1, HiddenDim);

        result[0].Should().Be(1.0f);
        result[1].Should().Be(2.0f);
        result[2].Should().Be(3.0f);
        result[3].Should().Be(4.0f);
    }

    [Fact]
    public void MaxPooling_WithNegativeValues()
    {
        var strategy = new MaxPoolingStrategy();

        var tokenEmbeddings = new float[]
        {
            -5, -3, -1, -2,
            -1, -2, -3, -4
        };
        var attentionMask = new long[] { 1, 1 };
        var result = new float[HiddenDim];

        strategy.Pool(tokenEmbeddings, attentionMask, result, 2, HiddenDim);

        result[0].Should().Be(-1.0f); // max(-5, -1)
        result[1].Should().Be(-2.0f); // max(-3, -2)
        result[2].Should().Be(-1.0f); // max(-1, -3)
        result[3].Should().Be(-2.0f); // max(-2, -4)
    }

    [Fact]
    public void ClsPooling_IgnoresOtherTokens()
    {
        var strategy = new ClsPoolingStrategy();

        var tokenEmbeddings = new float[]
        {
            1, 2, 3, 4,      // CLS token
            100, 200, 300, 400
        };
        var attentionMask = new long[] { 1, 1 };
        var result = new float[HiddenDim];

        strategy.Pool(tokenEmbeddings, attentionMask, result, 2, HiddenDim);

        // Should only return the first token regardless of attention mask
        result[0].Should().Be(1.0f);
        result[1].Should().Be(2.0f);
        result[2].Should().Be(3.0f);
        result[3].Should().Be(4.0f);
    }

    [Fact]
    public void PoolingFactory_ThrowsForInvalidMode()
    {
        var act = () => PoolingFactory.Create((PoolingMode)999);
        act.Should().Throw<ArgumentOutOfRangeException>();
    }
}
