using AwesomeAssertions;
using LMSupply.Embedder.Inference;
using LMSupply.Llama.Server;

namespace LMSupply.Embedder.Tests;

/// <summary>
/// A GGUF embedder pools the way its file says when the options leave pooling unset, and a pooling llama-server cannot
/// do is refused rather than replaced. Before, an unset pooling was always mean (a CLS model such as bge-m3 became a
/// different vector space) and Max was sent as last-token.
/// </summary>
public sealed class GgufEmbedderPoolingTests
{
    [Theory]
    [InlineData(1, PoolingMode.Mean)]
    [InlineData(2, PoolingMode.Cls)]
    [InlineData(3, PoolingMode.Last)]
    public void ThePoolingTheFileNames_IsUsed(int poolingType, PoolingMode expected)
        => LlamaServerEmbeddingModel.PoolingFromFile(poolingType).Should().Be(expected);

    [Theory]
    [InlineData(null)]
    [InlineData(0)]
    [InlineData(4)]
    [InlineData(99)]
    public void NoneRankOrAnUnknownValue_LeavesTheDefault(int? poolingType)
        => LlamaServerEmbeddingModel.PoolingFromFile(poolingType).Should().BeNull();

    [Theory]
    [InlineData(PoolingMode.Mean, PoolingType.Mean)]
    [InlineData(PoolingMode.Cls, PoolingType.Cls)]
    [InlineData(PoolingMode.Last, PoolingType.Last)]
    public void EachSupportedPooling_IsSentAsItself(PoolingMode mode, PoolingType expected)
        => LlamaServerEmbeddingModel.ServerPooling(mode).Should().Be(expected);

    [Fact]
    public void MaxPooling_IsRefused_NotSentAsAnotherPooling()
    {
        var act = () => LlamaServerEmbeddingModel.ServerPooling(PoolingMode.Max);

        act.Should().Throw<NotSupportedException>().WithMessage("*Max*mean, CLS or last token*");
    }
}
