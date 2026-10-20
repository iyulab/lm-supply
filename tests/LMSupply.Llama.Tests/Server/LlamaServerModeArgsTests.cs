using AwesomeAssertions;
using LMSupply.Llama.Server;

namespace LMSupply.Llama.Tests.Server;

/// <summary>
/// An embedding or reranking request always processes its whole input, so a slot state it leaves is never read back;
/// builds that put it in the RAM prompt cache anyway fill the cache limit (8 GiB by default) for nothing. Such servers
/// start without that cache, and a generation server keeps llama-server's default.
/// </summary>
public class LlamaServerModeArgsTests
{
    [Fact]
    public void Generation_AddsNothing()
    {
        LlamaServerProcess.ModeArguments(ServerMode.Generation, PoolingType.None).Should().BeEmpty();
    }

    [Fact]
    public void Embedding_TurnsOnEmbeddings_WithoutAPromptCache()
    {
        LlamaServerProcess.ModeArguments(ServerMode.Embedding, PoolingType.None)
            .Should().Equal("--embedding", "--cache-ram", "0");
    }

    [Fact]
    public void Embedding_KeepsTheGivenPooling()
    {
        LlamaServerProcess.ModeArguments(ServerMode.Embedding, PoolingType.Mean)
            .Should().Equal("--embedding", "--cache-ram", "0", "--pooling", "mean");
    }

    [Fact]
    public void Reranking_PoolsByRank_WithoutAPromptCache()
    {
        LlamaServerProcess.ModeArguments(ServerMode.Reranking, PoolingType.None)
            .Should().Equal("--embedding", "--cache-ram", "0", "--pooling", "rank");
    }

    [Fact]
    public void Generation_KeepsAGivenPooling()
    {
        LlamaServerProcess.ModeArguments(ServerMode.Generation, PoolingType.Mean).Should().Equal("--pooling", "mean");
    }
}
