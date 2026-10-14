using AwesomeAssertions;
using LMSupply.Llama.Server;

namespace LMSupply.Embedder.Tests;

public class EmbedderOptionsTests
{
    [Fact]
    public void DefaultValues_AreCorrect()
    {
        var options = new EmbedderOptions();

        options.CacheDirectory.Should().BeNull();
        options.MaxSequenceLength.Should().BeNull("unset: the model decides, then the catalog, then 512");
        options.NormalizeEmbeddings.Should().BeTrue();
        options.Provider.Should().Be(ExecutionProvider.Auto);
        options.PoolingMode.Should().BeNull("unset: the model decides, then the catalog, then Mean");
        EmbedderOptions.DefaultMaxSequenceLength.Should().Be(512);
        options.DoLowerCase.Should().BeTrue();
    }

    [Fact]
    public void Properties_CanBeSet()
    {
        var serverUpdateOptions = new LlamaServerUpdateOptions { PinnedVersion = "b7898" };
        var options = new EmbedderOptions
        {
            CacheDirectory = "/custom/path",
            MaxSequenceLength = 256,
            NormalizeEmbeddings = false,
            Provider = ExecutionProvider.Cuda,
            PoolingMode = PoolingMode.Cls,
            DoLowerCase = false,
            ServerUpdateOptions = serverUpdateOptions
        };

        options.CacheDirectory.Should().Be("/custom/path");
        options.MaxSequenceLength.Should().Be(256);
        options.NormalizeEmbeddings.Should().BeFalse();
        options.Provider.Should().Be(ExecutionProvider.Cuda);
        options.PoolingMode.Should().Be(PoolingMode.Cls);
        options.DoLowerCase.Should().BeFalse();
        options.ServerUpdateOptions.Should().BeSameAs(serverUpdateOptions);
    }

    [Fact]
    public void ServerUpdateOptions_DefaultsToNull()
    {
        new EmbedderOptions().ServerUpdateOptions.Should().BeNull();
    }

    [Fact]
    public void PoolingMode_HasExpectedValues()
    {
        Enum.GetValues<PoolingMode>().Should().HaveCount(4);
        // Ordinals are pinned: a configuration bound by number keeps its meaning (Last was appended in 0.121.0).
        ((int)PoolingMode.Mean).Should().Be(0);
        ((int)PoolingMode.Cls).Should().Be(1);
        ((int)PoolingMode.Max).Should().Be(2);
        ((int)PoolingMode.Last).Should().Be(3);
    }

    [Fact]
    public void ExecutionProvider_HasExpectedValues()
    {
        Enum.GetNames<ExecutionProvider>().Should().Contain("Auto");
        Enum.GetNames<ExecutionProvider>().Should().Contain("Cpu");
    }
}
