using FluentAssertions;

namespace LMSupply.Embedder.Tests;

public class EmbedderOptionsTests
{
    [Fact]
    public void DefaultValues_AreCorrect()
    {
        var options = new EmbedderOptions();

        options.CacheDirectory.Should().BeNull();
        options.MaxSequenceLength.Should().Be(512);
        options.NormalizeEmbeddings.Should().BeTrue();
        options.Provider.Should().Be(ExecutionProvider.Auto);
        options.PoolingMode.Should().Be(PoolingMode.Mean);
        options.DoLowerCase.Should().BeTrue();
    }

    [Fact]
    public void Properties_CanBeSet()
    {
        var options = new EmbedderOptions
        {
            CacheDirectory = "/custom/path",
            MaxSequenceLength = 256,
            NormalizeEmbeddings = false,
            Provider = ExecutionProvider.Cuda,
            PoolingMode = PoolingMode.Cls,
            DoLowerCase = false
        };

        options.CacheDirectory.Should().Be("/custom/path");
        options.MaxSequenceLength.Should().Be(256);
        options.NormalizeEmbeddings.Should().BeFalse();
        options.Provider.Should().Be(ExecutionProvider.Cuda);
        options.PoolingMode.Should().Be(PoolingMode.Cls);
        options.DoLowerCase.Should().BeFalse();
    }

    [Fact]
    public void PoolingMode_HasExpectedValues()
    {
        Enum.GetValues<PoolingMode>().Should().HaveCount(3);
        Enum.GetNames<PoolingMode>().Should().Contain(["Mean", "Cls", "Max"]);
    }

    [Fact]
    public void ExecutionProvider_HasExpectedValues()
    {
        Enum.GetNames<ExecutionProvider>().Should().Contain("Auto");
        Enum.GetNames<ExecutionProvider>().Should().Contain("Cpu");
    }
}
