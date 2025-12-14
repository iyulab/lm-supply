using FluentAssertions;
using LocalAI;

namespace LocalAI.Transcriber.Tests;

public class TranscriberOptionsTests
{
    [Fact]
    public void DefaultOptions_ShouldHaveExpectedDefaults()
    {
        var options = new TranscriberOptions();

        options.ModelId.Should().Be("default");
        options.Provider.Should().Be(ExecutionProvider.Auto);
        options.CacheDirectory.Should().BeNull();
        options.ThreadCount.Should().BeNull();
    }

    [Fact]
    public void Clone_ShouldCreateIndependentCopy()
    {
        var original = new TranscriberOptions
        {
            ModelId = "custom",
            Provider = ExecutionProvider.Cpu,
            CacheDirectory = "/custom/cache",
            ThreadCount = 4
        };

        var clone = original.Clone();

        clone.Should().NotBeSameAs(original);
        clone.ModelId.Should().Be(original.ModelId);
        clone.Provider.Should().Be(original.Provider);
        clone.CacheDirectory.Should().Be(original.CacheDirectory);
        clone.ThreadCount.Should().Be(original.ThreadCount);
    }

    [Fact]
    public void Clone_ModifyingClone_ShouldNotAffectOriginal()
    {
        var original = new TranscriberOptions { ModelId = "original" };
        var clone = original.Clone();

        clone.ModelId = "modified";

        original.ModelId.Should().Be("original");
        clone.ModelId.Should().Be("modified");
    }

    [Theory]
    [InlineData(ExecutionProvider.Auto)]
    [InlineData(ExecutionProvider.Cpu)]
    [InlineData(ExecutionProvider.Cuda)]
    [InlineData(ExecutionProvider.DirectML)]
    public void Provider_ShouldAcceptAllValidValues(ExecutionProvider provider)
    {
        var options = new TranscriberOptions { Provider = provider };

        options.Provider.Should().Be(provider);
    }
}
