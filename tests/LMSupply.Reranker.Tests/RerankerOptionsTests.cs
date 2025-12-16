using FluentAssertions;

namespace LMSupply.Reranker.Tests;

public class RerankerOptionsTests
{
    [Fact]
    public void DefaultOptions_ShouldHaveCorrectDefaults()
    {
        var options = new RerankerOptions();

        options.ModelId.Should().Be("default");
        options.MaxSequenceLength.Should().BeNull();
        options.CacheDirectory.Should().BeNull();
        options.Provider.Should().Be(ExecutionProvider.Auto);
        options.DisableAutoDownload.Should().BeFalse();
        options.ThreadCount.Should().BeNull();
        options.BatchSize.Should().Be(32);
    }

    [Fact]
    public void Clone_ShouldCreateIndependentCopy()
    {
        var original = new RerankerOptions
        {
            ModelId = "quality",
            MaxSequenceLength = 256,
            CacheDirectory = "/custom/cache",
            Provider = ExecutionProvider.Cuda,
            DisableAutoDownload = true,
            ThreadCount = 4,
            BatchSize = 64
        };

        var clone = original.Clone();

        // Modify original
        original.ModelId = "changed";
        original.BatchSize = 128;

        // Clone should not be affected
        clone.ModelId.Should().Be("quality");
        clone.MaxSequenceLength.Should().Be(256);
        clone.CacheDirectory.Should().Be("/custom/cache");
        clone.Provider.Should().Be(ExecutionProvider.Cuda);
        clone.DisableAutoDownload.Should().BeTrue();
        clone.ThreadCount.Should().Be(4);
        clone.BatchSize.Should().Be(64);
    }

    [Fact]
    public void ExecutionProvider_ShouldHaveAllOptions()
    {
        Enum.GetValues<ExecutionProvider>().Should().HaveCount(5);
        Enum.GetNames<ExecutionProvider>().Should().Contain(["Auto", "Cuda", "DirectML", "CoreML", "Cpu"]);
    }
}
