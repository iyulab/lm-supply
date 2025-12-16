using FluentAssertions;

namespace LMSupply.Segmenter.Tests;

public class SegmenterOptionsTests
{
    [Fact]
    public void DefaultOptions_ShouldHaveCorrectDefaults()
    {
        var options = new SegmenterOptions();

        options.ModelId.Should().Be("default");
        options.Provider.Should().Be(ExecutionProvider.Auto);
        options.ResizeToOriginal.Should().BeTrue();
        options.DisableAutoDownload.Should().BeFalse();
        options.CacheDirectory.Should().BeNull();
        options.ThreadCount.Should().BeNull();
    }

    [Fact]
    public void Clone_ShouldCreateIndependentCopy()
    {
        var original = new SegmenterOptions
        {
            ModelId = "custom",
            CacheDirectory = "/custom/path",
            Provider = ExecutionProvider.Cpu,
            DisableAutoDownload = true,
            ThreadCount = 4,
            ResizeToOriginal = false
        };

        var cloned = original.Clone();

        cloned.Should().NotBeSameAs(original);
        cloned.ModelId.Should().Be("custom");
        cloned.CacheDirectory.Should().Be("/custom/path");
        cloned.Provider.Should().Be(ExecutionProvider.Cpu);
        cloned.DisableAutoDownload.Should().BeTrue();
        cloned.ThreadCount.Should().Be(4);
        cloned.ResizeToOriginal.Should().BeFalse();
    }

    [Fact]
    public void Clone_ModifyingClone_ShouldNotAffectOriginal()
    {
        var original = new SegmenterOptions { ModelId = "original" };
        var cloned = original.Clone();

        cloned.ModelId = "modified";

        original.ModelId.Should().Be("original");
    }
}
