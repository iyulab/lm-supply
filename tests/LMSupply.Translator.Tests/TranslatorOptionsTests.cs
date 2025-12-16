using FluentAssertions;

namespace LMSupply.Translator.Tests;

public class TranslatorOptionsTests
{
    [Fact]
    public void DefaultOptions_ShouldHaveCorrectDefaults()
    {
        var options = new TranslatorOptions();

        options.ModelId.Should().Be("default");
        options.Provider.Should().Be(ExecutionProvider.Auto);
        options.DisableAutoDownload.Should().BeFalse();
        options.CacheDirectory.Should().BeNull();
        options.ThreadCount.Should().BeNull();
        options.MaxLength.Should().Be(512);
        options.BeamWidth.Should().Be(4);
        options.UseGreedyDecoding.Should().BeFalse();
    }

    [Fact]
    public void Clone_ShouldCreateIndependentCopy()
    {
        var original = new TranslatorOptions
        {
            ModelId = "ko-en",
            CacheDirectory = "/custom/path",
            Provider = ExecutionProvider.Cpu,
            DisableAutoDownload = true,
            ThreadCount = 4,
            MaxLength = 256,
            BeamWidth = 8,
            UseGreedyDecoding = true
        };

        var cloned = original.Clone();

        cloned.Should().NotBeSameAs(original);
        cloned.ModelId.Should().Be("ko-en");
        cloned.CacheDirectory.Should().Be("/custom/path");
        cloned.Provider.Should().Be(ExecutionProvider.Cpu);
        cloned.DisableAutoDownload.Should().BeTrue();
        cloned.ThreadCount.Should().Be(4);
        cloned.MaxLength.Should().Be(256);
        cloned.BeamWidth.Should().Be(8);
        cloned.UseGreedyDecoding.Should().BeTrue();
    }

    [Fact]
    public void Clone_ModifyingClone_ShouldNotAffectOriginal()
    {
        var original = new TranslatorOptions { ModelId = "original" };
        var cloned = original.Clone();

        cloned.ModelId = "modified";
        cloned.MaxLength = 100;

        original.ModelId.Should().Be("original");
        original.MaxLength.Should().Be(512);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(4)]
    [InlineData(16)]
    public void BeamWidth_ShouldAcceptValidValues(int beamWidth)
    {
        var options = new TranslatorOptions { BeamWidth = beamWidth };
        options.BeamWidth.Should().Be(beamWidth);
    }

    [Theory]
    [InlineData(64)]
    [InlineData(256)]
    [InlineData(1024)]
    public void MaxLength_ShouldAcceptValidValues(int maxLength)
    {
        var options = new TranslatorOptions { MaxLength = maxLength };
        options.MaxLength.Should().Be(maxLength);
    }
}
