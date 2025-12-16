using FluentAssertions;
using LMSupply;
using LMSupply.Synthesizer;

namespace LMSupply.Synthesizer.Tests;

public class SynthesizerOptionsTests
{
    [Fact]
    public void SynthesizerOptions_HasCorrectDefaults()
    {
        // Act
        var options = new SynthesizerOptions();

        // Assert
        options.ModelId.Should().Be("default");
        options.Provider.Should().Be(ExecutionProvider.Auto);
        options.CacheDirectory.Should().BeNull();
        options.ThreadCount.Should().BeNull();
    }

    [Fact]
    public void SynthesizerOptions_Clone_CreatesDeepCopy()
    {
        // Arrange
        var original = new SynthesizerOptions
        {
            ModelId = "fast",
            Provider = ExecutionProvider.Cpu,
            CacheDirectory = "/custom/cache",
            ThreadCount = 4
        };

        // Act
        var clone = original.Clone();

        // Assert
        clone.Should().NotBeSameAs(original);
        clone.ModelId.Should().Be(original.ModelId);
        clone.Provider.Should().Be(original.Provider);
        clone.CacheDirectory.Should().Be(original.CacheDirectory);
        clone.ThreadCount.Should().Be(original.ThreadCount);
    }

    [Fact]
    public void SynthesizerOptions_Clone_IsIndependent()
    {
        // Arrange
        var original = new SynthesizerOptions { ModelId = "original" };
        var clone = original.Clone();

        // Act
        clone.ModelId = "modified";

        // Assert
        original.ModelId.Should().Be("original");
        clone.ModelId.Should().Be("modified");
    }
}

public class SynthesizeOptionsTests
{
    [Fact]
    public void SynthesizeOptions_HasCorrectDefaults()
    {
        // Act
        var options = new SynthesizeOptions();

        // Assert
        options.Speed.Should().Be(1.0f);
        options.Pitch.Should().Be(0.0f);
        options.SpeakerId.Should().Be(0);
        options.NoiseScale.Should().BeApproximately(0.667f, 0.001f);
        options.NoiseWidth.Should().BeApproximately(0.8f, 0.001f);
        options.OutputFormat.Should().Be(AudioFormat.Wav);
    }

    [Fact]
    public void SynthesizeOptions_CanBeCustomized()
    {
        // Act
        var options = new SynthesizeOptions
        {
            Speed = 1.5f,
            Pitch = 2.0f,
            SpeakerId = 5,
            NoiseScale = 0.5f,
            NoiseWidth = 0.6f,
            OutputFormat = AudioFormat.RawPcm16
        };

        // Assert
        options.Speed.Should().Be(1.5f);
        options.Pitch.Should().Be(2.0f);
        options.SpeakerId.Should().Be(5);
        options.NoiseScale.Should().Be(0.5f);
        options.NoiseWidth.Should().Be(0.6f);
        options.OutputFormat.Should().Be(AudioFormat.RawPcm16);
    }
}

public class AudioFormatTests
{
    [Fact]
    public void AudioFormat_ContainsExpectedValues()
    {
        // Assert
        Enum.GetValues<AudioFormat>().Should().HaveCount(3);
        Enum.IsDefined(AudioFormat.Wav).Should().BeTrue();
        Enum.IsDefined(AudioFormat.RawPcm16).Should().BeTrue();
        Enum.IsDefined(AudioFormat.RawFloat32).Should().BeTrue();
    }
}
