using FluentAssertions;
using LMSupply.Captioner;

namespace LMSupply.Captioner.Tests;

public class CaptionerOptionsTests
{
    [Fact]
    public void DefaultOptions_ShouldHaveCorrectDefaults()
    {
        // Act
        var options = new CaptionerOptions();

        // Assert
        options.MaxLength.Should().Be(50);
        options.NumBeams.Should().Be(1);
        options.Temperature.Should().Be(1.0f);
        options.Prompt.Should().BeNull();
        options.Provider.Should().Be(ExecutionProvider.Auto);
        options.CacheDirectory.Should().BeNull();
    }

    [Fact]
    public void MaxLength_ShouldBeSettable()
    {
        // Arrange
        var options = new CaptionerOptions();

        // Act
        options.MaxLength = 100;

        // Assert
        options.MaxLength.Should().Be(100);
    }

    [Fact]
    public void NumBeams_ShouldBeSettable()
    {
        // Arrange
        var options = new CaptionerOptions();

        // Act
        options.NumBeams = 5;

        // Assert
        options.NumBeams.Should().Be(5);
    }

    [Fact]
    public void Temperature_ShouldBeSettable()
    {
        // Arrange
        var options = new CaptionerOptions();

        // Act
        options.Temperature = 0.7f;

        // Assert
        options.Temperature.Should().Be(0.7f);
    }

    [Fact]
    public void Prompt_ShouldBeSettable()
    {
        // Arrange
        var options = new CaptionerOptions();

        // Act
        options.Prompt = "A photo of";

        // Assert
        options.Prompt.Should().Be("A photo of");
    }

    [Theory]
    [InlineData(ExecutionProvider.Auto)]
    [InlineData(ExecutionProvider.Cpu)]
    [InlineData(ExecutionProvider.Cuda)]
    [InlineData(ExecutionProvider.DirectML)]
    [InlineData(ExecutionProvider.CoreML)]
    public void Provider_ShouldSupportAllValues(ExecutionProvider provider)
    {
        // Arrange
        var options = new CaptionerOptions();

        // Act
        options.Provider = provider;

        // Assert
        options.Provider.Should().Be(provider);
    }

    [Fact]
    public void CacheDirectory_ShouldBeSettable()
    {
        // Arrange
        var options = new CaptionerOptions();
        var customDir = "/custom/cache/path";

        // Act
        options.CacheDirectory = customDir;

        // Assert
        options.CacheDirectory.Should().Be(customDir);
    }
}
