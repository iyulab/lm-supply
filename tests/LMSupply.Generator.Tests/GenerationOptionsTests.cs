using FluentAssertions;
using LMSupply.Generator.Models;

namespace LMSupply.Generator.Tests;

public class GenerationOptionsTests
{
    [Fact]
    public void Default_ReturnsExpectedValues()
    {
        // Act
        var options = GenerationOptions.Default;

        // Assert
        options.MaxTokens.Should().Be(512);
        options.Temperature.Should().Be(0.7f);
        options.TopP.Should().Be(0.9f);
        options.TopK.Should().Be(50);
        options.RepetitionPenalty.Should().Be(1.1f);
    }

    [Fact]
    public void Creative_HasHigherTemperature()
    {
        // Act
        var options = GenerationOptions.Creative;

        // Assert
        options.Temperature.Should().Be(0.9f);
        options.TopP.Should().Be(0.95f);
        options.TopK.Should().Be(100);
    }

    [Fact]
    public void Precise_HasLowerTemperature()
    {
        // Act
        var options = GenerationOptions.Precise;

        // Assert
        options.Temperature.Should().Be(0.1f);
        options.TopP.Should().Be(0.5f);
        options.TopK.Should().Be(10);
    }

    [Fact]
    public void Default_HasExpectedSamplingOptions()
    {
        // Act
        var options = GenerationOptions.Default;

        // Assert - New options from research-05
        options.DoSample.Should().BeTrue();
        options.NumBeams.Should().Be(1);
        options.PastPresentShareBuffer.Should().BeTrue();
        options.MaxNewTokens.Should().BeNull();
    }

    [Fact]
    public void BeamSearch_Configuration()
    {
        // Arrange
        var options = new GenerationOptions
        {
            NumBeams = 4,
            DoSample = false
        };

        // Assert
        options.NumBeams.Should().Be(4);
        options.DoSample.Should().BeFalse();
    }

    [Fact]
    public void MaxNewTokens_CanBeLimited()
    {
        // Arrange
        var options = new GenerationOptions
        {
            MaxTokens = 2048,
            MaxNewTokens = 100
        };

        // Assert
        options.MaxTokens.Should().Be(2048);
        options.MaxNewTokens.Should().Be(100);
    }
}
