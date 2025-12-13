using FluentAssertions;
using LocalAI.Generator.Models;

namespace LocalAI.Generator.Tests;

public class GeneratorOptionsTests
{
    [Fact]
    public void Default_ReturnsExpectedValues()
    {
        // Act
        var options = GeneratorOptions.Default;

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
        var options = GeneratorOptions.Creative;

        // Assert
        options.Temperature.Should().Be(0.9f);
        options.TopP.Should().Be(0.95f);
        options.TopK.Should().Be(100);
    }

    [Fact]
    public void Precise_HasLowerTemperature()
    {
        // Act
        var options = GeneratorOptions.Precise;

        // Assert
        options.Temperature.Should().Be(0.1f);
        options.TopP.Should().Be(0.5f);
        options.TopK.Should().Be(10);
    }
}
