using FluentAssertions;
using LMSupply.Runtime;

namespace LMSupply.Generator.Tests;

public class HardwareDetectorTests
{
    [Fact]
    public void GetRecommendation_ReturnsValidRecommendation()
    {
        // Act
        var recommendation = HardwareDetector.GetRecommendation();

        // Assert
        recommendation.Should().NotBeNull();
        recommendation.GpuInfo.Should().NotBeNull();
        recommendation.SystemMemoryBytes.Should().BeGreaterThan(0);
        recommendation.MaxModelParameters.Should().BeGreaterThan(0);
        recommendation.MaxContextLength.Should().BeGreaterThan(0);
        recommendation.RecommendedQuantization.Should().NotBeNullOrEmpty();
        recommendation.RecommendedModels.Should().NotBeEmpty();
    }

    [Fact]
    public void GetBestProvider_ReturnsValidProvider()
    {
        // Act
        var provider = HardwareDetector.GetBestProvider();

        // Assert
        provider.Should().BeOneOf(
            ExecutionProvider.Cpu,
            ExecutionProvider.Cuda,
            ExecutionProvider.DirectML,
            ExecutionProvider.CoreML);
    }

    [Theory]
    [InlineData(ExecutionProvider.Cpu)]
    [InlineData(ExecutionProvider.Cuda)]
    [InlineData(ExecutionProvider.DirectML)]
    [InlineData(ExecutionProvider.CoreML)]
    public void ResolveProvider_WithExplicitProvider_ReturnsSameProvider(ExecutionProvider provider)
    {
        // Act
        var resolved = HardwareDetector.ResolveProvider(provider);

        // Assert
        resolved.Should().Be(provider);
    }

    [Fact]
    public void ResolveProvider_WithAuto_ReturnsDetectedProvider()
    {
        // Arrange
        var expected = HardwareDetector.GetBestProvider();

        // Act
        var resolved = HardwareDetector.ResolveProvider(ExecutionProvider.Auto);

        // Assert
        resolved.Should().Be(expected);
    }

    [Fact]
    public void GetRecommendation_RecommendedModelsContainValidModels()
    {
        // Act
        var recommendation = HardwareDetector.GetRecommendation();

        // Assert
        recommendation.RecommendedModels.Should().AllSatisfy(model =>
        {
            model.Should().Contain("/"); // Should be in format "org/model"
        });
    }

    [Fact]
    public void GetRecommendation_GetSummary_ReturnsFormattedString()
    {
        // Act
        var recommendation = HardwareDetector.GetRecommendation();
        var summary = recommendation.GetSummary();

        // Assert
        summary.Should().Contain("Hardware:");
        summary.Should().Contain("System Memory:");
        summary.Should().Contain("Provider:");
        summary.Should().Contain("Quantization:");
        summary.Should().Contain("Max Context:");
        summary.Should().Contain("Recommended Models:");
    }
}
