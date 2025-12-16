using FluentAssertions;
using LMSupply.Ocr;

namespace LMSupply.Ocr.Tests;

public class OcrOptionsTests
{
    [Fact]
    public void DefaultOptions_ShouldHaveCorrectDefaults()
    {
        // Act
        var options = new OcrOptions();

        // Assert
        options.LanguageHint.Should().Be("en");
        options.DetectionThreshold.Should().Be(0.5f);
        options.RecognitionThreshold.Should().Be(0.5f);
        options.BinarizationThreshold.Should().Be(0.3f);
        options.MaxCandidates.Should().Be(1000);
        options.UnclipRatio.Should().Be(1.5f);
        options.MinBoxArea.Should().Be(10);
        options.UsePolygon.Should().BeTrue();
        options.Provider.Should().Be(ExecutionProvider.Auto);
        options.CacheDirectory.Should().BeNull();
    }

    [Fact]
    public void LanguageHint_ShouldBeSettable()
    {
        // Arrange
        var options = new OcrOptions();

        // Act
        options.LanguageHint = "ko";

        // Assert
        options.LanguageHint.Should().Be("ko");
    }

    [Fact]
    public void DetectionThreshold_ShouldBeSettable()
    {
        // Arrange
        var options = new OcrOptions();

        // Act
        options.DetectionThreshold = 0.7f;

        // Assert
        options.DetectionThreshold.Should().Be(0.7f);
    }

    [Fact]
    public void RecognitionThreshold_ShouldBeSettable()
    {
        // Arrange
        var options = new OcrOptions();

        // Act
        options.RecognitionThreshold = 0.6f;

        // Assert
        options.RecognitionThreshold.Should().Be(0.6f);
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
        var options = new OcrOptions();

        // Act
        options.Provider = provider;

        // Assert
        options.Provider.Should().Be(provider);
    }

    [Fact]
    public void CacheDirectory_ShouldBeSettable()
    {
        // Arrange
        var options = new OcrOptions();
        var customDir = "/custom/cache/path";

        // Act
        options.CacheDirectory = customDir;

        // Assert
        options.CacheDirectory.Should().Be(customDir);
    }

    [Fact]
    public void BinarizationThreshold_ShouldBeSettable()
    {
        // Arrange
        var options = new OcrOptions();

        // Act
        options.BinarizationThreshold = 0.4f;

        // Assert
        options.BinarizationThreshold.Should().Be(0.4f);
    }

    [Fact]
    public void UnclipRatio_ShouldBeSettable()
    {
        // Arrange
        var options = new OcrOptions();

        // Act
        options.UnclipRatio = 2.0f;

        // Assert
        options.UnclipRatio.Should().Be(2.0f);
    }

    [Fact]
    public void MinBoxArea_ShouldBeSettable()
    {
        // Arrange
        var options = new OcrOptions();

        // Act
        options.MinBoxArea = 20;

        // Assert
        options.MinBoxArea.Should().Be(20);
    }

    [Fact]
    public void UsePolygon_ShouldBeSettable()
    {
        // Arrange
        var options = new OcrOptions();

        // Act
        options.UsePolygon = false;

        // Assert
        options.UsePolygon.Should().BeFalse();
    }
}
