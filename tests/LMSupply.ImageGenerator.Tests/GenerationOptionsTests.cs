using FluentAssertions;

namespace LMSupply.ImageGenerator.Tests;

public class GenerationOptionsTests
{
    [Fact]
    public void Validate_WithDefaultOptions_Succeeds()
    {
        // Arrange
        var options = new GenerationOptions();

        // Act & Assert
        var act = () => options.Validate();
        act.Should().NotThrow();
    }

    [Theory]
    [InlineData(512, 512)]
    [InlineData(768, 768)]
    [InlineData(1024, 1024)]
    [InlineData(512, 768)]
    public void Validate_WithValidDimensions_Succeeds(int width, int height)
    {
        // Arrange
        var options = new GenerationOptions
        {
            Width = width,
            Height = height
        };

        // Act & Assert
        var act = () => options.Validate();
        act.Should().NotThrow();
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    [InlineData(-512)]
    public void Validate_WithInvalidWidth_ThrowsException(int width)
    {
        // Arrange
        var options = new GenerationOptions { Width = width };

        // Act & Assert
        var act = () => options.Validate();
        act.Should().Throw<ArgumentException>().WithParameterName("Width");
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    [InlineData(-512)]
    public void Validate_WithInvalidHeight_ThrowsException(int height)
    {
        // Arrange
        var options = new GenerationOptions { Height = height };

        // Act & Assert
        var act = () => options.Validate();
        act.Should().Throw<ArgumentException>().WithParameterName("Height");
    }

    [Theory]
    [InlineData(511)]
    [InlineData(513)]
    [InlineData(100)]
    public void Validate_WithNonMultipleOf8Width_ThrowsException(int width)
    {
        // Arrange
        var options = new GenerationOptions { Width = width };

        // Act & Assert
        var act = () => options.Validate();
        act.Should().Throw<ArgumentException>().WithParameterName("Width");
    }

    [Theory]
    [InlineData(0)]
    [InlineData(51)]
    [InlineData(-1)]
    public void Validate_WithInvalidSteps_ThrowsException(int steps)
    {
        // Arrange
        var options = new GenerationOptions { Steps = steps };

        // Act & Assert
        var act = () => options.Validate();
        act.Should().Throw<ArgumentException>().WithParameterName("Steps");
    }

    [Theory]
    [InlineData(1)]
    [InlineData(4)]
    [InlineData(50)]
    public void Validate_WithValidSteps_Succeeds(int steps)
    {
        // Arrange
        var options = new GenerationOptions { Steps = steps };

        // Act & Assert
        var act = () => options.Validate();
        act.Should().NotThrow();
    }

    [Fact]
    public void Validate_WithNegativeGuidanceScale_ThrowsException()
    {
        // Arrange
        var options = new GenerationOptions { GuidanceScale = -1.0f };

        // Act & Assert
        var act = () => options.Validate();
        act.Should().Throw<ArgumentException>().WithParameterName("GuidanceScale");
    }

    [Fact]
    public void WithSeed_CreatesNewInstanceWithSeed()
    {
        // Arrange
        var original = new GenerationOptions
        {
            Width = 768,
            Height = 768,
            Steps = 4,
            GuidanceScale = 1.5f,
            NegativePrompt = "blurry"
        };

        // Act
        var withSeed = original.WithSeed(42);

        // Assert
        withSeed.Seed.Should().Be(42);
        withSeed.Width.Should().Be(768);
        withSeed.Height.Should().Be(768);
        withSeed.Steps.Should().Be(4);
        withSeed.GuidanceScale.Should().Be(1.5f);
        withSeed.NegativePrompt.Should().Be("blurry");
    }

    [Fact]
    public void WithSeed_DoesNotModifyOriginal()
    {
        // Arrange
        var original = new GenerationOptions { Seed = 100 };

        // Act
        _ = original.WithSeed(42);

        // Assert
        original.Seed.Should().Be(100);
    }
}
