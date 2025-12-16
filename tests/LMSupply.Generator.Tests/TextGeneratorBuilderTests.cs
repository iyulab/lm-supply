using FluentAssertions;

namespace LMSupply.Generator.Tests;

public class TextGeneratorBuilderTests
{
    [Fact]
    public void Create_ReturnsNewBuilder()
    {
        // Act
        var builder = TextGeneratorBuilder.Create();

        // Assert
        builder.Should().NotBeNull();
    }

    [Fact]
    public void WithModelPath_SetsModelPath()
    {
        // Arrange
        var builder = TextGeneratorBuilder.Create();

        // Act & Assert - should not throw
        builder.WithModelPath("C:/models/phi3");
    }

    [Fact]
    public void WithHuggingFaceModel_SetsModelId()
    {
        // Arrange
        var builder = TextGeneratorBuilder.Create();

        // Act & Assert - should not throw
        builder.WithHuggingFaceModel("microsoft/Phi-3.5-mini-instruct-onnx");
    }

    [Fact]
    public void WithDefaultModel_UsesPhi35Mini()
    {
        // Arrange
        var builder = TextGeneratorBuilder.Create();

        // Act & Assert - should not throw
        builder.WithDefaultModel();
    }

    [Theory]
    [InlineData(GeneratorModelPreset.Default)]
    [InlineData(GeneratorModelPreset.Fast)]
    [InlineData(GeneratorModelPreset.Quality)]
    [InlineData(GeneratorModelPreset.Small)]
    public void WithModel_AcceptsAllPresets(GeneratorModelPreset preset)
    {
        // Arrange
        var builder = TextGeneratorBuilder.Create();

        // Act & Assert - should not throw
        builder.WithModel(preset);
    }

    [Fact]
    public void FluentChaining_Works()
    {
        // Act
        var builder = TextGeneratorBuilder.Create()
            .WithDefaultModel()
            .WithProvider(ExecutionProvider.Cpu)
            .WithMaxContextLength(4096)
            .WithConcurrency(2)
            .WithVerboseLogging()
            .ForCreativeGeneration();

        // Assert
        builder.Should().NotBeNull();
    }

    [Fact]
    public async Task BuildAsync_WithoutModel_ThrowsInvalidOperationException()
    {
        // Arrange
        var builder = TextGeneratorBuilder.Create();

        // Act & Assert
        var action = () => builder.BuildAsync();
        await action.Should().ThrowAsync<InvalidOperationException>()
            .WithMessage("*Model path or ID is required*");
    }

    [Fact]
    public void WithModelPath_NullPath_ThrowsArgumentNullException()
    {
        // Arrange
        var builder = TextGeneratorBuilder.Create();

        // Act & Assert
        var action = () => builder.WithModelPath(null!);
        action.Should().Throw<ArgumentNullException>();
    }

    [Fact]
    public void WithHuggingFaceModel_NullId_ThrowsArgumentNullException()
    {
        // Arrange
        var builder = TextGeneratorBuilder.Create();

        // Act & Assert
        var action = () => builder.WithHuggingFaceModel(null!);
        action.Should().Throw<ArgumentNullException>();
    }
}
