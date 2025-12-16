using FluentAssertions;

namespace LMSupply.Generator.Tests;

public class OnnxGeneratorModelFactoryTests
{
    [Fact]
    public void Constructor_Default_CreatesFactory()
    {
        // Act
        var factory = new OnnxGeneratorModelFactory();

        // Assert
        factory.Should().NotBeNull();
    }

    [Fact]
    public void Constructor_WithParameters_CreatesFactory()
    {
        // Arrange
        var cacheDir = Path.GetTempPath();

        // Act
        var factory = new OnnxGeneratorModelFactory(cacheDir, ExecutionProvider.Cpu);

        // Assert
        factory.Should().NotBeNull();
    }

    [Fact]
    public void IsModelAvailable_NonExistentModel_ReturnsFalse()
    {
        // Arrange
        var factory = new OnnxGeneratorModelFactory();

        // Act
        var isAvailable = factory.IsModelAvailable("nonexistent/model");

        // Assert
        isAvailable.Should().BeFalse();
    }

    [Fact]
    public void GetModelCachePath_ReturnsExpectedFormat()
    {
        // Arrange
        var cacheDir = "/test/cache";
        var factory = new OnnxGeneratorModelFactory(cacheDir, ExecutionProvider.Auto);

        // Act
        var path = factory.GetModelCachePath("microsoft/Phi-3.5-mini-instruct-onnx");

        // Assert
        path.Should().Contain("models--microsoft--Phi-3.5-mini-instruct-onnx");
    }

    [Fact]
    public void GetAvailableModels_EmptyCache_ReturnsEmpty()
    {
        // Arrange
        var tempDir = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString());
        var factory = new OnnxGeneratorModelFactory(tempDir, ExecutionProvider.Auto);

        // Act
        var models = factory.GetAvailableModels();

        // Assert
        models.Should().BeEmpty();
    }

    [Fact]
    public async Task LoadAsync_NonExistentModel_ThrowsHttpException()
    {
        // Arrange
        var tempDir = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString());
        using var factory = new OnnxGeneratorModelFactory(tempDir, ExecutionProvider.Auto);

        // Act & Assert - attempts download which fails with HTTP error
        var action = () => factory.LoadAsync("nonexistent/model");
        await action.Should().ThrowAsync<HttpRequestException>();
    }

    [Fact]
    public async Task DownloadModelAsync_InvalidModel_ThrowsHttpException()
    {
        // Arrange
        var tempDir = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString());
        using var factory = new OnnxGeneratorModelFactory(tempDir, ExecutionProvider.Auto);

        // Act & Assert - attempts download which fails with HTTP error
        var action = () => factory.DownloadModelAsync("test/model");
        await action.Should().ThrowAsync<HttpRequestException>();
    }

    [Fact]
    public async Task DownloadModelAsync_AlreadyAvailable_SkipsDownload()
    {
        // Arrange
        var tempDir = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString());
        var modelDir = Path.Combine(tempDir, "models--test--model");
        Directory.CreateDirectory(modelDir);
        try
        {
            // Create a valid model structure
            File.WriteAllText(Path.Combine(modelDir, "genai_config.json"), "{}");

            using var factory = new OnnxGeneratorModelFactory(tempDir, ExecutionProvider.Auto);

            // Act - should not throw, as model is available
            await factory.DownloadModelAsync("test/model");

            // Assert - if we get here without exception, model was found
            factory.IsModelAvailable("test/model").Should().BeTrue();
        }
        finally
        {
            Directory.Delete(tempDir, recursive: true);
        }
    }

    [Fact]
    public void Dispose_CanBeCalledMultipleTimes()
    {
        // Arrange
        var factory = new OnnxGeneratorModelFactory();

        // Act & Assert - should not throw
        factory.Dispose();
        factory.Dispose();
    }
}
