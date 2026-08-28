using System.Text;
using AwesomeAssertions;
using LMSupply.Generator.Internal;
using LMSupply.Generator.Internal.Llama;
using LMSupply.Generator.Models;
using Xunit;

namespace LMSupply.Generator.Tests.Gguf;

/// <summary>
/// Integration tests for GGUF model support.
/// These tests require actual model downloads and inference,
/// so they are marked as Integration tests and skipped in CI.
/// Run with: dotnet test --filter "Category=Integration"
/// </summary>
[Trait("Category", "Integration")]
public class GgufIntegrationTests
{
    /// <summary>
    /// Tests that a GGUF model can be loaded from a registry alias.
    /// </summary>
    [Fact]
    public async Task LoadAsync_WithRegistryAlias_LoadsModel()
    {
        // Arrange
        var options = new GeneratorOptions
        {
            MaxContextLength = 2048 // Smaller context for faster testing
        };

        // Act
        await using var model = await LocalGenerator.LoadAsync(
            "gguf:qwen3-fast", // Uses the smallest/fastest Qwen3 model
            options);

        // Assert
        model.Should().NotBeNull();
        model.ModelId.Should().NotBeNullOrEmpty();
        model.MaxContextLength.Should().BeGreaterThan(0);
    }

    /// <summary>
    /// Tests text generation with a GGUF model.
    /// </summary>
    [Fact]
    public async Task GenerateAsync_WithGgufModel_GeneratesText()
    {
        // Arrange
        await using var model = await LocalGenerator.LoadAsync("gguf:qwen3-fast");

        // Act
        var result = new StringBuilder();
        await foreach (var token in model.GenerateAsync(
            "Hello, my name is",
            new GenerationOptions { MaxTokens = 20 }))
        {
            result.Append(token);
        }

        // Assert
        result.ToString().Should().NotBeNullOrEmpty();
        result.Length.Should().BeGreaterThan(0);
    }

    /// <summary>
    /// Tests chat generation with a GGUF model.
    /// </summary>
    [Fact]
    public async Task GenerateChatAsync_WithGgufModel_GeneratesResponse()
    {
        // Arrange
        await using var model = await LocalGenerator.LoadAsync("gguf:qwen3-fast");

        var messages = new[]
        {
            ChatMessage.System("You are a helpful assistant."),
            ChatMessage.User("What is 2+2?")
        };

        // Act
        var result = new StringBuilder();
        await foreach (var token in model.GenerateChatAsync(
            messages,
            new GenerationOptions { MaxTokens = 512 }))
        {
            result.Append(token);
        }

        // Assert
        result.ToString().Should().NotBeNullOrEmpty();
    }

    /// <summary>
    /// Tests that GgufModelDownloader can list files from a repository.
    /// </summary>
    [Fact]
    public async Task GgufModelDownloader_ListGgufFilesAsync_ReturnsFiles()
    {
        // Arrange
        using var downloader = new GgufModelDownloader();

        // Act
        var files = await downloader.ListGgufFilesAsync(
            "bartowski/Llama-3.2-1B-Instruct-GGUF");

        // Assert
        files.Should().NotBeEmpty();
        files.Should().AllSatisfy(f =>
        {
            f.FileName.Should().EndWith(".gguf");
            f.SizeBytes.Should().BeGreaterThan(0);
        });
    }

    /// <summary>
    /// Tests that warmup works correctly for GGUF models.
    /// </summary>
    [Fact]
    public async Task WarmupAsync_WithGgufModel_CompletesSuccessfully()
    {
        // Arrange
        await using var model = await LocalGenerator.LoadAsync("gguf:qwen3-fast");

        // Act
        var warmupTask = model.WarmupAsync();

        // Assert
        await warmupTask.Invoking(t => t).Should().NotThrowAsync();
    }

    /// <summary>
    /// Tests that model info is correctly populated for GGUF models.
    /// </summary>
    [Fact]
    public async Task GetModelInfo_WithGgufModel_ReturnsCorrectInfo()
    {
        // Arrange
        await using var model = await LocalGenerator.LoadAsync("gguf:qwen3-fast");

        // Act
        var info = model.GetModelInfo();

        // Assert
        info.ModelId.Should().NotBeNullOrEmpty();
        info.ModelPath.Should().EndWith(".gguf");
        info.MaxContextLength.Should().BeGreaterThan(0);
        info.ChatFormat.Should().NotBeNullOrEmpty();
        info.ExecutionProvider.Should().StartWith("llama-server-");
    }
}

/// <summary>
/// Unit tests for GGUF model format detection.
/// These tests don't require model downloads.
/// </summary>
public class GgufModelFormatTests
{
    /// <summary>
    /// Tests model format detection for GGUF files.
    /// </summary>
    [Theory]
    [InlineData("gguf:qwen3-default", true)]
    [InlineData("gguf:qwen3-fast", true)]
    [InlineData("gguf:quality", true)]
    [InlineData("gguf:korean", true)]
    [InlineData("bartowski/Llama-3.2-3B-Instruct-GGUF", true)]
    [InlineData("microsoft/Phi-4-mini-instruct-onnx", false)]
    public void ModelFormatDetector_DetectsGgufFormat(string modelId, bool expectedGguf)
    {
        // Act
        var format = ModelFormatDetector.Detect(modelId);

        // Assert
        if (expectedGguf)
        {
            format.Should().Be(ModelFormat.Gguf);
        }
        else
        {
            format.Should().NotBe(ModelFormat.Gguf);
        }
    }

    [Theory]
    [InlineData("/path/to/model.gguf", true)]
    [InlineData("/path/to/model.onnx", false)]
    [InlineData("C:\\models\\test.gguf", true)]
    public void ModelFormatDetector_DetectsFromFilePath(string path, bool expectedGguf)
    {
        // Act
        var format = ModelFormatDetector.Detect(path);

        // Assert
        if (expectedGguf)
        {
            format.Should().Be(ModelFormat.Gguf);
        }
        else
        {
            format.Should().NotBe(ModelFormat.Gguf);
        }
    }
}
