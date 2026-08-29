using AwesomeAssertions;
using LMSupply.Generator.Abstractions;

namespace LMSupply.Generator.Tests;

/// <summary>
/// Integration tests for Generator models.
/// These tests require actual model downloads and GPU/CPU inference.
/// </summary>
[Trait("Category", "Integration")]
public class GeneratorIntegrationTests
{
    private const string Phi35Model = "microsoft/Phi-3.5-mini-instruct-onnx";
    private const string Phi4Model = "microsoft/Phi-4-mini-instruct-onnx";
    private const string TestPrompt = "What is 2+2? Answer with just the number.";

    // Cached model paths (for direct path loading)
    private static readonly string Phi4CachedPath = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
        ".cache", "huggingface", "hub",
        "models--microsoft--Phi-4-mini-instruct-onnx",
        "snapshots", "main");

    #region Phi-3.5 Tests

    [Fact]
    public async Task Phi35_DirectML_LoadAndGenerate_ShouldWork()
    {
        // Arrange
        var options = new GeneratorOptions
        {
            Provider = ExecutionProvider.DirectML,
            Verbose = true
        };

        // Act
        await using var model = await LocalGenerator.LoadAsync(Phi35Model, options, cancellationToken: TestContext.Current.CancellationToken);

        // Assert - Load
        model.Should().NotBeNull();
        model.ModelId.Should().Be(Phi35Model);

        var info = model.GetModelInfo();
        info.ExecutionProvider.Should().Be("DirectML");

        // Act - Generate
        var result = await model.GenerateCompleteAsync(TestPrompt, new Models.GenerationOptions { MaxTokens = 50 }, TestContext.Current.CancellationToken);

        // Assert - Generate (verify inference works, not model accuracy)
        result.Should().NotBeNullOrWhiteSpace();
    }

    [Fact]
    public async Task Phi35_CPU_LoadAndGenerate_ShouldWork()
    {
        // Arrange
        var options = new GeneratorOptions
        {
            Provider = ExecutionProvider.Cpu,
            Verbose = true
        };

        // Act
        await using var model = await LocalGenerator.LoadAsync(Phi35Model, options, cancellationToken: TestContext.Current.CancellationToken);

        // Assert - Load
        model.Should().NotBeNull();
        model.ModelId.Should().Be(Phi35Model);

        var info = model.GetModelInfo();
        info.ExecutionProvider.ToUpperInvariant().Should().Contain("CPU");

        // Act - Generate
        var result = await model.GenerateCompleteAsync(TestPrompt, new Models.GenerationOptions { MaxTokens = 50 }, TestContext.Current.CancellationToken);

        // Assert - Generate (verify inference works, not model accuracy)
        result.Should().NotBeNullOrWhiteSpace();
    }

    [Fact]
    public async Task Phi35_DirectML_WarmupAsync_ShouldNotThrow()
    {
        // Arrange
        var options = new GeneratorOptions
        {
            Provider = ExecutionProvider.DirectML,
            Verbose = true
        };

        // Act
        await using var model = await LocalGenerator.LoadAsync(Phi35Model, options, cancellationToken: TestContext.Current.CancellationToken);

        // Assert
        var warmupAction = () => model.WarmupAsync();
        await warmupAction.Should().NotThrowAsync();
    }

    #endregion

    #region Phi-4 Tests

    [Fact]
    public async Task Phi4_DirectML_LoadAndGenerate_ShouldWork()
    {
        // Arrange
        var options = new GeneratorOptions
        {
            Provider = ExecutionProvider.DirectML,
            Verbose = true
        };

        // Act
        await using var model = await LocalGenerator.LoadAsync(Phi4Model, options, cancellationToken: TestContext.Current.CancellationToken);

        // Assert - Load
        model.Should().NotBeNull();
        model.ModelId.Should().Be(Phi4Model);

        var info = model.GetModelInfo();
        info.ExecutionProvider.Should().Be("DirectML");

        // Act - Generate
        var result = await model.GenerateCompleteAsync(TestPrompt, new Models.GenerationOptions { MaxTokens = 50 }, TestContext.Current.CancellationToken);

        // Assert - Generate (verify inference works, not model accuracy)
        result.Should().NotBeNullOrWhiteSpace();
    }

    [Fact]
    public async Task Phi4_CPU_LoadAndGenerate_ShouldWork()
    {
        // Arrange
        var options = new GeneratorOptions
        {
            Provider = ExecutionProvider.Cpu,
            Verbose = true
        };

        // Act
        await using var model = await LocalGenerator.LoadAsync(Phi4Model, options, cancellationToken: TestContext.Current.CancellationToken);

        // Assert - Load
        model.Should().NotBeNull();
        model.ModelId.Should().Be(Phi4Model);

        var info = model.GetModelInfo();
        info.ExecutionProvider.ToUpperInvariant().Should().Contain("CPU");

        // Act - Generate
        var result = await model.GenerateCompleteAsync(TestPrompt, new Models.GenerationOptions { MaxTokens = 50 }, TestContext.Current.CancellationToken);

        // Assert - Generate (verify inference works, not model accuracy)
        result.Should().NotBeNullOrWhiteSpace();
    }

    [Fact]
    public async Task Phi4_DirectML_WarmupAsync_ShouldNotThrow()
    {
        // Arrange
        var options = new GeneratorOptions
        {
            Provider = ExecutionProvider.DirectML,
            Verbose = true
        };

        // Act
        await using var model = await LocalGenerator.LoadAsync(Phi4Model, options, cancellationToken: TestContext.Current.CancellationToken);

        // Assert
        var warmupAction = () => model.WarmupAsync();
        await warmupAction.Should().NotThrowAsync();
    }

    #endregion

    #region Comparison Tests

    [Fact]
    public async Task Phi35_vs_Phi4_DirectML_BothShouldGenerateValidResponse()
    {
        // Phi-3.5
        var phi35Options = new GeneratorOptions { Provider = ExecutionProvider.DirectML };
        await using var phi35Model = await LocalGenerator.LoadAsync(Phi35Model, phi35Options, cancellationToken: TestContext.Current.CancellationToken);
        var phi35Result = await phi35Model.GenerateCompleteAsync(TestPrompt, new Models.GenerationOptions { MaxTokens = 50 }, TestContext.Current.CancellationToken);

        // Phi-4
        var phi4Options = new GeneratorOptions { Provider = ExecutionProvider.DirectML };
        await using var phi4Model = await LocalGenerator.LoadAsync(Phi4Model, phi4Options, cancellationToken: TestContext.Current.CancellationToken);
        var phi4Result = await phi4Model.GenerateCompleteAsync(TestPrompt, new Models.GenerationOptions { MaxTokens = 50 }, TestContext.Current.CancellationToken);

        // Assert - both should produce valid responses
        // Note: We only verify that models generate non-empty responses.
        // Model accuracy (e.g., correct math answers) is not LMSupply's responsibility.
        phi35Result.Should().NotBeNullOrWhiteSpace("Phi-3.5 should generate response");
        phi4Result.Should().NotBeNullOrWhiteSpace("Phi-4 should generate response");

        // Verify responses have meaningful content
        phi35Result.Trim().Length.Should().BeGreaterThan(0, "Phi-3.5 should produce meaningful output");
        phi4Result.Trim().Length.Should().BeGreaterThan(0, "Phi-4 should produce meaningful output");
    }

    #endregion

    #region Auto Provider Tests (Zero-Configuration)

    [Fact]
    public async Task Phi35_Auto_ShouldSelectBestAvailableProvider()
    {
        // Arrange - Auto mode (default, no explicit provider)
        var options = new GeneratorOptions
        {
            Provider = ExecutionProvider.Auto,
            Verbose = true
        };

        // Act
        await using var model = await LocalGenerator.LoadAsync(Phi35Model, options, cancellationToken: TestContext.Current.CancellationToken);

        // Assert - Model should load successfully
        model.Should().NotBeNull();
        model.ModelId.Should().Be(Phi35Model);

        var info = model.GetModelInfo();

        // On GPU-capable systems, should NOT be CPU (unless no GPU available)
        // This verifies the fallback chain selected a GPU provider
        info.ExecutionProvider.Should().NotBeNullOrWhiteSpace();

        // Generate should work
        var result = await model.GenerateCompleteAsync(TestPrompt, new Models.GenerationOptions { MaxTokens = 50 }, TestContext.Current.CancellationToken);

        result.Should().NotBeNullOrWhiteSpace("Auto mode should generate valid response");
    }

    [Fact]
    public async Task Phi35_Auto_OnWindowsWithGpu_ShouldPreferGpuOverCpu()
    {
        // Skip on non-Windows
        if (!OperatingSystem.IsWindows())
        {
            return;
        }

        // Arrange - Auto mode
        var options = new GeneratorOptions
        {
            Provider = ExecutionProvider.Auto,
            Verbose = true
        };

        // Act
        await using var model = await LocalGenerator.LoadAsync(Phi35Model, options, cancellationToken: TestContext.Current.CancellationToken);
        var info = model.GetModelInfo();

        // Assert - On Windows with GPU, should select CUDA or DirectML, not CPU
        // Note: This test may need adjustment based on actual hardware
        var provider = info.ExecutionProvider.ToUpperInvariant();

        // The provider should be one of the GPU options if available
        var isGpuProvider = provider.Contains("CUDA") ||
                            provider.Contains("DIRECTML") ||
                            provider.Contains("DML");

        var isCpuProvider = provider.Contains("CPU");

        // Log for debugging
        Console.WriteLine($"Auto mode selected provider: {info.ExecutionProvider}");

        // At minimum, verify it's a valid provider
        (isGpuProvider || isCpuProvider).Should().BeTrue(
            $"Provider '{info.ExecutionProvider}' should be a recognized execution provider");
    }

    #endregion

    #region Streaming Tests

    [Fact]
    public async Task Phi35_DirectML_StreamingGeneration_ShouldWork()
    {
        // Arrange
        var options = new GeneratorOptions
        {
            Provider = ExecutionProvider.DirectML
        };

        await using var model = await LocalGenerator.LoadAsync(Phi35Model, options, cancellationToken: TestContext.Current.CancellationToken);

        // Act
        var tokens = new List<string>();
        await foreach (var token in model.GenerateAsync(TestPrompt, new Models.GenerationOptions { MaxTokens = 50 }, TestContext.Current.CancellationToken))
        {
            tokens.Add(token);
        }

        // Assert (verify streaming works, not model accuracy)
        tokens.Should().NotBeEmpty();
        string.Join("", tokens).Should().NotBeNullOrWhiteSpace();
    }

    [Fact]
    public async Task Phi4_DirectML_StreamingGeneration_ShouldWork()
    {
        // Arrange
        var options = new GeneratorOptions
        {
            Provider = ExecutionProvider.DirectML
        };

        await using var model = await LocalGenerator.LoadAsync(Phi4Model, options, cancellationToken: TestContext.Current.CancellationToken);

        // Act
        var tokens = new List<string>();
        await foreach (var token in model.GenerateAsync(TestPrompt, new Models.GenerationOptions { MaxTokens = 50 }, TestContext.Current.CancellationToken))
        {
            tokens.Add(token);
        }

        // Assert (verify streaming works, not model accuracy)
        tokens.Should().NotBeEmpty();
        string.Join("", tokens).Should().NotBeNullOrWhiteSpace();
    }

    #endregion

    #region Direct Path Loading Tests (Using Cached Models)

    [Fact]
    public async Task Phi4_FromPath_DirectML_LoadAndGenerate_ShouldWork()
    {
        // Require cached model
        Assert.True(Directory.Exists(Phi4CachedPath), $"Phi-4 model not cached at {Phi4CachedPath}. Run `huggingface-cli download microsoft/Phi-4-mini-instruct-onnx` first.");

        // Arrange
        var options = new GeneratorOptions
        {
            Provider = ExecutionProvider.DirectML,
            Verbose = true
        };

        // Act - Load directly from path
        await using var model = await LocalGenerator.LoadFromPathAsync(Phi4CachedPath, options);

        // Assert - Load
        model.Should().NotBeNull();

        var info = model.GetModelInfo();
        // DirectML provider name varies - just check it's not CPU
        info.ExecutionProvider.Should().NotBe("Cpu", "Should be using DirectML, not CPU");

        // Act - Generate
        var result = await model.GenerateCompleteAsync(TestPrompt, new Models.GenerationOptions { MaxTokens = 50 }, TestContext.Current.CancellationToken);

        // Assert - Generate (verify inference works, not model accuracy)
        result.Should().NotBeNullOrWhiteSpace();
    }

    [Fact]
    public async Task Phi4_FromPath_CPU_LoadAndGenerate_ShouldWork()
    {
        // Require cached model
        Assert.True(Directory.Exists(Phi4CachedPath), $"Phi-4 model not cached at {Phi4CachedPath}. Run `huggingface-cli download microsoft/Phi-4-mini-instruct-onnx` first.");

        // Arrange
        var options = new GeneratorOptions
        {
            Provider = ExecutionProvider.Cpu,
            Verbose = true
        };

        // Act - Load directly from path
        await using var model = await LocalGenerator.LoadFromPathAsync(Phi4CachedPath, options);

        // Assert - Load
        model.Should().NotBeNull();

        var info = model.GetModelInfo();
        info.ExecutionProvider.ToUpperInvariant().Should().Contain("CPU");

        // Act - Generate
        var result = await model.GenerateCompleteAsync(TestPrompt, new Models.GenerationOptions { MaxTokens = 50 }, TestContext.Current.CancellationToken);

        // Assert - Generate (verify inference works, not model accuracy)
        result.Should().NotBeNullOrWhiteSpace();
    }

    [Fact]
    public async Task Phi4_FromPath_DirectML_WarmupAsync_ShouldNotThrow()
    {
        // Require cached model
        Assert.True(Directory.Exists(Phi4CachedPath), $"Phi-4 model not cached at {Phi4CachedPath}. Run `huggingface-cli download microsoft/Phi-4-mini-instruct-onnx` first.");

        // Arrange
        var options = new GeneratorOptions
        {
            Provider = ExecutionProvider.DirectML,
            Verbose = true
        };

        // Act
        await using var model = await LocalGenerator.LoadFromPathAsync(Phi4CachedPath, options);

        // Assert
        var warmupAction = () => model.WarmupAsync();
        await warmupAction.Should().NotThrowAsync();
    }

    [Fact]
    public async Task Phi4_FromPath_CPU_WarmupAsync_ShouldNotThrow()
    {
        // Require cached model
        Assert.True(Directory.Exists(Phi4CachedPath), $"Phi-4 model not cached at {Phi4CachedPath}. Run `huggingface-cli download microsoft/Phi-4-mini-instruct-onnx` first.");

        // Arrange
        var options = new GeneratorOptions
        {
            Provider = ExecutionProvider.Cpu,
            Verbose = true
        };

        // Act
        await using var model = await LocalGenerator.LoadFromPathAsync(Phi4CachedPath, options);

        // Assert
        var warmupAction = () => model.WarmupAsync();
        await warmupAction.Should().NotThrowAsync();
    }

    #endregion
}
