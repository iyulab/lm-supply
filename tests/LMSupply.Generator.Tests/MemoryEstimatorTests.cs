using FluentAssertions;

namespace LMSupply.Generator.Tests;

public class MemoryEstimatorTests
{
    [Fact]
    public void CalculateModelMemory_INT4_ReturnsHalfBytePerParam()
    {
        // Arrange
        var paramCount = 3_800_000_000L; // Phi-3.5-mini

        // Act
        var memoryBytes = MemoryEstimator.CalculateModelMemory(paramCount, Quantization.INT4);

        // Assert
        memoryBytes.Should().Be(1_900_000_000L); // 3.8B * 0.5 bytes
    }

    [Fact]
    public void CalculateModelMemory_FP16_ReturnsTwoBytesPerParam()
    {
        // Arrange
        var paramCount = 1_000_000_000L;

        // Act
        var memoryBytes = MemoryEstimator.CalculateModelMemory(paramCount, Quantization.FP16);

        // Assert
        memoryBytes.Should().Be(2_000_000_000L); // 1B * 2 bytes
    }

    [Fact]
    public void CalculateKvCacheMemory_ReturnsExpectedSize()
    {
        // Arrange
        int batchSize = 1;
        int contextLength = 4096;
        int numLayers = 32;
        int hiddenSize = 3072;

        // Act
        var kvCacheBytes = MemoryEstimator.CalculateKvCacheMemory(
            batchSize, contextLength, numLayers, hiddenSize, KvCachePrecision.FP16);

        // Assert
        // Formula: 1 * 4096 * 2 * 32 * 3072 * 2 = 1,610,612,736 bytes
        var expected = 1L * 4096 * 2 * 32 * 3072 * 2;
        kvCacheBytes.Should().Be(expected);
    }

    [Fact]
    public void Calculate_ReturnsCompleteEstimate()
    {
        // Arrange
        var config = new ModelMemoryConfig
        {
            ParameterCount = 3_800_000_000,
            NumLayers = 32,
            HiddenSize = 3072,
            ContextLength = 4096,
            Quantization = Quantization.INT4
        };

        // Act
        var estimate = MemoryEstimator.Calculate(config);

        // Assert
        estimate.ModelMemoryBytes.Should().BeGreaterThan(0);
        estimate.KvCacheMemoryBytes.Should().BeGreaterThan(0);
        estimate.OverheadBytes.Should().BeGreaterThan(0);
        estimate.TotalBytes.Should().Be(
            estimate.ModelMemoryBytes + estimate.KvCacheMemoryBytes + estimate.OverheadBytes);
    }

    [Fact]
    public void CanFitInMemory_WithSufficientMemory_ReturnsTrue()
    {
        // Arrange
        var config = new ModelMemoryConfig
        {
            ParameterCount = 1_000_000_000,
            NumLayers = 16,
            HiddenSize = 2048,
            ContextLength = 2048,
            Quantization = Quantization.INT4
        };

        var availableMemory = 8L * 1024 * 1024 * 1024; // 8GB

        // Act
        var canFit = MemoryEstimator.CanFitInMemory(config, availableMemory);

        // Assert
        canFit.Should().BeTrue();
    }

    [Fact]
    public void CanFitInMemory_WithInsufficientMemory_ReturnsFalse()
    {
        // Arrange
        var config = new ModelMemoryConfig
        {
            ParameterCount = 14_000_000_000,
            NumLayers = 40,
            HiddenSize = 5120,
            ContextLength = 8192,
            Quantization = Quantization.FP16 // Large model in FP16
        };

        var availableMemory = 4L * 1024 * 1024 * 1024; // 4GB

        // Act
        var canFit = MemoryEstimator.CanFitInMemory(config, availableMemory);

        // Assert
        canFit.Should().BeFalse();
    }

    [Theory]
    [InlineData("phi-3.5-mini")]
    [InlineData("phi-4")]
    [InlineData("llama-3.2-1b")]
    [InlineData("llama-3.2-3b")]
    public void GetDefaultConfig_ReturnsValidConfig(string modelFamily)
    {
        // Act
        var config = MemoryEstimator.GetDefaultConfig(modelFamily);

        // Assert
        config.ParameterCount.Should().BeGreaterThan(0);
        config.NumLayers.Should().BeGreaterThan(0);
        config.HiddenSize.Should().BeGreaterThan(0);
        config.ContextLength.Should().BeGreaterThan(0);
    }

    [Fact]
    public void MemoryEstimate_GetSummary_ReturnsFormattedString()
    {
        // Arrange
        var config = MemoryEstimator.GetDefaultConfig("phi-3.5-mini");

        // Act
        var estimate = MemoryEstimator.Calculate(config);
        var summary = estimate.GetSummary();

        // Assert
        summary.Should().Contain("Model Weights:");
        summary.Should().Contain("KV Cache:");
        summary.Should().Contain("Total:");
        summary.Should().Contain("INT4");
    }

    [Fact]
    public void Phi35Mini_FitsIn5GB_WithINT4()
    {
        // Arrange - Phi-3.5-mini with INT4 requires ~3.6GB + 20% safety margin = ~4.3GB
        var config = MemoryEstimator.GetDefaultConfig("phi-3.5-mini");
        var availableMemory = 5L * 1024 * 1024 * 1024; // 5GB to accommodate safety margin

        // Act
        var estimate = MemoryEstimator.Calculate(config);
        var canFit = MemoryEstimator.CanFitInMemory(config, availableMemory);

        // Assert
        canFit.Should().BeTrue($"Phi-3.5-mini INT4 requires {estimate.TotalBytes / (1024.0 * 1024 * 1024):F2}GB + 20% margin");
    }
}
