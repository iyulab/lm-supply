using AwesomeAssertions;
using LMSupply.Hardware;

namespace LMSupply.Core.Tests.Hardware;

public class VramFitResultTests
{
    private const long GB = 1024L * 1024 * 1024;

    [Fact]
    public void Evaluate_ModelFitsComfortably_FullGpu()
    {
        // Arrange: 10GB available, 4GB model, 32 layers
        var result = VramFitResult.Evaluate(
            availableVramBytes: 10 * GB,
            modelSizeBytes: 4 * GB,
            totalLayers: 32);

        // Assert: model uses 40% of VRAM (< 70%), so full GPU
        result.Fits.Should().BeTrue();
        result.GpuLayerCount.Should().Be(-1);
        result.OffloadRatio.Should().Be(1.0f);
    }

    [Fact]
    public void Evaluate_ModelPartialFit_PartialOffload()
    {
        // Arrange: 4GB available, 8GB model, 32 layers
        var result = VramFitResult.Evaluate(
            availableVramBytes: 4 * GB,
            modelSizeBytes: 8 * GB,
            totalLayers: 32);

        // Assert: model exceeds VRAM, partial offload
        result.Fits.Should().BeTrue();
        result.GpuLayerCount.Should().BeInRange(1, 31);
        result.OffloadRatio.Should().BeInRange(0.01f, 0.99f);
    }

    [Fact]
    public void Evaluate_NoGpu_CpuOnly()
    {
        // Arrange: 0 VRAM available
        var result = VramFitResult.Evaluate(
            availableVramBytes: 0,
            modelSizeBytes: 4 * GB,
            totalLayers: 32);

        // Assert: CPU-only mode
        result.Fits.Should().BeTrue();
        result.GpuLayerCount.Should().Be(0);
        result.OffloadRatio.Should().Be(0f);
    }

    [Fact]
    public void RecommendBatchSize_ScalesWithAvailableVram()
    {
        // Arrange: high VRAM (16GB available, 4GB model) vs low VRAM (4GB available, 3GB model)
        var highVram = VramFitResult.Evaluate(
            availableVramBytes: 16 * GB,
            modelSizeBytes: 4 * GB,
            totalLayers: 32);

        var lowVram = VramFitResult.Evaluate(
            availableVramBytes: 4 * GB,
            modelSizeBytes: 3 * GB,
            totalLayers: 32);

        // Assert: more remaining VRAM → larger batch size
        highVram.RecommendedBatchSize.Should().BeGreaterThan(lowVram.RecommendedBatchSize);
    }

    [Fact]
    public void RecommendKvQuantType_HighVram_UsesQ8()
    {
        // Arrange: 16GB available, 4GB model → plenty of remaining VRAM
        var result = VramFitResult.Evaluate(
            availableVramBytes: 16 * GB,
            modelSizeBytes: 4 * GB,
            totalLayers: 32);

        // Assert: high remaining VRAM → Q8_0 for better quality
        result.RecommendedKvQuantType.Should().Be(KvCacheQuantizationType.Q8_0);
    }

    [Fact]
    public void RecommendKvQuantType_LowVram_UsesQ4()
    {
        // Arrange: 4GB available, 3.5GB model → tight VRAM
        var result = VramFitResult.Evaluate(
            availableVramBytes: 4 * GB,
            modelSizeBytes: (long)(3.5 * GB),
            totalLayers: 32);

        // Assert: low remaining VRAM → Q4_0 for memory savings
        result.RecommendedKvQuantType.Should().Be(KvCacheQuantizationType.Q4_0);
    }
}
