using AwesomeAssertions;
using LMSupply.Hardware;
using LMSupply.Runtime;
using LMSupply.Generator.Internal.Llama;
using Xunit;

namespace LMSupply.Generator.Tests;

[Collection("VramBudget")]
public class VramAwareSelectionIntegrationTests
{
    [Fact]
    public void FullPipeline_SimulatedGpu_SelectsModelAndTunesOptions()
    {
        // Simulate a mid-range GPU
        var gpu = new GpuInfo
        {
            Vendor = GpuVendor.Nvidia,
            DeviceName = "Test GPU",
            TotalMemoryBytes = 8L * 1024 * 1024 * 1024,
            FreeMemoryBytes = 6L * 1024 * 1024 * 1024,
            CudaComputeCapabilityMajor = 8,
            CudaComputeCapabilityMinor = 6
        };

        // Step 1: Calculate available VRAM.
        // Budget is based on total (not free) — host owns the GPU for its lifetime.
        var availableVram = VramBudget.GetAvailableBytes(gpu);
        availableVram.Should().BeGreaterThan(0);
        availableVram.Should().BeLessThan(8L * 1024 * 1024 * 1024); // 8GB * 0.85 = 6.8GB

        // Step 2: Select model via GGUF registry
        var model = GgufModelRegistry.GetAutoModel(gpu);
        model.Should().NotBeNull();
        model.EstimatedSizeBytes.Should().HaveValue();
        model.EstimatedSizeBytes!.Value.Should().BeLessThanOrEqualTo(availableVram);

        // Step 3: Calculate optimal options
        var options = LlamaOptions.GetOptimalForHardware(
            gpu, model.EstimatedSizeBytes.Value, totalLayers: 32);
        options.Should().NotBeNull();
        options.BatchSize.Should().BeGreaterThan(0);
        options.UseMemoryMap.Should().BeTrue();

        // Step 4: Verify VRAM fit result coherence
        var fitResult = VramFitResult.Evaluate(availableVram, model.EstimatedSizeBytes.Value, 32);
        fitResult.Fits.Should().BeTrue();
        fitResult.GpuLayerCount.Should().Be(options.GpuLayerCount);
    }

    [Fact]
    public void FullPipeline_CpuOnly_FallsBackGracefully()
    {
        var gpu = new GpuInfo { Vendor = GpuVendor.Unknown };

        var availableVram = VramBudget.GetAvailableBytes(gpu);
        availableVram.Should().Be(0);

        var model = GgufModelRegistry.GetAutoModel(gpu);
        model.Should().NotBeNull();
        // CPU-only → 0 VRAM → nothing fits → fallback to smallest in auto pool (qwen3-fast: ~2.25GB total)
        model.AliasName.Should().Be("gguf:qwen3-fast");

        var options = LlamaOptions.GetOptimalForHardware(
            gpu, model.EstimatedSizeBytes ?? 2_000_000_000L);
        options.GpuLayerCount.Should().Be(0);
        options.FlashAttention.Should().Be(false);
    }

    [Fact]
    public void FullPipeline_LargeGpu_SelectsLargestModel()
    {
        var gpu = new GpuInfo
        {
            Vendor = GpuVendor.Nvidia,
            DeviceName = "Test RTX 4090",
            TotalMemoryBytes = 24L * 1024 * 1024 * 1024,
            FreeMemoryBytes = 22L * 1024 * 1024 * 1024
        };

        var model = GgufModelRegistry.GetAutoModel(gpu);
        model.Should().NotBeNull();
        // With ~18.7GB available, should select gguf:large (19GB) or gguf:quality (7GB)
        model.ParameterCount.Should().BeGreaterThanOrEqualTo(12_000_000_000);
    }

    [Fact]
    public void ModelMemoryEstimator_ConsistentWithRegisteredSizes()
    {
        // Verify that the estimator produces reasonable results compared to known sizes
        var models = GgufModelRegistry.GetAllModels();
        foreach (var model in models)
        {
            if (model.EstimatedSizeBytes is > 0 && model.ParameterCount > 0)
            {
                var estimated = ModelMemoryEstimator.EstimateModelSizeBytes(
                    model.ParameterCount, model.QuantizationType);

                // Estimated should be within 3x of known size (rough sanity check)
                estimated.Should().BeGreaterThan(model.EstimatedSizeBytes.Value / 3,
                    because: $"estimate for {model.DisplayName} should be in range");
                estimated.Should().BeLessThan(model.EstimatedSizeBytes.Value * 3,
                    because: $"estimate for {model.DisplayName} should be in range");
            }
        }
    }
}
