using AwesomeAssertions;
using LMSupply.Generator;
using LMSupply.Runtime;

namespace LMSupply.Generator.Tests;

public class LlamaOptionsAutoTuneTests
{
    [Fact]
    public void GetOptimalForHardware_SmallModel_FullGpu()
    {
        var gpu = new GpuInfo
        {
            Vendor = GpuVendor.Nvidia,
            TotalMemoryBytes = 16L * 1024 * 1024 * 1024,
            FreeMemoryBytes = 14L * 1024 * 1024 * 1024
        };

        var options = LlamaOptions.GetOptimalForHardware(gpu, modelSizeBytes: 4L * 1024 * 1024 * 1024);
        options.GpuLayerCount.Should().Be(-1);
        options.BatchSize.Should().BeGreaterThanOrEqualTo(2048);
    }

    [Fact]
    public void GetOptimalForHardware_LargeModel_PartialOffload()
    {
        var gpu = new GpuInfo
        {
            Vendor = GpuVendor.Nvidia,
            TotalMemoryBytes = 8L * 1024 * 1024 * 1024,
            FreeMemoryBytes = 6L * 1024 * 1024 * 1024
        };

        var options = LlamaOptions.GetOptimalForHardware(gpu, modelSizeBytes: 19L * 1024 * 1024 * 1024, totalLayers: 64);
        options.GpuLayerCount.Should().BeInRange(1, 63);
    }

    [Fact]
    public void GetOptimalForHardware_CpuOnly()
    {
        var gpu = new GpuInfo { Vendor = GpuVendor.Unknown };
        var options = LlamaOptions.GetOptimalForHardware(gpu, modelSizeBytes: 4L * 1024 * 1024 * 1024);
        options.GpuLayerCount.Should().Be(0);
        options.BatchSize.Should().Be(512);
    }

    [Fact]
    public void GetOptimalForHardware_OriginalOverload_StillWorks()
    {
        var options = LlamaOptions.GetOptimalForHardware();
        options.Should().NotBeNull();
        options.UseMemoryMap.Should().BeTrue();
    }
}
