using AwesomeAssertions;
using LMSupply;
using LMSupply.Generator.Internal;
using LMSupply.Runtime;

namespace LMSupply.Generator.Tests.Internal;

/// <summary>
/// Tests for GeneratorRoutingPolicy.ShouldUseOnnx — the single decision shared by LocalGenerator and
/// ModelFormatDetector for routing the auto path to ONNX (DirectML) vs GGUF (llama.cpp).
/// ONNX/DirectML is for a *discrete* non-NVIDIA GPU on Windows; integrated GPUs route to GGUF
/// (their VRAM reading is unreliable, so DirectML is not a real win — consistent with D1/D2 CPU demotion).
/// </summary>
public class GeneratorRoutingPolicyTests
{
    private const long GB = 1024L * 1024 * 1024;
    private const long MB = 1024L * 1024;

    [Fact]
    public void IntegratedIntel_DirectML_UsesGguf()
    {
        var gpu = new GpuInfo
        {
            Vendor = GpuVendor.Intel,
            DeviceName = "Intel(R) Iris(R) Xe Graphics",
            TotalMemoryBytes = 128 * MB,
            SharedMemoryBytes = 16 * GB,
        };
        gpu.IsIntegrated.Should().BeTrue();
        GeneratorRoutingPolicy.ShouldUseOnnx(gpu, ExecutionProvider.DirectML).Should().BeFalse();
    }

    [Fact]
    public void DiscreteIntelArc_DirectML_UsesOnnx()
    {
        var gpu = new GpuInfo
        {
            Vendor = GpuVendor.Intel,
            DeviceName = "Intel(R) Arc(TM) A770",
            TotalMemoryBytes = 12 * GB,
            SharedMemoryBytes = 16 * GB,
        };
        gpu.IsIntegrated.Should().BeFalse();
        GeneratorRoutingPolicy.ShouldUseOnnx(gpu, ExecutionProvider.DirectML).Should().BeTrue();
    }

    [Fact]
    public void DiscreteAmd_DirectML_UsesOnnx()
    {
        var gpu = new GpuInfo { Vendor = GpuVendor.Amd, TotalMemoryBytes = 16 * GB, SharedMemoryBytes = 16 * GB };
        GeneratorRoutingPolicy.ShouldUseOnnx(gpu, ExecutionProvider.DirectML).Should().BeTrue();
    }

    [Fact]
    public void Nvidia_UsesGguf()
    {
        var gpu = new GpuInfo { Vendor = GpuVendor.Nvidia, TotalMemoryBytes = 24 * GB };
        GeneratorRoutingPolicy.ShouldUseOnnx(gpu, ExecutionProvider.Cuda).Should().BeFalse();
    }

    [Fact]
    public void CpuOnly_UsesGguf()
    {
        var gpu = new GpuInfo { Vendor = GpuVendor.Unknown };
        GeneratorRoutingPolicy.ShouldUseOnnx(gpu, ExecutionProvider.Cpu).Should().BeFalse();
    }

    [Fact]
    public void AppleCoreML_UsesGguf()
    {
        var gpu = new GpuInfo { Vendor = GpuVendor.Apple };
        GeneratorRoutingPolicy.ShouldUseOnnx(gpu, ExecutionProvider.CoreML).Should().BeFalse();
    }
}
