using AwesomeAssertions;
using LMSupply;
using LMSupply.Generator.Abstractions;
using LMSupply.Generator.Internal;
using LMSupply.Runtime;

namespace LMSupply.Generator.Tests.Internal;

/// <summary>
/// Tests for GeneratorRoutingPolicy.ShouldUseOnnx — the single decision shared by LocalGenerator and
/// ModelFormatDetector for routing the auto path to ONNX (DirectML) vs GGUF (llama.cpp).
/// ONNX/DirectML is for a *discrete* non-NVIDIA GPU on Windows; integrated GPUs route to GGUF
/// (their VRAM reading is unreliable, so DirectML is not a real win — consistent with D1/D2 CPU demotion).
/// This assembly disables test parallelization (AssemblyInfo.cs), which is what makes mutating the
/// process-global <see cref="OnnxGeneratorBackendRegistry"/> in these tests safe.
/// </summary>
public class GeneratorRoutingPolicyTests
{
    private const long GB = 1024L * 1024 * 1024;
    private const long MB = 1024L * 1024;

    private static readonly GpuInfo DiscreteIntelArc = new()
    {
        Vendor = GpuVendor.Intel,
        DeviceName = "Intel(R) Arc(TM) A770",
        TotalMemoryBytes = 12 * GB,
        SharedMemoryBytes = 16 * GB,
    };

    private static readonly GpuInfo DiscreteAmd = new()
    {
        Vendor = GpuVendor.Amd,
        TotalMemoryBytes = 16 * GB,
        SharedMemoryBytes = 16 * GB,
    };

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
    public void DiscreteIntelArc_DirectML_WithoutBackendRegistered_UsesGguf()
    {
        DiscreteIntelArc.IsIntegrated.Should().BeFalse();

        // Qualifying hardware alone is not enough: a consumer who only referenced
        // LMSupply.Generator (no LMSupply.Generator.Onnx package) must never have "auto" pick a
        // backend it cannot load.
        GeneratorRoutingPolicy.ShouldUseOnnx(DiscreteIntelArc, ExecutionProvider.DirectML).Should().BeFalse();
    }

    [Fact]
    public void DiscreteIntelArc_DirectML_WithBackendRegistered_UsesOnnx()
    {
        WithFakeOnnxBackendRegistered(() =>
            GeneratorRoutingPolicy.ShouldUseOnnx(DiscreteIntelArc, ExecutionProvider.DirectML).Should().BeTrue());
    }

    [Fact]
    public void DiscreteAmd_DirectML_WithoutBackendRegistered_UsesGguf()
    {
        GeneratorRoutingPolicy.ShouldUseOnnx(DiscreteAmd, ExecutionProvider.DirectML).Should().BeFalse();
    }

    [Fact]
    public void DiscreteAmd_DirectML_WithBackendRegistered_UsesOnnx()
    {
        WithFakeOnnxBackendRegistered(() =>
            GeneratorRoutingPolicy.ShouldUseOnnx(DiscreteAmd, ExecutionProvider.DirectML).Should().BeTrue());
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

    /// <summary>
    /// Registers a no-op fake backend for the duration of <paramref name="assertion"/>, then resets
    /// the registry back to unregistered via the internal test-only reset (production code never
    /// unregisters — a process either referenced LMSupply.Generator.Onnx and registered once at
    /// startup, or it never did).
    /// </summary>
    private static void WithFakeOnnxBackendRegistered(Action assertion)
    {
        OnnxGeneratorBackendRegistry.Register(new NullOnnxGeneratorBackend());
        try
        {
            assertion();
        }
        finally
        {
            OnnxGeneratorBackendRegistry.ResetForTests();
        }
    }

    private sealed class NullOnnxGeneratorBackend : IOnnxGeneratorBackend
    {
        public IGeneratorModel CreateModel(string modelId, string modelPath, IChatFormatter chatFormatter, GeneratorOptions options, string? configBasePath = null)
            => throw new NotSupportedException("fake backend — not used by ShouldUseOnnx");

        public IOnnxGeneratorModelFactory CreateFactory(string cacheDirectory, ExecutionProvider provider)
            => throw new NotSupportedException("fake backend — not used by ShouldUseOnnx");

        public Task EnsureRuntimeAsync(ExecutionProvider provider, IProgress<LMSupply.DownloadProgress>? progress, CancellationToken cancellationToken)
            => throw new NotSupportedException("fake backend — not used by ShouldUseOnnx");
    }
}
