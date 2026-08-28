using AwesomeAssertions;
using LMSupply;
using LMSupply.Llama;
using LMSupply.Llama.Server;
using LMSupply.Runtime;

namespace LMSupply.Llama.Tests;

/// <summary>
/// Tests for <see cref="LlamaBackendSelector"/> — the single source of truth for mapping an
/// <see cref="ExecutionProvider"/> to a concrete <see cref="LlamaServerBackend"/> shared by the
/// generator, embedder, and reranker. Covers the VRAM-budget gate that demotes a dedicated-VRAM
/// GPU backend to CPU when no meaningful offload is possible (low-end / integrated GPU), and the
/// consistent Intel legacy→CPU handling that previously diverged across the three consumers.
/// HW-free: drives selection by passing constructed <see cref="GpuInfo"/> values.
/// </summary>
public class LlamaBackendSelectorTests
{
    private const long GB = 1024L * 1024 * 1024;
    private const long MB = 1024L * 1024;

    private static GpuInfo Gpu(GpuVendor vendor, string? name = null, long? total = null, bool directMl = false, long? shared = null)
        => new() { Vendor = vendor, DeviceName = name, TotalMemoryBytes = total, DirectMLSupported = directMl, SharedMemoryBytes = shared };

    // ─── Explicit providers map directly, independent of GPU and budget ───

    [Theory]
    [InlineData(ExecutionProvider.Cpu, LlamaServerBackend.Cpu)]
    [InlineData(ExecutionProvider.Cuda, LlamaServerBackend.Cuda12)]
    [InlineData(ExecutionProvider.DirectML, LlamaServerBackend.Vulkan)]
    [InlineData(ExecutionProvider.CoreML, LlamaServerBackend.Metal)]
    public void MapProvider_ExplicitPin_MapsDirectly_NoDemotion(ExecutionProvider provider, LlamaServerBackend expected)
    {
        // Even a 128MB iGPU must not change an explicit pin — the caller asked for it.
        var gpu = Gpu(GpuVendor.Intel, "Intel(R) Iris(R) Xe Graphics", total: 128 * MB);
        LlamaBackendSelector.MapProvider(provider, gpu).Should().Be(expected);
    }

    // ─── Auto: dedicated-VRAM GPU with sufficient budget keeps the GPU backend ───

    [Fact]
    public void Auto_Nvidia_SufficientVram_PicksCuda()
        => LlamaBackendSelector.MapProvider(ExecutionProvider.Auto, Gpu(GpuVendor.Nvidia, "RTX 4090", 24 * GB))
            .Should().Be(LlamaServerBackend.Cuda12);

    [Fact]
    public void Auto_IntelArc_DedicatedVram_PicksVulkan()
        => LlamaBackendSelector.MapProvider(ExecutionProvider.Auto, Gpu(GpuVendor.Intel, "Intel(R) Arc(TM) A770", 12 * GB))
            .Should().Be(LlamaServerBackend.Vulkan);

    // ─── Auto: D1 — dedicated-VRAM GPU with budget below the offload threshold demotes to CPU ───

    [Fact]
    public void Auto_IntelIrisXe_IntegratedTinyVram_DemotesToCpu()
    {
        // The discovery case: DXGI reports 128MB dedicated VRAM for the shared-memory iGPU.
        // Vulkan would be downloaded/initialized for zero offloaded layers → demote to CPU.
        var gpu = Gpu(GpuVendor.Intel, "Intel(R) Iris(R) Xe Graphics", total: 128 * MB);
        LlamaBackendSelector.MapProvider(ExecutionProvider.Auto, gpu).Should().Be(LlamaServerBackend.Cpu);
    }

    [Fact]
    public void Auto_Nvidia_TinyVram_DemotesToCpu()
        => LlamaBackendSelector.MapProvider(ExecutionProvider.Auto, Gpu(GpuVendor.Nvidia, "GT 710", 1 * GB))
            .Should().Be(LlamaServerBackend.Cpu);

    // ─── Auto: D2 — integrated GPU (shared memory) routed to CPU via IsIntegrated ───

    [Fact]
    public void Auto_IntegratedGpu_WithSharedMemory_DemotesToCpu()
    {
        // Iris Xe: tiny dedicated VRAM (128MB) + large shared system memory → IsIntegrated → CPU.
        var gpu = Gpu(GpuVendor.Intel, "Intel(R) Iris(R) Xe Graphics",
            total: 128 * MB, shared: 16 * GB);
        gpu.IsIntegrated.Should().BeTrue();
        LlamaBackendSelector.MapProvider(ExecutionProvider.Auto, gpu).Should().Be(LlamaServerBackend.Cpu);
    }

    [Fact]
    public void Auto_IntelArc_DedicatedWithSharedMemory_NotIntegrated_KeepsVulkan()
    {
        // Arc A770: 12GB dedicated VRAM even though shared memory is also present → not integrated.
        var gpu = Gpu(GpuVendor.Intel, "Intel(R) Arc(TM) A770", total: 12 * GB, shared: 16 * GB);
        gpu.IsIntegrated.Should().BeFalse();
        LlamaBackendSelector.MapProvider(ExecutionProvider.Auto, gpu).Should().Be(LlamaServerBackend.Vulkan);
    }

    // ─── Auto: D4 — legacy Intel HD goes to CPU consistently (was Vulkan in embedder/reranker) ───

    [Fact]
    public void Auto_LegacyIntelHd_AlwaysCpu_EvenWithLargeReportedVram()
    {
        // IsModernIntelGpu("HD Graphics 4000") == false → CPU regardless of (bogus) reported VRAM.
        var gpu = Gpu(GpuVendor.Intel, "Intel(R) HD Graphics 4000", total: 8 * GB);
        LlamaBackendSelector.MapProvider(ExecutionProvider.Auto, gpu).Should().Be(LlamaServerBackend.Cpu);
    }

    // ─── Auto: Apple Metal is exempt from the VRAM gate (unified memory, no dedicated reading) ───

    [Fact]
    public void Auto_AppleSilicon_NoVramReading_KeepsMetal()
        => LlamaBackendSelector.MapProvider(ExecutionProvider.Auto, Gpu(GpuVendor.Apple, "Apple Silicon", total: null))
            .Should().Be(LlamaServerBackend.Metal);

    // ─── Auto: unknown vendor falls back on DirectML capability + budget ───

    [Fact]
    public void Auto_UnknownVendor_DirectMlCapable_SufficientVram_PicksVulkan()
        => LlamaBackendSelector.MapProvider(ExecutionProvider.Auto, Gpu(GpuVendor.Unknown, "Some GPU", 8 * GB, directMl: true))
            .Should().Be(LlamaServerBackend.Vulkan);

    [Fact]
    public void Auto_NoGpu_PicksCpu()
        => LlamaBackendSelector.MapProvider(ExecutionProvider.Auto, Gpu(GpuVendor.Unknown, "CPU Only"))
            .Should().Be(LlamaServerBackend.Cpu);

    // ─── IsModernIntelGpu classification ───

    [Theory]
    [InlineData("Intel(R) Iris(R) Xe Graphics", true)]
    [InlineData("Intel(R) Arc(TM) A770", true)]
    [InlineData("Intel(R) UHD Graphics 770", true)]
    [InlineData("Intel(R) HD Graphics 4000", false)]
    [InlineData("Intel(R) HD Graphics 6000", false)]
    [InlineData(null, false)]
    [InlineData("", false)]
    public void IsModernIntelGpu_Classifies(string? name, bool expected)
        => LlamaBackendSelector.IsModernIntelGpu(name).Should().Be(expected);
}
