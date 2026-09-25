using AwesomeAssertions;
using LMSupply.Llama.Server;

namespace LMSupply.Llama.Tests.Server;

/// <summary>
/// Tests for the silent-CPU-fallback detector. A CUDA/GPU llama-server binary that cannot load its
/// runtime (e.g. cudart/cublas missing) starts successfully and serves, but llama.cpp silently runs
/// CPU-only — no error. <see cref="LlamaServerProcess.StartupLogShowsGpuDevice"/> parses the server's
/// own device_info/system_info so a GPU backend that fell back to CPU can be detected and warned
/// about instead of failing silently.
///
/// HW-free: drives detection from captured real startup-log fragments (no GPU, no process launch).
/// </summary>
public class LlamaServerGpuFallbackDetectionTests
{
    // Captured from llama-server b9692 (cuda12 binary) on RTX 4060 WITHOUT cudart present:
    // ggml-cuda.dll fails to load, so only CPU is enumerated as a device.
    private const string CpuOnlyLog =
        "0.00.230.154 I device_info:\n" +
        "0.00.230.158 I   - CPU     : 13th Gen Intel(R) Core(TM) i9-13900HX (32487 MiB, 22489 MiB free)\n" +
        "0.00.230.204 I system_info: n_threads = 24 (n_threads_batch = 24) / 32 | CPU : SSE3 = 1 | SSSE3 = 1 | AVX = 1 | AVX2 = 1 | FMA = 1 | OPENMP = 1 |\n" +
        "0.00.235.799 I common_init_result: fitting params to device memory ...\n" +
        "0.00.755.448 I common_params_fit_impl: projected to use 3131 MiB of host memory vs. 32487 MiB of total host memory\n";

    // Captured from the same binary WITH cudart present: the CUDA device is enumerated and engaged.
    private const string CudaEngagedLog =
        "0.00.460.559 I device_info:\n" +
        "0.00.621.607 I   - CUDA0   : NVIDIA GeForce RTX 4060 Laptop GPU (8187 MiB, 7099 MiB free)\n" +
        "0.00.621.672 I system_info: n_threads = 24 (n_threads_batch = 24) / 32 | CUDA : ARCHS = 500,610,700,750,800,860,890,900 | USE_GRAPHS = 1 | CPU : SSE3 = 1 | AVX2 = 1 |\n";

    [Fact]
    public void CpuOnlyLog_ShowsNoGpuDevice()
    {
        LlamaServerProcess.StartupLogShowsGpuDevice(CpuOnlyLog).Should().BeFalse(
            "a CUDA binary that silently fell back to CPU enumerates only a CPU device");
    }

    [Fact]
    public void CudaEngagedLog_ShowsGpuDevice()
    {
        LlamaServerProcess.StartupLogShowsGpuDevice(CudaEngagedLog).Should().BeTrue(
            "an engaged CUDA device appears as a 'CUDA0' device line in device_info");
    }

    [Theory]
    [InlineData("   - Vulkan0 : Intel(R) Arc(TM) Graphics (8000 MiB free)")]
    [InlineData("   - Metal   : Apple M3 Max")]
    [InlineData("   - ROCm0   : AMD Radeon RX 7900 XTX")]
    public void NonCpuDeviceLine_ShowsGpuDevice(string deviceLine)
    {
        var log = "device_info:\n" + deviceLine + "\n";
        LlamaServerProcess.StartupLogShowsGpuDevice(log).Should().BeTrue(
            "any non-CPU compute device line means the GPU backend engaged");
    }

    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData("   ")]
    public void NullOrEmptyLog_ShowsNoGpuDevice(string? log)
    {
        LlamaServerProcess.StartupLogShowsGpuDevice(log).Should().BeFalse(
            "absent log evidence must not be treated as a GPU device (conservative: warn)");
    }

    // Captured from llama-server b11146 (cuda12) on RTX 4060 with the GPU in use (3.1 GB VRAM, 85-90% utilisation
    // during generation): the default verbosity no longer prints device_info at all.
    private const string B11146EngagedLog =
        "0.00.001.078 I srv  llama_server: initializing ...\n" +
        "0.00.140.328 I cmn  common_param: common_params_print_info: verbosity = 3 (adjust with the `-lv N` CLI arg)\n" +
        "0.05.937.721 I cmn          init: llama threadpool init, n_threads = 24\n" +
        "0.06.031.603 I srv    load_model: initializing, n_slots = 4, n_ctx_slot = 2048, kv_unified = 'true'\n" +
        "0.06.041.657 I srv  llama_server: model loaded\n";

    [Fact]
    public void ALogWithoutADeviceSection_IsNotEvidenceOfACpuFallback()
    {
        // Until 0.77.0 this read as "CPU-only" and warned on every load with b11146, while the GPU was working.
        LlamaServerProcess.StartupLogDeviceEvidence(B11146EngagedLog).Should().BeNull();
    }

    [Fact]
    public void DeviceEvidence_ReadsBothOlderLogShapes()
    {
        LlamaServerProcess.StartupLogDeviceEvidence(CudaEngagedLog).Should().BeTrue();
        LlamaServerProcess.StartupLogDeviceEvidence(CpuOnlyLog).Should().BeFalse();
    }

    [Fact]
    public void ListDevicesOutput_NamingACudaDevice_ShowsTheGpuLoaded()
    {
        // Captured from `llama-server --list-devices` (b11146 and b10964 print the same shape).
        const string output = "Available devices:\n  CUDA0: NVIDIA GeForce RTX 4060 Laptop GPU (8187 MiB, 7099 MiB free)\n";

        LlamaServerProcess.ListDevicesOutputShowsGpuDevice(output).Should().BeTrue();
    }

    [Fact]
    public void ListDevicesOutput_WithNoDevice_ShowsNoGpu()
    {
        LlamaServerProcess.ListDevicesOutputShowsGpuDevice("Available devices:\n").Should().BeFalse();
    }
}
