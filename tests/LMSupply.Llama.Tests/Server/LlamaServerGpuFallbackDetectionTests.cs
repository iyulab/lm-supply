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
}
