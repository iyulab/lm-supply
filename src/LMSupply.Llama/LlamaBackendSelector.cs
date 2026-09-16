using System.Diagnostics;
using LMSupply.Hardware;
using LMSupply.Llama.Server;
using LMSupply.Runtime;

namespace LMSupply.Llama;

/// <summary>
/// Single source of truth for mapping an <see cref="ExecutionProvider"/> to a concrete
/// <see cref="LlamaServerBackend"/> for llama-server (GGUF) inference.
/// Used by the generator, embedder, and reranker so all three pick the same backend for the
/// same hardware — replacing three drifting copies of the same vendor switch.
/// </summary>
/// <remarks>
/// <para>
/// For <see cref="ExecutionProvider.Auto"/> the selection is vendor-driven but gated by the actual
/// VRAM budget: when a GPU backend would be chosen yet the budget cannot support a meaningful
/// offload (&lt; <see cref="MinVramForGpuOffloadBytes"/>), the backend is demoted to
/// <see cref="LlamaServerBackend.Cpu"/>. This avoids downloading and initializing a GPU
/// llama-server binary (e.g. Vulkan on an integrated Intel iGPU reporting 128 MB of dedicated
/// VRAM) only to offload zero layers — the case the floored-context CPU fallback cannot catch
/// because zero GPU layers consume no VRAM and never floor the context.
/// </para>
/// <para>
/// An explicit GPU pin (<see cref="ExecutionProvider.Cuda"/>/
/// <see cref="ExecutionProvider.CoreML"/>) is honored as requested and never demoted — the caller
/// asked for it, so it surfaces honestly (and downstream load may fail fast) rather than being
/// silently swapped.
/// </para>
/// </remarks>
public static class LlamaBackendSelector
{
    /// <summary>
    /// Minimum VRAM budget (bytes) for an <see cref="ExecutionProvider.Auto"/> GPU backend to be
    /// kept instead of demoted to CPU. Mirrors the system-overhead reservation in
    /// <c>MemoryEstimator.EstimateForGguf</c> (2 GB): below this, GPU layer offload is always zero,
    /// so a GPU binary would be downloaded and initialized for no benefit.
    /// </summary>
    public const long MinVramForGpuOffloadBytes = 2L * 1024 * 1024 * 1024;

    /// <summary>
    /// Maps an execution provider to a llama-server backend for the given GPU.
    /// Explicit providers map directly; <see cref="ExecutionProvider.Auto"/> delegates to
    /// <see cref="SelectAutoBackend"/>.
    /// </summary>
    public static LlamaServerBackend MapProvider(ExecutionProvider provider, GpuInfo gpu)
    {
        // Same refusal as every ONNX path: an explicit DirectML pin is not quietly mapped to Vulkan.
        ExecutionProviderSupport.ThrowIfUnsupported(provider);
        return provider switch
        {
            ExecutionProvider.Cpu => LlamaServerBackend.Cpu,
            ExecutionProvider.Cuda => LlamaServerBackend.Cuda12,
            ExecutionProvider.CoreML => LlamaServerBackend.Metal,
            ExecutionProvider.Auto => SelectAutoBackend(gpu),
            _ => LlamaServerBackend.Cpu
        };
    }

    /// <summary>
    /// Selects the best llama-server backend for the given GPU under Auto, gated by VRAM budget.
    /// </summary>
    public static LlamaServerBackend SelectAutoBackend(GpuInfo gpu)
    {
        var vendorBackend = SelectVendorBackend(gpu);
        if (vendorBackend == LlamaServerBackend.Cpu)
            return LlamaServerBackend.Cpu;

        // Metal is exempt from the VRAM-budget gate: Apple Silicon uses unified memory, so a
        // dedicated-VRAM reading is unavailable (budget would be 0) yet Metal acceleration is
        // always beneficial. The gate only governs backends bound by dedicated VRAM.
        if (!IsDedicatedVramBackend(vendorBackend))
            return vendorBackend;

        var hasBudgetOverride = VramBudget.TryGetEnvOverrideBytes(out _);

        // Integrated-GPU gate (D2): an iGPU's dedicated-VRAM reading is unreliable (memory is carved
        // from shared system RAM), so do not trust it to keep a GPU backend. Prefer CPU unless the
        // user forces a budget via LMSUPPLY_VRAM_BUDGET_MB. This closes the gap where a BIOS that
        // pre-allocates a large (e.g. 4GB) iGPU aperture would otherwise pass the VRAM gate below.
        if (gpu.IsIntegrated && !hasBudgetOverride)
        {
            Trace.TraceInformation(
                $"[LlamaBackendSelector] Auto: integrated GPU '{gpu.DeviceName}' -> CPU " +
                $"(shared-memory adapter; dedicated-VRAM reading unreliable). " +
                $"Set LMSUPPLY_VRAM_BUDGET_MB to force a GPU budget.");
            return LlamaServerBackend.Cpu;
        }

        // VRAM-budget gate (D1): a dedicated-VRAM backend whose budget is too small for any
        // meaningful offload is worse than CPU — it pays GPU init cost for zero accelerated layers.
        // VramBudget honors the LMSUPPLY_VRAM_BUDGET_MB override, so a user who knows their iGPU
        // shares system memory can force a GPU budget and keep the GPU backend.
        var budget = VramBudget.GetAvailableBytes(gpu);
        if (budget < MinVramForGpuOffloadBytes)
        {
            Trace.TraceInformation(
                $"[LlamaBackendSelector] Auto: GPU backend '{vendorBackend}' demoted to CPU — " +
                $"VRAM budget {budget / (1024 * 1024)}MB below the {MinVramForGpuOffloadBytes / (1024 * 1024)}MB " +
                $"offload threshold (no GPU layers would fit). Set LMSUPPLY_VRAM_BUDGET_MB to override.");
            return LlamaServerBackend.Cpu;
        }

        return vendorBackend;
    }

    /// <summary>
    /// True for backends bound by dedicated GPU VRAM (subject to the budget gate). Metal is excluded
    /// because Apple Silicon shares unified memory with the CPU and has no separate VRAM reading.
    /// </summary>
    private static bool IsDedicatedVramBackend(LlamaServerBackend backend) => backend switch
    {
        LlamaServerBackend.Cuda12 => true,
        LlamaServerBackend.Cuda13 => true,
        LlamaServerBackend.Vulkan => true,
        LlamaServerBackend.Hip => true,
        LlamaServerBackend.Sycl => true,
        _ => false   // Cpu, Metal
    };

    /// <summary>
    /// Vendor-driven backend selection (no VRAM gate). NVIDIA → CUDA, Apple → Metal,
    /// AMD → ROCm/HIP on Linux else Vulkan, modern Intel iGPU → Vulkan (legacy → CPU),
    /// any other Direct3D 12 capable GPU → Vulkan, otherwise CPU.
    /// </summary>
    private static LlamaServerBackend SelectVendorBackend(GpuInfo gpu) => gpu.Vendor switch
    {
        // NVIDIA: prefer CUDA for best performance.
        GpuVendor.Nvidia => LlamaServerBackend.Cuda12,

        // AMD: ROCm/HIP on Linux, Vulkan on Windows.
        GpuVendor.Amd => OperatingSystem.IsLinux()
            ? LlamaServerBackend.Hip
            : LlamaServerBackend.Vulkan,

        // Intel: Vulkan for modern iGPUs (Iris, Arc, Xe, UHD 600+), CPU for legacy (HD Graphics).
        // Note: Intel iGPUs use shared memory, so TotalMemoryBytes is not reliable for sizing.
        GpuVendor.Intel => IsModernIntelGpu(gpu.DeviceName)
            ? LlamaServerBackend.Vulkan
            : LlamaServerBackend.Cpu,

        // Apple: Metal.
        GpuVendor.Apple => LlamaServerBackend.Metal,

        // Unknown vendor but Direct3D 12 capable → Vulkan is the cross-platform GPU path.
        _ when gpu.DirectMLSupported => LlamaServerBackend.Vulkan,

        // Fallback to CPU.
        _ => LlamaServerBackend.Cpu
    };

    /// <summary>
    /// Checks if the Intel GPU is modern enough to use Vulkan acceleration.
    /// Modern: Iris, Arc, Xe, UHD 600+ series. Legacy: HD Graphics 4000 and older → CPU.
    /// </summary>
    internal static bool IsModernIntelGpu(string? deviceName)
    {
        if (string.IsNullOrEmpty(deviceName))
            return false;

        var name = deviceName.ToUpperInvariant();

        // Modern Intel GPUs that work well with Vulkan.
        if (name.Contains("IRIS") || name.Contains("ARC"))
            return true;

        // UHD Graphics 600 series and newer (Gen 9.5+).
        if (name.Contains("UHD"))
            return true;

        // Intel Xe Graphics.
        if (name.Contains(" XE"))
            return true;

        // Legacy HD Graphics (4000/5000/6000) are too old for good Vulkan performance → CPU.
        return false;
    }
}
