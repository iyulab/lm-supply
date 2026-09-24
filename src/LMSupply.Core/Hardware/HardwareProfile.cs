using System.Diagnostics;
using LMSupply.Runtime;

namespace LMSupply.Hardware;

/// <summary>
/// Unified hardware profile for adaptive model selection across all domains.
/// Provides a singleton instance with detected hardware capabilities.
/// </summary>
public sealed class HardwareProfile
{
    private static readonly Lazy<HardwareProfile> _current = new(Detect, LazyThreadSafetyMode.ExecutionAndPublication);

    private static readonly Lazy<HardwareProfile> _cpuOnly = new(DetectCpuOnly, LazyThreadSafetyMode.ExecutionAndPublication);

    /// <summary>
    /// Gets the current hardware profile (singleton, lazily initialized). Reading it probes the GPU, which loads the
    /// vendor driver libraries (NVML and the CUDA driver on an NVIDIA host).
    /// </summary>
    public static HardwareProfile Current => _current.Value;

    /// <summary>
    /// The profile a load with <paramref name="provider"/> works with. An explicit
    /// <see cref="ExecutionProvider.Cpu"/> gets a CPU-only profile — system memory and a CPU tier, no GPU — built without
    /// probing the GPU, so a caller that ruled the GPU out does not load its driver libraries. Every other provider gets
    /// <see cref="Current"/>. The CPU-only profile is never cached as <see cref="Current"/>: a later GPU or Auto load in the
    /// same process still sees the GPU.
    /// </summary>
    public static HardwareProfile For(ExecutionProvider provider)
        => provider == ExecutionProvider.Cpu ? _cpuOnly.Value : _current.Value;

    /// <summary>
    /// Gets the detected GPU information.
    /// </summary>
    public GpuInfo GpuInfo { get; private init; }

    /// <summary>
    /// Gets the total system memory in bytes.
    /// </summary>
    public long SystemMemoryBytes { get; private init; }

    /// <summary>
    /// Gets the performance tier based on available hardware.
    /// </summary>
    public PerformanceTier Tier { get; private init; }

    /// <summary>
    /// Gets the recommended execution provider for this hardware.
    /// </summary>
    public ExecutionProvider RecommendedProvider { get; private init; }

    /// <summary>
    /// Gets the GPU VRAM in gigabytes, or 0 if no GPU available.
    /// </summary>
    public double GpuMemoryGB => (GpuInfo.TotalMemoryBytes ?? 0) / (1024.0 * 1024 * 1024);

    /// <summary>
    /// Gets the system memory in gigabytes.
    /// </summary>
    public double SystemMemoryGB => SystemMemoryBytes / (1024.0 * 1024 * 1024);

    /// <summary>
    /// Gets whether GPU acceleration is available.
    /// </summary>
    public bool HasGpu => GpuInfo.Vendor != GpuVendor.Unknown ||
                          GpuInfo.DirectMLSupported ||
                          GpuInfo.CoreMLSupported;

    private HardwareProfile()
    {
        GpuInfo = new GpuInfo { Vendor = GpuVendor.Unknown, DeviceName = "Unknown" };
    }

    /// <summary>
    /// Detects hardware and creates a profile.
    /// </summary>
    private static HardwareProfile Detect()
    {
        var gpuInfo = GpuDetector.DetectPrimaryGpu();
        var systemMemory = GetSystemMemoryBytes();
        var provider = DetermineRecommendedProvider(gpuInfo);
        var tier = DeterminePerformanceTier(gpuInfo, systemMemory, provider);

        return new HardwareProfile
        {
            GpuInfo = gpuInfo,
            SystemMemoryBytes = systemMemory,
            RecommendedProvider = provider,
            Tier = tier
        };
    }

    private static HardwareProfile DetectCpuOnly()
    {
        var gpuInfo = new GpuInfo { Vendor = GpuVendor.Unknown, DeviceName = "Unknown" };
        var systemMemory = GetSystemMemoryBytes();

        return new HardwareProfile
        {
            GpuInfo = gpuInfo,
            SystemMemoryBytes = systemMemory,
            RecommendedProvider = ExecutionProvider.Cpu,
            Tier = DeterminePerformanceTier(gpuInfo, systemMemory, ExecutionProvider.Cpu)
        };
    }

    /// <summary>
    /// Determines the best execution provider based on GPU capabilities.
    /// </summary>
    private static ExecutionProvider DetermineRecommendedProvider(GpuInfo gpuInfo)
    {
        // CUDA has best performance for NVIDIA GPUs with sufficient VRAM
        if (gpuInfo.Vendor == GpuVendor.Nvidia && gpuInfo.TotalMemoryBytes >= 4L * 1024 * 1024 * 1024)
        {
            return ExecutionProvider.Cuda;
        }

        // CoreML for Apple Silicon
        if (gpuInfo.CoreMLSupported)
        {
            return ExecutionProvider.CoreML;
        }

        return ExecutionProvider.Cpu;
    }

    /// <summary>
    /// Determines performance tier based on hardware resources.
    /// </summary>
    private static PerformanceTier DeterminePerformanceTier(GpuInfo gpuInfo, long systemMemoryBytes, ExecutionProvider provider)
    {
        var gpuMemoryGB = (gpuInfo.TotalMemoryBytes ?? 0) / (1024.0 * 1024 * 1024);
        var systemMemoryGB = systemMemoryBytes / (1024.0 * 1024 * 1024);

        // GPU-based tiers (when GPU acceleration is available)
        if (provider != ExecutionProvider.Cpu)
        {
            return gpuMemoryGB switch
            {
                >= 16 => PerformanceTier.Ultra,
                >= 8 => PerformanceTier.High,
                >= 4 => PerformanceTier.Medium,
                _ => PerformanceTier.Low
            };
        }

        // CPU-only tiers (based on system memory)
        return systemMemoryGB switch
        {
            >= 32 => PerformanceTier.High,  // Lots of RAM can handle larger models
            >= 16 => PerformanceTier.Medium,
            >= 8 => PerformanceTier.Low,
            _ => PerformanceTier.Low
        };
    }

    /// <summary>
    /// Environment variable that overrides detected system RAM with an absolute value in megabytes.
    /// When set to a positive integer, that value is used for all RAM-aware decisions (CPU model
    /// auto-selection and quantization downscale). Use it to match a cgroup/container memory limit
    /// or to simulate a low-spec machine. Mirrors <see cref="VramBudget.BudgetOverrideEnvVar"/>.
    /// </summary>
    public const string SystemRamOverrideEnvVar = "LMSUPPLY_SYSTEM_RAM_MB";

    /// <summary>
    /// Gets total available system memory, honoring <see cref="SystemRamOverrideEnvVar"/>.
    /// </summary>
    private static long GetSystemMemoryBytes()
    {
        var raw = Environment.GetEnvironmentVariable(SystemRamOverrideEnvVar);
        if (!string.IsNullOrWhiteSpace(raw)
            && long.TryParse(raw.Trim(), System.Globalization.NumberStyles.Integer,
                System.Globalization.CultureInfo.InvariantCulture, out var mb)
            && mb > 0)
        {
            return mb * 1024L * 1024L;
        }

        try
        {
            return (long)GC.GetGCMemoryInfo().TotalAvailableMemoryBytes;
        }
        catch (Exception ex)
        {
            Trace.TraceInformation($"[HardwareProfile] Memory detection failed, assuming 8GB: {ex.Message}");
            return 8L * 1024 * 1024 * 1024;
        }
    }

    /// <summary>
    /// Gets a human-readable summary of the hardware profile.
    /// </summary>
    public string GetSummary()
    {
        var gpuDesc = GpuInfo.Vendor == GpuVendor.Unknown && GpuInfo.DeviceName == "CPU Only"
            ? "CPU only"
            : $"{GpuInfo.DeviceName} ({GpuMemoryGB:F1}GB VRAM)";

        return $"""
            Hardware Profile
            ================
            GPU: {gpuDesc}
            System Memory: {SystemMemoryGB:F1}GB
            Provider: {RecommendedProvider}
            Performance Tier: {Tier}
            """;
    }

    /// <summary>
    /// Forces re-detection of hardware profile (useful after driver updates).
    /// Note: This creates a new profile instance but doesn't replace Current.
    /// </summary>
    public static HardwareProfile Refresh() => Detect();
}
