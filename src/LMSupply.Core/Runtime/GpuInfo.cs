namespace LMSupply.Runtime;

/// <summary>
/// Contains information about detected GPU hardware.
/// </summary>
public sealed record GpuInfo
{
    /// <summary>
    /// Gets the GPU vendor (NVIDIA, AMD, Intel, Apple, Unknown).
    /// </summary>
    public required GpuVendor Vendor { get; init; }

    /// <summary>
    /// Gets the GPU device name, if available.
    /// </summary>
    public string? DeviceName { get; init; }

    /// <summary>
    /// Gets the total GPU memory in bytes, if available.
    /// </summary>
    public long? TotalMemoryBytes { get; init; }

    /// <summary>
    /// Gets the free (available) GPU memory in bytes at detection time, if available.
    /// Only populated when NVML is available (NVIDIA GPUs).
    /// </summary>
    public long? FreeMemoryBytes { get; init; }

    /// <summary>
    /// Gets the shared system memory in bytes the adapter can address, if available.
    /// Populated from DXGI (<c>SharedSystemMemory</c>) on Windows. Large shared memory next to a
    /// tiny <see cref="TotalMemoryBytes"/> is the signature of an integrated GPU (see <see cref="IsIntegrated"/>).
    /// </summary>
    public long? SharedMemoryBytes { get; init; }

    /// <summary>
    /// Dedicated-VRAM ceiling (bytes) below which a non-NVIDIA/non-Apple adapter that also exposes
    /// shared system memory is treated as integrated. Discrete cards report multi-GB dedicated VRAM;
    /// integrated GPUs carve a small (often ≤512 MB) dedicated region from shared system RAM.
    /// </summary>
    public const long IntegratedDedicatedCeilingBytes = 2L * 1024 * 1024 * 1024;

    /// <summary>
    /// Gets whether this adapter is an integrated GPU (shares system RAM rather than having
    /// dedicated VRAM). Integrated GPUs report an unreliable dedicated-VRAM size, so VRAM-budget
    /// decisions should not trust <see cref="TotalMemoryBytes"/> for them. NVIDIA and Apple are
    /// never flagged here (discrete / unified-memory paths handle them).
    /// </summary>
    public bool IsIntegrated =>
        Vendor is not GpuVendor.Nvidia and not GpuVendor.Apple
        && SharedMemoryBytes is > 0
        && (TotalMemoryBytes ?? 0) < IntegratedDedicatedCeilingBytes;

    /// <summary>
    /// Gets the best estimate of available VRAM for model loading.
    /// Returns FreeMemoryBytes if detected, otherwise TotalMemoryBytes.
    /// </summary>
    public long? EffectiveAvailableBytes => FreeMemoryBytes ?? TotalMemoryBytes;

    /// <summary>
    /// Gets the CUDA compute capability major version (NVIDIA only).
    /// </summary>
    public int? CudaComputeCapabilityMajor { get; init; }

    /// <summary>
    /// Gets the CUDA compute capability minor version (NVIDIA only).
    /// </summary>
    public int? CudaComputeCapabilityMinor { get; init; }

    /// <summary>
    /// Gets the CUDA driver version major (NVIDIA only).
    /// </summary>
    public int? CudaDriverVersionMajor { get; init; }

    /// <summary>
    /// Gets the CUDA driver version minor (NVIDIA only).
    /// </summary>
    public int? CudaDriverVersionMinor { get; init; }

    /// <summary>
    /// Gets whether a Direct3D 12 capable GPU is present (Windows only). Hardware information only:
    /// it selects the Vulkan llama-server backend on AMD/Intel GPUs, but no ONNX execution provider --
    /// the DirectML provider is gone from ONNX Runtime 1.25+ (see <see cref="ExecutionProviderSupport"/>).
    /// </summary>
    public bool DirectMLSupported { get; init; }

    /// <summary>
    /// Gets whether CoreML/Metal is supported (macOS only).
    /// </summary>
    public bool CoreMLSupported { get; init; }

    /// <summary>
    /// Gets the recommended execution provider for this GPU.
    /// </summary>
    public ExecutionProvider RecommendedProvider => Vendor switch
    {
        GpuVendor.Nvidia when CudaDriverVersionMajor >= 11 => ExecutionProvider.Cuda,
        GpuVendor.Apple => ExecutionProvider.CoreML,
        _ => ExecutionProvider.Cpu
    };

    /// <summary>
    /// Gets a prioritized list of execution providers to try based on GPU capabilities.
    /// The fallback chain ensures zero-configuration GPU acceleration:
    /// CUDA → CoreML → CPU
    /// </summary>
    public IReadOnlyList<ExecutionProvider> GetFallbackProviders()
    {
        var providers = new List<ExecutionProvider>();

        // CUDA first (if NVIDIA with sufficient driver)
        if (Vendor == GpuVendor.Nvidia && CudaDriverVersionMajor >= 11)
            providers.Add(ExecutionProvider.Cuda);

        // CoreML (macOS/iOS)
        if (CoreMLSupported)
            providers.Add(ExecutionProvider.CoreML);

        // CPU always as final fallback
        providers.Add(ExecutionProvider.Cpu);

        return providers;
    }

    /// <summary>
    /// Gets the total GPU memory in megabytes, if available.
    /// </summary>
    public long? TotalMemoryMB => TotalMemoryBytes.HasValue
        ? TotalMemoryBytes.Value / (1024 * 1024)
        : null;

    /// <summary>
    /// Gets the CUDA compute capability string (e.g., "8.6").
    /// </summary>
    public string? CudaComputeCapability => CudaComputeCapabilityMajor.HasValue && CudaComputeCapabilityMinor.HasValue
        ? $"{CudaComputeCapabilityMajor}.{CudaComputeCapabilityMinor}"
        : null;

    public override string ToString()
    {
        var parts = new List<string> { Vendor.ToString() };
        if (!string.IsNullOrEmpty(DeviceName))
            parts.Add(DeviceName);
        if (TotalMemoryMB.HasValue)
            parts.Add($"{TotalMemoryMB}MB");
        if (CudaComputeCapability is not null)
            parts.Add($"CC {CudaComputeCapability}");
        return string.Join(" | ", parts);
    }
}

/// <summary>
/// GPU hardware vendor.
/// </summary>
public enum GpuVendor
{
    /// <summary>Unknown or no GPU detected.</summary>
    Unknown,

    /// <summary>NVIDIA GPU.</summary>
    Nvidia,

    /// <summary>AMD GPU.</summary>
    Amd,

    /// <summary>Intel GPU.</summary>
    Intel,

    /// <summary>Apple Silicon (Metal/CoreML).</summary>
    Apple,

    /// <summary>Qualcomm GPU.</summary>
    Qualcomm
}
