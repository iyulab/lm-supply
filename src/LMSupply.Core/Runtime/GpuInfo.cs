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
    /// Gets whether DirectML is supported (Windows only).
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
        _ when DirectMLSupported => ExecutionProvider.DirectML,
        _ => ExecutionProvider.Cpu
    };

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
