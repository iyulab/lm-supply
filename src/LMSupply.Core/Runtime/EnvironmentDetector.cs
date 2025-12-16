using System.Runtime.InteropServices;

namespace LMSupply.Runtime;

/// <summary>
/// Detects the current execution environment including platform and GPU capabilities.
/// </summary>
public static class EnvironmentDetector
{
    private static PlatformInfo? _cachedPlatformInfo;
    private static GpuInfo? _cachedGpuInfo;
    private static IReadOnlyList<GpuInfo>? _cachedAllGpus;
    private static readonly object _lock = new();

    /// <summary>
    /// Detects and returns information about the current platform.
    /// Results are cached after first detection.
    /// </summary>
    public static PlatformInfo DetectPlatform()
    {
        if (_cachedPlatformInfo is not null)
            return _cachedPlatformInfo;

        lock (_lock)
        {
            _cachedPlatformInfo ??= new PlatformInfo
            {
                OS = GetOperatingSystem(),
                Architecture = RuntimeInformation.OSArchitecture,
                RuntimeIdentifier = GetRuntimeIdentifier()
            };
        }

        return _cachedPlatformInfo;
    }

    /// <summary>
    /// Detects and returns information about the primary GPU.
    /// Results are cached after first detection.
    /// </summary>
    public static GpuInfo DetectGpu()
    {
        if (_cachedGpuInfo is not null)
            return _cachedGpuInfo;

        lock (_lock)
        {
            _cachedGpuInfo ??= GpuDetector.DetectPrimaryGpu();
        }

        return _cachedGpuInfo;
    }

    /// <summary>
    /// Detects and returns information about all available GPUs.
    /// Results are cached after first detection.
    /// </summary>
    public static IReadOnlyList<GpuInfo> DetectAllGpus()
    {
        if (_cachedAllGpus is not null)
            return _cachedAllGpus;

        lock (_lock)
        {
            _cachedAllGpus ??= GpuDetector.DetectAllGpus();
        }

        return _cachedAllGpus;
    }

    /// <summary>
    /// Gets the recommended execution provider based on detected hardware.
    /// </summary>
    public static ExecutionProvider GetRecommendedProvider()
    {
        var gpu = DetectGpu();
        return gpu.RecommendedProvider;
    }

    /// <summary>
    /// Gets all available execution providers in priority order.
    /// </summary>
    public static IEnumerable<ExecutionProvider> GetAvailableProviders()
    {
        var platform = DetectPlatform();
        var gpu = DetectGpu();

        // CUDA for NVIDIA with sufficient driver version
        if (gpu.Vendor == GpuVendor.Nvidia && gpu.CudaDriverVersionMajor >= 11)
            yield return ExecutionProvider.Cuda;

        // DirectML for Windows with D3D12 support
        if (platform.IsWindows && gpu.DirectMLSupported)
            yield return ExecutionProvider.DirectML;

        // CoreML for macOS
        if (platform.IsMacOS && gpu.CoreMLSupported)
            yield return ExecutionProvider.CoreML;

        // CPU is always available
        yield return ExecutionProvider.Cpu;
    }

    /// <summary>
    /// Clears the cached detection results.
    /// Useful for testing or when hardware configuration changes.
    /// </summary>
    public static void ClearCache()
    {
        lock (_lock)
        {
            _cachedPlatformInfo = null;
            _cachedGpuInfo = null;
            _cachedAllGpus = null;
        }
    }

    /// <summary>
    /// Gets a summary string of the detected environment.
    /// </summary>
    public static string GetEnvironmentSummary()
    {
        var platform = DetectPlatform();
        var gpu = DetectGpu();
        var recommended = GetRecommendedProvider();

        return $"""
            Platform: {platform}
            GPU: {gpu}
            Recommended Provider: {recommended}
            Available Providers: {string.Join(", ", GetAvailableProviders())}
            """;
    }

    private static OSPlatform GetOperatingSystem()
    {
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            return OSPlatform.Windows;
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
            return OSPlatform.Linux;
        if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
            return OSPlatform.OSX;
        if (RuntimeInformation.IsOSPlatform(OSPlatform.FreeBSD))
            return OSPlatform.FreeBSD;

        // Default to Linux for unknown Unix-like systems
        return OSPlatform.Linux;
    }

    private static string GetRuntimeIdentifier()
    {
        // Use the actual RID if available
        var rid = RuntimeInformation.RuntimeIdentifier;
        if (!string.IsNullOrEmpty(rid))
            return rid;

        // Build RID from components
        var os = GetOperatingSystem();
        var arch = RuntimeInformation.OSArchitecture;

        var osPrefix = os switch
        {
            _ when os == OSPlatform.Windows => "win",
            _ when os == OSPlatform.Linux => "linux",
            _ when os == OSPlatform.OSX => "osx",
            _ when os == OSPlatform.FreeBSD => "freebsd",
            _ => "linux"
        };

        var archSuffix = arch switch
        {
            Architecture.X64 => "x64",
            Architecture.X86 => "x86",
            Architecture.Arm64 => "arm64",
            Architecture.Arm => "arm",
            _ => "x64"
        };

        return $"{osPrefix}-{archSuffix}";
    }
}
