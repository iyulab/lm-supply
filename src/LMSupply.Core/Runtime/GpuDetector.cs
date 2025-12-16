using System.Runtime.InteropServices;

namespace LMSupply.Runtime;

/// <summary>
/// Detects GPU hardware capabilities without requiring CUDA toolkit or other SDK installations.
/// Uses NVML for NVIDIA detection (ships with display drivers).
/// </summary>
public static class GpuDetector
{
    /// <summary>
    /// Detects the primary GPU on the system.
    /// </summary>
    public static GpuInfo DetectPrimaryGpu()
    {
        var allGpus = DetectAllGpus();
        return allGpus.FirstOrDefault() ?? CreateCpuOnlyInfo();
    }

    /// <summary>
    /// Detects all available GPUs on the system.
    /// </summary>
    public static IReadOnlyList<GpuInfo> DetectAllGpus()
    {
        var gpus = new List<GpuInfo>();

        // Try NVIDIA detection via NVML
        var nvidiaGpus = NvmlDetector.DetectNvidiaGpus();
        gpus.AddRange(nvidiaGpus);

        // Check DirectML support on Windows
        var directMLSupported = DirectMLDetector.IsSupported();

        // Check CoreML support on macOS
        var coreMLSupported = CoreMLDetector.IsSupported();

        // If no discrete GPUs found, check for integrated/Apple Silicon
        if (gpus.Count == 0)
        {
            if (coreMLSupported)
            {
                gpus.Add(new GpuInfo
                {
                    Vendor = GpuVendor.Apple,
                    DeviceName = GetAppleSiliconName(),
                    CoreMLSupported = true,
                    DirectMLSupported = false
                });
            }
            else if (directMLSupported)
            {
                // Windows with DirectML but no NVIDIA - likely AMD or Intel integrated
                gpus.Add(new GpuInfo
                {
                    Vendor = GpuVendor.Unknown,
                    DeviceName = "DirectML Compatible GPU",
                    DirectMLSupported = true,
                    CoreMLSupported = false
                });
            }
        }
        else
        {
            // Update existing GPU entries with DirectML/CoreML support status
            for (int i = 0; i < gpus.Count; i++)
            {
                gpus[i] = gpus[i] with
                {
                    DirectMLSupported = directMLSupported,
                    CoreMLSupported = coreMLSupported
                };
            }
        }

        // If still no GPUs found, return CPU-only info
        if (gpus.Count == 0)
        {
            gpus.Add(CreateCpuOnlyInfo());
        }

        return gpus;
    }

    private static GpuInfo CreateCpuOnlyInfo() => new()
    {
        Vendor = GpuVendor.Unknown,
        DeviceName = "CPU Only",
        DirectMLSupported = false,
        CoreMLSupported = false
    };

    private static string GetAppleSiliconName()
    {
        // Apple Silicon detection
        if (RuntimeInformation.OSArchitecture == Architecture.Arm64 &&
            RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
        {
            return "Apple Silicon";
        }
        return "Apple GPU";
    }
}

/// <summary>
/// NVIDIA GPU detection using NVML (NVIDIA Management Library).
/// NVML ships with display drivers, no CUDA toolkit required.
/// Uses NativeLibrary for cross-platform library loading.
/// </summary>
internal static class NvmlDetector
{
    // NVML Return codes
    private const int NVML_SUCCESS = 0;

    // Library handle and function pointers
    private static IntPtr _nvmlHandle;
    private static bool _initialized;
    private static readonly object _initLock = new();

    // Function pointer delegates
    private delegate int NvmlInitDelegate();
    private delegate int NvmlShutdownDelegate();
    private delegate int NvmlGetCudaDriverVersionDelegate(out int version);
    private delegate int NvmlGetDeviceCountDelegate(out uint count);
    private delegate int NvmlGetDeviceByIndexDelegate(uint index, out IntPtr device);
    private delegate int NvmlGetDeviceNameDelegate(IntPtr device, IntPtr name, uint length);
    private delegate int NvmlGetMemoryInfoDelegate(IntPtr device, out NvmlMemory memory);
    private delegate int NvmlGetComputeCapabilityDelegate(IntPtr device, out int major, out int minor);

    // Cached function pointers
    private static NvmlInitDelegate? _nvmlInit;
    private static NvmlShutdownDelegate? _nvmlShutdown;
    private static NvmlGetCudaDriverVersionDelegate? _nvmlGetCudaDriverVersion;
    private static NvmlGetDeviceCountDelegate? _nvmlGetDeviceCount;
    private static NvmlGetDeviceByIndexDelegate? _nvmlGetDeviceByIndex;
    private static NvmlGetDeviceNameDelegate? _nvmlGetDeviceName;
    private static NvmlGetMemoryInfoDelegate? _nvmlGetMemoryInfo;
    private static NvmlGetComputeCapabilityDelegate? _nvmlGetComputeCapability;

    public static IReadOnlyList<GpuInfo> DetectNvidiaGpus()
    {
        var gpus = new List<GpuInfo>();

        try
        {
            if (!TryLoadNvmlLibrary())
                return gpus;

            if (!TryInitialize())
                return gpus;

            try
            {
                // Get CUDA driver version
                int cudaMajor = 0, cudaMinor = 0;
                if (TryGetCudaDriverVersion(out var cudaVersion))
                {
                    cudaMajor = cudaVersion / 1000;
                    cudaMinor = (cudaVersion % 1000) / 10;
                }

                // Get device count
                if (!TryGetDeviceCount(out var deviceCount) || deviceCount == 0)
                    return gpus;

                // Enumerate devices
                for (uint i = 0; i < deviceCount; i++)
                {
                    if (TryGetDeviceInfo(i, out var gpuInfo))
                    {
                        gpus.Add(gpuInfo with
                        {
                            CudaDriverVersionMajor = cudaMajor,
                            CudaDriverVersionMinor = cudaMinor
                        });
                    }
                }
            }
            finally
            {
                Shutdown();
            }
        }
        catch
        {
            // NVML not available or failed
        }

        return gpus;
    }

    private static bool TryLoadNvmlLibrary()
    {
        if (_nvmlHandle != IntPtr.Zero)
            return true;

        lock (_initLock)
        {
            if (_nvmlHandle != IntPtr.Zero)
                return true;

            // Try platform-specific library paths
            var libraryNames = GetNvmlLibraryNames();

            foreach (var libraryName in libraryNames)
            {
                if (NativeLibrary.TryLoad(libraryName, out _nvmlHandle))
                {
                    return TryLoadFunctionPointers();
                }
            }

            return false;
        }
    }

    private static string[] GetNvmlLibraryNames()
    {
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            var programFiles = Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles);
            var system32 = Environment.GetFolderPath(Environment.SpecialFolder.System);

            return new[]
            {
                Path.Combine(programFiles, "NVIDIA Corporation", "NVSMI", "nvml.dll"),
                Path.Combine(system32, "nvml.dll"),
                "nvml.dll",
                "nvml"
            };
        }
        else if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
        {
            return new[]
            {
                "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1",
                "/usr/lib64/libnvidia-ml.so.1",
                "/usr/lib/libnvidia-ml.so.1",
                "libnvidia-ml.so.1",
                "libnvidia-ml.so",
                "nvidia-ml"
            };
        }

        return Array.Empty<string>();
    }

    private static bool TryLoadFunctionPointers()
    {
        try
        {
            _nvmlInit = GetFunction<NvmlInitDelegate>("nvmlInit_v2");
            _nvmlShutdown = GetFunction<NvmlShutdownDelegate>("nvmlShutdown");
            _nvmlGetCudaDriverVersion = GetFunction<NvmlGetCudaDriverVersionDelegate>("nvmlSystemGetCudaDriverVersion_v2");
            _nvmlGetDeviceCount = GetFunction<NvmlGetDeviceCountDelegate>("nvmlDeviceGetCount_v2");
            _nvmlGetDeviceByIndex = GetFunction<NvmlGetDeviceByIndexDelegate>("nvmlDeviceGetHandleByIndex_v2");
            _nvmlGetDeviceName = GetFunction<NvmlGetDeviceNameDelegate>("nvmlDeviceGetName");
            _nvmlGetMemoryInfo = GetFunction<NvmlGetMemoryInfoDelegate>("nvmlDeviceGetMemoryInfo");
            _nvmlGetComputeCapability = GetFunction<NvmlGetComputeCapabilityDelegate>("nvmlDeviceGetCudaComputeCapability");

            return _nvmlInit is not null && _nvmlShutdown is not null;
        }
        catch
        {
            return false;
        }
    }

    private static T? GetFunction<T>(string name) where T : Delegate
    {
        if (_nvmlHandle == IntPtr.Zero)
            return null;

        if (NativeLibrary.TryGetExport(_nvmlHandle, name, out var address))
        {
            return Marshal.GetDelegateForFunctionPointer<T>(address);
        }

        return null;
    }

    private static bool TryInitialize()
    {
        if (_initialized)
            return true;

        try
        {
            if (_nvmlInit is null)
                return false;

            _initialized = _nvmlInit() == NVML_SUCCESS;
            return _initialized;
        }
        catch
        {
            return false;
        }
    }

    private static void Shutdown()
    {
        try
        {
            _nvmlShutdown?.Invoke();
            _initialized = false;
        }
        catch
        {
            // Ignore shutdown errors
        }
    }

    private static bool TryGetCudaDriverVersion(out int version)
    {
        version = 0;
        try
        {
            if (_nvmlGetCudaDriverVersion is null)
                return false;
            return _nvmlGetCudaDriverVersion(out version) == NVML_SUCCESS;
        }
        catch
        {
            return false;
        }
    }

    private static bool TryGetDeviceCount(out uint count)
    {
        count = 0;
        try
        {
            if (_nvmlGetDeviceCount is null)
                return false;
            return _nvmlGetDeviceCount(out count) == NVML_SUCCESS;
        }
        catch
        {
            return false;
        }
    }

    private static bool TryGetDeviceInfo(uint index, out GpuInfo gpuInfo)
    {
        gpuInfo = null!;

        try
        {
            if (_nvmlGetDeviceByIndex is null)
                return false;

            // Get device handle
            if (_nvmlGetDeviceByIndex(index, out var device) != NVML_SUCCESS)
                return false;

            // Get device name
            string? deviceName = null;
            if (_nvmlGetDeviceName is not null)
            {
                var nameBuffer = Marshal.AllocHGlobal(64);
                try
                {
                    if (_nvmlGetDeviceName(device, nameBuffer, 64) == NVML_SUCCESS)
                    {
                        deviceName = Marshal.PtrToStringAnsi(nameBuffer);
                    }
                }
                finally
                {
                    Marshal.FreeHGlobal(nameBuffer);
                }
            }

            // Get memory info
            long? totalMemory = null;
            if (_nvmlGetMemoryInfo is not null &&
                _nvmlGetMemoryInfo(device, out var memoryInfo) == NVML_SUCCESS)
            {
                totalMemory = (long)memoryInfo.total;
            }

            // Get compute capability
            int? ccMajor = null, ccMinor = null;
            if (_nvmlGetComputeCapability is not null &&
                _nvmlGetComputeCapability(device, out var major, out var minor) == NVML_SUCCESS)
            {
                ccMajor = major;
                ccMinor = minor;
            }

            gpuInfo = new GpuInfo
            {
                Vendor = GpuVendor.Nvidia,
                DeviceName = deviceName,
                TotalMemoryBytes = totalMemory,
                CudaComputeCapabilityMajor = ccMajor,
                CudaComputeCapabilityMinor = ccMinor
            };

            return true;
        }
        catch
        {
            return false;
        }
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct NvmlMemory
    {
        public ulong total;
        public ulong free;
        public ulong used;
    }
}

/// <summary>
/// DirectML support detection for Windows.
/// DirectML requires Windows 10 1903 (build 18362) or later with D3D12 support.
/// </summary>
internal static class DirectMLDetector
{
    // Windows build number for Windows 10 1903
    private const int MinimumWindowsBuild = 18362;

    public static bool IsSupported()
    {
        if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            return false;

        try
        {
            // Check Windows version
            var osVersion = Environment.OSVersion;
            if (osVersion.Version.Major < 10)
                return false;

            // Check build number for Windows 10+
            if (osVersion.Version.Build < MinimumWindowsBuild)
                return false;

            // DirectML requires D3D12, which is available on Windows 10 1903+
            // Additional check: verify d3d12.dll is available
            return IsD3D12Available();
        }
        catch
        {
            return false;
        }
    }

    private static bool IsD3D12Available()
    {
        try
        {
            // Check if D3D12.dll exists in System32
            var systemPath = Environment.GetFolderPath(Environment.SpecialFolder.System);
            var d3d12Path = Path.Combine(systemPath, "d3d12.dll");
            return File.Exists(d3d12Path);
        }
        catch
        {
            return false;
        }
    }
}

/// <summary>
/// CoreML support detection for macOS.
/// CoreML is available on macOS 10.13+ and works best with Apple Silicon.
/// </summary>
internal static class CoreMLDetector
{
    public static bool IsSupported()
    {
        if (!RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
            return false;

        try
        {
            // Apple Silicon (ARM64) has native CoreML/Metal support
            if (RuntimeInformation.OSArchitecture == Architecture.Arm64)
                return true;

            // Intel Macs also support CoreML but with less acceleration
            // Check macOS version (CoreML requires 10.13+)
            var osVersion = Environment.OSVersion;
            return osVersion.Version.Major >= 10 && osVersion.Version.Minor >= 13;
        }
        catch
        {
            return false;
        }
    }
}
