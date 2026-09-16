using System.Diagnostics;
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
        return allGpus.Count > 0 ? allGpus[0] : CreateCpuOnlyInfo();
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
        Trace.TraceInformation(
            $"[GpuDetector] NVML probe: found {nvidiaGpus.Count} NVIDIA GPU(s)" +
            (nvidiaGpus.Count > 0
                ? string.Join(string.Empty, nvidiaGpus.Select(g =>
                    $" | {g.DeviceName ?? "<unnamed>"} total={FormatBytes(g.TotalMemoryBytes)} free={FormatBytes(g.FreeMemoryBytes)}"))
                : string.Empty));

        // Check DirectML support on Windows
        var directMLSupported = DirectMLDetector.IsSupported();

        // Check CoreML support on macOS
        var coreMLSupported = CoreMLDetector.IsSupported();

        // If no discrete GPUs found via NVML, try DXGI on Windows
        if (gpus.Count == 0 && directMLSupported)
        {
            var dxgiGpus = DxgiDetector.DetectGpus();
            gpus.AddRange(dxgiGpus.Select(g => g with
            {
                DirectMLSupported = true,
                CoreMLSupported = false
            }));
            Trace.TraceInformation(
                $"[GpuDetector] DXGI fallback: enumerated {dxgiGpus.Count} adapter(s)" +
                (dxgiGpus.Count > 0
                    ? string.Join(string.Empty, dxgiGpus.Select(g =>
                        $" | vendor={g.Vendor} {g.DeviceName ?? "<unnamed>"} dedicated={FormatBytes(g.TotalMemoryBytes)}"))
                    : string.Empty));
        }

        // If still no GPUs, check for Apple Silicon
        if (gpus.Count == 0 && coreMLSupported)
        {
            gpus.Add(new GpuInfo
            {
                Vendor = GpuVendor.Apple,
                DeviceName = GetAppleSiliconName(),
                CoreMLSupported = true,
                DirectMLSupported = false
            });
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

    private static string FormatBytes(long? bytes)
    {
        if (bytes is null)
            return "n/a";
        const double mb = 1024.0 * 1024.0;
        return $"{bytes.Value / mb:F0}MB";
    }

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
        catch (Exception ex)
        {
            Trace.TraceInformation($"[GpuDetector] NVML detection failed: {ex.Message}");
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
        catch (Exception ex)
        {
            Trace.TraceInformation($"[GpuDetector] NVML function pointer loading failed: {ex.Message}");
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
        catch (Exception ex)
        {
            Trace.TraceInformation($"[GpuDetector] NVML initialization failed: {ex.Message}");
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
        catch (Exception ex)
        {
            Trace.TraceInformation($"[GpuDetector] NVML shutdown error: {ex.Message}");
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
        catch (Exception ex)
        {
            Trace.TraceInformation($"[GpuDetector] CUDA driver version query failed: {ex.Message}");
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
        catch (Exception ex)
        {
            Trace.TraceInformation($"[GpuDetector] NVML device count query failed: {ex.Message}");
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
            long? freeMemory = null;
            if (_nvmlGetMemoryInfo is not null &&
                _nvmlGetMemoryInfo(device, out var memoryInfo) == NVML_SUCCESS)
            {
                totalMemory = (long)memoryInfo.total;
                freeMemory = (long)memoryInfo.free;
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
                FreeMemoryBytes = freeMemory,
                CudaComputeCapabilityMajor = ccMajor,
                CudaComputeCapabilityMinor = ccMinor
            };

            return true;
        }
        catch (Exception ex)
        {
            Trace.TraceInformation($"[GpuDetector] NVML device info query failed: {ex.Message}");
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
/// Direct3D 12 capability detection for Windows (surfaces as <see cref="GpuInfo.DirectMLSupported"/>).
/// Requires Windows 10 1903 (build 18362) or later. Hardware information only since 0.67.0 -- it drives
/// the Vulkan llama-server backend, not an ONNX execution provider.
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
        catch (Exception ex)
        {
            Trace.TraceInformation($"[GpuDetector] DirectML support check failed: {ex.Message}");
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
        catch (Exception ex)
        {
            Trace.TraceInformation($"[GpuDetector] D3D12 availability check failed: {ex.Message}");
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
        catch (Exception ex)
        {
            Trace.TraceInformation($"[GpuDetector] CoreML support check failed: {ex.Message}");
            return false;
        }
    }
}

/// <summary>
/// DXGI-based GPU detection for Windows.
/// Uses DirectX Graphics Infrastructure to enumerate adapters and identify vendors.
/// </summary>
internal static class DxgiDetector
{
    // Vendor IDs
    private const uint VENDOR_NVIDIA = 0x10DE;
    private const uint VENDOR_AMD = 0x1002;
    private const uint VENDOR_INTEL = 0x8086;
    private const uint VENDOR_QUALCOMM = 0x17CB;
    private const uint VENDOR_MICROSOFT = 0x1414; // Microsoft Basic Render Driver

    // DXGI interfaces and methods
    private static readonly Guid IID_IDXGIFactory1 = new("770aae78-f26f-4dba-a829-253c83d1b387");

    // Function pointers
    private delegate int CreateDXGIFactory1Delegate(ref Guid riid, out IntPtr ppFactory);
    private static CreateDXGIFactory1Delegate? _createFactory;
    private static IntPtr _dxgiHandle;
    private static readonly object _initLock = new();

    public static IReadOnlyList<GpuInfo> DetectGpus()
    {
        var gpus = new List<GpuInfo>();

        if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            return gpus;

        try
        {
            if (!TryLoadDxgi())
                return gpus;

            var iid = IID_IDXGIFactory1;
            if (_createFactory!(ref iid, out var factoryPtr) != 0)
                return gpus;

            try
            {
                EnumerateAdapters(factoryPtr, gpus);
            }
            finally
            {
                Marshal.Release(factoryPtr);
            }
        }
        catch (Exception ex)
        {
            Trace.TraceInformation($"[GpuDetector] DXGI detection failed: {ex.Message}");
        }

        return gpus;
    }

    private static bool TryLoadDxgi()
    {
        if (_createFactory != null)
            return true;

        lock (_initLock)
        {
            if (_createFactory != null)
                return true;

            try
            {
                if (!NativeLibrary.TryLoad("dxgi.dll", out _dxgiHandle))
                    return false;

                if (!NativeLibrary.TryGetExport(_dxgiHandle, "CreateDXGIFactory1", out var funcPtr))
                    return false;

                _createFactory = Marshal.GetDelegateForFunctionPointer<CreateDXGIFactory1Delegate>(funcPtr);
                return true;
            }
            catch (Exception ex)
            {
                Trace.TraceInformation($"[GpuDetector] DXGI library loading failed: {ex.Message}");
                return false;
            }
        }
    }

    private static void EnumerateAdapters(IntPtr factory, List<GpuInfo> gpus)
    {
        // IDXGIFactory1::EnumAdapters1 is at vtable index 12
        var enumAdapters = Marshal.GetDelegateForFunctionPointer<EnumAdaptersDelegate>(
            GetVTableEntry(factory, 12));

        uint adapterIndex = 0;
        while (enumAdapters(factory, adapterIndex, out var adapterPtr) == 0)
        {
            try
            {
                var desc = GetAdapterDescription(adapterPtr);
                if (desc.HasValue && !IsBasicRenderDriver(desc.Value))
                {
                    var vendor = GetVendorFromId(desc.Value.VendorId);
                    gpus.Add(new GpuInfo
                    {
                        Vendor = vendor,
                        DeviceName = desc.Value.Description,
                        TotalMemoryBytes = (long)desc.Value.DedicatedVideoMemory,
                        SharedMemoryBytes = (long)desc.Value.SharedSystemMemory
                    });
                }
            }
            finally
            {
                Marshal.Release(adapterPtr);
            }
            adapterIndex++;
        }
    }

    private static DxgiAdapterDesc? GetAdapterDescription(IntPtr adapter)
    {
        // IDXGIAdapter1::GetDesc1 is at vtable index 10
        var getDesc = Marshal.GetDelegateForFunctionPointer<GetDescDelegate>(
            GetVTableEntry(adapter, 10));

        var descBuffer = Marshal.AllocHGlobal(Marshal.SizeOf<DXGI_ADAPTER_DESC1>());
        try
        {
            if (getDesc(adapter, descBuffer) != 0)
                return null;

            var nativeDesc = Marshal.PtrToStructure<DXGI_ADAPTER_DESC1>(descBuffer);
            return new DxgiAdapterDesc
            {
                Description = nativeDesc.Description,
                VendorId = nativeDesc.VendorId,
                DedicatedVideoMemory = nativeDesc.DedicatedVideoMemory,
                SharedSystemMemory = nativeDesc.SharedSystemMemory
            };
        }
        finally
        {
            Marshal.FreeHGlobal(descBuffer);
        }
    }

    private static IntPtr GetVTableEntry(IntPtr obj, int index)
    {
        var vtable = Marshal.ReadIntPtr(obj);
        return Marshal.ReadIntPtr(vtable, index * IntPtr.Size);
    }

    private static bool IsBasicRenderDriver(DxgiAdapterDesc desc)
    {
        // Skip Microsoft Basic Render Driver (software renderer)
        return desc.VendorId == VENDOR_MICROSOFT ||
               desc.Description.Contains("Basic Render", StringComparison.OrdinalIgnoreCase) ||
               desc.Description.Contains("Microsoft", StringComparison.OrdinalIgnoreCase);
    }

    private static GpuVendor GetVendorFromId(uint vendorId) => vendorId switch
    {
        VENDOR_NVIDIA => GpuVendor.Nvidia,
        VENDOR_AMD => GpuVendor.Amd,
        VENDOR_INTEL => GpuVendor.Intel,
        VENDOR_QUALCOMM => GpuVendor.Qualcomm,
        _ => GpuVendor.Unknown
    };

    // Delegates for COM vtable calls
    private delegate int EnumAdaptersDelegate(IntPtr factory, uint index, out IntPtr adapter);
    private delegate int GetDescDelegate(IntPtr adapter, IntPtr desc);

    // DXGI_ADAPTER_DESC1 structure
    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
    private struct DXGI_ADAPTER_DESC1
    {
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 128)]
        public string Description;
        public uint VendorId;
        public uint DeviceId;
        public uint SubSysId;
        public uint Revision;
        public nuint DedicatedVideoMemory;
        public nuint DedicatedSystemMemory;
        public nuint SharedSystemMemory;
        public long AdapterLuid;
        public uint Flags;
    }

    private readonly struct DxgiAdapterDesc
    {
        public string Description { get; init; }
        public uint VendorId { get; init; }
        public nuint DedicatedVideoMemory { get; init; }
        public nuint SharedSystemMemory { get; init; }
    }
}
