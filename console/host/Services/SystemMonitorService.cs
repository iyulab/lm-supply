using System.Diagnostics;
using System.Runtime.InteropServices;
using LMSupply.Console.Host.Models.Responses;
using LMSupply.Generator;
using LMSupply.Runtime;
using HostMemoryMetrics = LMSupply.Console.Host.Models.Responses.MemoryMetrics;

namespace LMSupply.Console.Host.Services;

/// <summary>
/// System resource monitoring service
/// </summary>
public sealed partial class SystemMonitorService : IDisposable
{
    private readonly ILogger<SystemMonitorService> _logger;
    private readonly Process _currentProcess;
    private DateTime _lastCpuTime;
    private TimeSpan _lastTotalProcessorTime;
    private bool _disposed;

    public SystemMonitorService(ILogger<SystemMonitorService> logger)
    {
        _logger = logger;
        _currentProcess = Process.GetCurrentProcess();
        _lastCpuTime = DateTime.UtcNow;
        _lastTotalProcessorTime = _currentProcess.TotalProcessorTime;
    }

    /// <summary>
    /// Gets the overall system status
    /// </summary>
    public SystemStatus GetStatus()
    {
        var gpuInfo = GetGpuInfo();
        var memoryMetrics = GetMemoryMetrics();

        return new SystemStatus
        {
            EngineReady = true, // ONNX Runtime is always available
            GpuAvailable = gpuInfo?.IsAvailable ?? false,
            GpuProvider = GetDetectedProvider(),
            GpuName = gpuInfo?.Name,
            CpuUsage = GetCpuUsage(),
            RamUsageMB = memoryMetrics.UsedMB,
            RamTotalMB = memoryMetrics.TotalMB,
            RamUsagePercent = memoryMetrics.UsagePercent,
            VramUsageMB = gpuInfo?.UsedVramMB,
            VramTotalMB = gpuInfo?.TotalVramMB,
            VramUsagePercent = gpuInfo?.VramUsagePercent,
            ProcessMemoryMB = _currentProcess.WorkingSet64 / (1024.0 * 1024.0),
            Timestamp = DateTime.UtcNow
        };
    }

    /// <summary>
    /// CPU usage (0-100), measured from the current process
    /// </summary>
    public float GetCpuUsage()
    {
        try
        {
            var currentTime = DateTime.UtcNow;
            var currentCpuTime = _currentProcess.TotalProcessorTime;

            var cpuUsedMs = (currentCpuTime - _lastTotalProcessorTime).TotalMilliseconds;
            var totalTimeMs = (currentTime - _lastCpuTime).TotalMilliseconds;

            _lastCpuTime = currentTime;
            _lastTotalProcessorTime = currentCpuTime;

            if (totalTimeMs > 0)
            {
                var cpuUsage = (cpuUsedMs / (Environment.ProcessorCount * totalTimeMs)) * 100;
                return (float)Math.Min(100, Math.Max(0, cpuUsage));
            }

            return 0;
        }
        catch (Exception ex)
        {
            Trace.TraceInformation($"[SystemMonitorService] CPU usage measurement failed: {ex.Message}");
            return 0;
        }
    }

    /// <summary>
    /// Memory metrics
    /// </summary>
    public HostMemoryMetrics GetMemoryMetrics()
    {
        try
        {
            var gcMemory = GC.GetGCMemoryInfo();
            var totalMemory = gcMemory.TotalAvailableMemoryBytes;
            var usedMemory = totalMemory - gcMemory.HighMemoryLoadThresholdBytes;

            // System-wide memory info
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                return GetWindowsMemoryMetrics();
            }

            return new HostMemoryMetrics
            {
                TotalMB = totalMemory / (1024.0 * 1024.0),
                UsedMB = _currentProcess.WorkingSet64 / (1024.0 * 1024.0),
                UsagePercent = 0
            };
        }
        catch (Exception ex)
        {
            LogGetMemoryMetricsFailed(_logger, ex);
            return new HostMemoryMetrics();
        }
    }

    /// <summary>
    /// GPU info
    /// </summary>
    public Models.Responses.GpuInfo? GetGpuInfo()
    {
        // Use HardwareDetector to get GPU info
        var provider = HardwareDetector.ResolveProvider(ExecutionProvider.Auto);

        if (provider == ExecutionProvider.Cpu)
        {
            return new Models.Responses.GpuInfo
            {
                IsAvailable = false,
                Name = "CPU Only",
                Provider = "CPU"
            };
        }

        return new Models.Responses.GpuInfo
        {
            IsAvailable = true,
            Name = GetGpuName(provider),
            Provider = provider.ToString(),
            // VRAM info requires an external tool such as nvidia-smi
            TotalVramMB = null,
            UsedVramMB = null
        };
    }

    /// <summary>
    /// Real-time metrics stream
    /// </summary>
    public async IAsyncEnumerable<SystemMetrics> StreamMetricsAsync(
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken,
        int intervalMs = 1000)
    {
        while (!cancellationToken.IsCancellationRequested)
        {
            yield return new SystemMetrics
            {
                CpuUsage = GetCpuUsage(),
                RamUsageMB = _currentProcess.WorkingSet64 / (1024.0 * 1024.0),
                Timestamp = DateTime.UtcNow
            };

            await Task.Delay(intervalMs, cancellationToken);
        }
    }

    private static string GetDetectedProvider()
    {
        var provider = HardwareDetector.ResolveProvider(ExecutionProvider.Auto);
        return provider.ToString();
    }

    private static string GetGpuName(ExecutionProvider provider)
    {
        return provider switch
        {
            ExecutionProvider.Cuda => "NVIDIA GPU (CUDA)",
            ExecutionProvider.CoreML => "Apple Silicon (CoreML)",
            _ => "Unknown"
        };
    }

    private static HostMemoryMetrics GetWindowsMemoryMetrics()
    {
        try
        {
            var memStatus = new MEMORYSTATUSEX { dwLength = (uint)Marshal.SizeOf<MEMORYSTATUSEX>() };
            if (GlobalMemoryStatusEx(ref memStatus))
            {
                return new HostMemoryMetrics
                {
                    TotalMB = memStatus.ullTotalPhys / (1024.0 * 1024.0),
                    UsedMB = (memStatus.ullTotalPhys - memStatus.ullAvailPhys) / (1024.0 * 1024.0),
                    UsagePercent = memStatus.dwMemoryLoad
                };
            }
        }
        catch (Exception ex)
        {
            Trace.TraceInformation($"[SystemMonitorService] Windows memory metrics failed: {ex.Message}");
        }

        return new HostMemoryMetrics();
    }

    [DllImport("kernel32.dll")]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool GlobalMemoryStatusEx(ref MEMORYSTATUSEX lpBuffer);

    [StructLayout(LayoutKind.Sequential)]
    private struct MEMORYSTATUSEX
    {
        public uint dwLength;
        public uint dwMemoryLoad;
        public ulong ullTotalPhys;
        public ulong ullAvailPhys;
        public ulong ullTotalPageFile;
        public ulong ullAvailPageFile;
        public ulong ullTotalVirtual;
        public ulong ullAvailVirtual;
        public ulong ullAvailExtendedVirtual;
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        _currentProcess.Dispose();
    }

    [LoggerMessage(Level = LogLevel.Warning, Message = "Failed to get memory metrics")]
    private static partial void LogGetMemoryMetricsFailed(ILogger logger, Exception exception);
}
