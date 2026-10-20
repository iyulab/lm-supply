using System.Diagnostics;
using System.Text.RegularExpressions;

namespace LMSupply.Llama.Server;

public sealed partial class LlamaServerProcess
{
    /// <summary>
    /// Returns true if the llama-server startup log shows an accelerated (non-CPU) compute device
    /// was enumerated. A GPU llama-server binary that cannot load its runtime (e.g. missing
    /// cudart/cublas) starts and serves normally but enumerates only a CPU device — llama.cpp
    /// silently runs on CPU. This predicate lets callers detect that silent fallback and warn
    /// instead of failing silently. Conservative: a null/empty log returns false (no GPU evidence).
    /// </summary>
    internal static bool StartupLogShowsGpuDevice(string? startupLog)
        => !string.IsNullOrWhiteSpace(startupLog) && GpuDeviceLineRegex.IsMatch(startupLog);

    /// <summary>
    /// What the startup log says about the devices: true when it lists an accelerated device, false when it lists
    /// devices and none is accelerated, and null when it lists none at all — builds from b11146 on no longer print
    /// <c>device_info</c> at the default verbosity, so its absence is not evidence of a CPU fallback.
    /// </summary>
    internal static bool? StartupLogDeviceEvidence(string? startupLog)
    {
        if (StartupLogShowsGpuDevice(startupLog))
            return true;
        return startupLog is not null && startupLog.Contains("device_info:", StringComparison.Ordinal) ? false : null;
    }

    // "--list-devices" output: "Available devices:" then one "  CUDA0: NVIDIA ..." line per device.
    private static readonly Regex ListedGpuDeviceRegex = new(
        @"^\s*(?:CUDA|Vulkan|Metal|ROCm|HIP|SYCL|CANN|OpenCL)\d*\s*:",
        RegexOptions.IgnoreCase | RegexOptions.Multiline | RegexOptions.Compiled);

    /// <summary>True when <c>--list-devices</c> output names an accelerated device.</summary>
    internal static bool ListDevicesOutputShowsGpuDevice(string? output)
        => !string.IsNullOrWhiteSpace(output) && ListedGpuDeviceRegex.IsMatch(output);

    /// <summary>
    /// Asks the same binary which devices it can load (<c>--list-devices</c> loads the backends and exits). Used only
    /// when the startup log is silent about devices. A probe that cannot run says nothing either way, so it answers
    /// true: the warning is for a fallback that was observed, not one that was merely not ruled out.
    /// </summary>
    private static bool ProbeListsGpuDevice(string serverPath)
    {
        try
        {
            using var probe = Process.Start(new ProcessStartInfo
            {
                FileName = serverPath,
                Arguments = "--list-devices",
                WorkingDirectory = Path.GetDirectoryName(serverPath)!,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true,
            });
            if (probe is null)
                return true;
            var stdout = probe.StandardOutput.ReadToEndAsync();
            var stderr = probe.StandardError.ReadToEndAsync();
            if (!probe.WaitForExit(TimeSpan.FromSeconds(15)))
            {
                try { probe.Kill(entireProcessTree: true); } catch (InvalidOperationException) { }
                return true;
            }
            return ListDevicesOutputShowsGpuDevice(stdout.Result + "\n" + stderr.Result);
        }
        catch (Exception ex) when (ex is System.ComponentModel.Win32Exception or InvalidOperationException or IOException)
        {
            Trace.TraceInformation($"[LlamaServerProcess] --list-devices probe could not run: {ex.Message}");
            return true;
        }
    }
}
