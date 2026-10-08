using System.Diagnostics;
using System.Text;
using System.Text.Json;
using LMSupply.Runtime;
using Microsoft.ML.OnnxRuntime;

namespace LMSupply.Inference;

/// <summary>
/// <see cref="ExecutionProvider.OpenVino"/>: the Intel OpenVINO execution provider, an ONNX Runtime plugin. Provisioned
/// on first use, registered with the process-wide ONNX Runtime environment once, and appended to a session on the
/// plugin's GPU device.
/// </summary>
internal static class OpenVinoExecutionProvider
{
    /// <summary>The name the plugin reports for its execution provider (also the active-provider name).</summary>
    internal const string EpName = "OpenVINOExecutionProvider";

    /// <summary>The name this library registers the plugin library under.</summary>
    internal const string RegistrationName = "LMSupply.OpenVINO";

    /// <summary>
    /// The plugin's provider option carrying OpenVINO device properties as JSON (<c>{"GPU": {"CACHE_DIR": ...}}</c>). The
    /// plugin ignores a bare <c>cache_dir</c> option and says so.
    /// </summary>
    internal const string LoadConfigOption = "load_config";

    private static readonly SemaphoreSlim s_gate = new(1, 1);
    private static volatile string? s_registeredLibrary;

    /// <summary>Whether the plugin library is registered with ONNX Runtime in this process.</summary>
    internal static bool IsRegistered => s_registeredLibrary is not null;

    /// <summary>
    /// Provisions the plugin (the base ONNX Runtime must already be provisioned) and registers it with ONNX Runtime,
    /// once per process. Registration is never undone: sessions created from it may live as long as the process.
    /// </summary>
    internal static async Task EnsureRegisteredAsync(IProgress<DownloadProgress>? progress, CancellationToken cancellationToken)
    {
        if (IsRegistered)
            return;

        var library = await RuntimeManager.Instance.EnsureExecutionProviderPluginAsync(
            RuntimePackageRegistry.Providers.OpenVino, progress, cancellationToken).ConfigureAwait(false);

        await s_gate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            if (IsRegistered)
                return;
            OrtEnv.Instance().RegisterExecutionProviderLibrary(RegistrationName, library);
            s_registeredLibrary = library;
            Trace.TraceInformation($"[OpenVINO] Registered execution provider plugin {library}");
        }
        finally
        {
            s_gate.Release();
        }
    }

    /// <summary>The plugin's GPU devices, in the order ONNX Runtime lists them; empty when the plugin is not registered.</summary>
    internal static IReadOnlyList<OrtEpDevice> GpuDevices()
    {
        if (!IsRegistered)
            return [];
        return OrtEnv.Instance().GetEpDevices()
            .Where(d => string.Equals(d.EpName, EpName, StringComparison.Ordinal)
                        && d.HardwareDevice.Type == OrtHardwareDeviceType.GPU)
            .ToList();
    }

    /// <summary>
    /// Appends the provider on GPU device <paramref name="deviceId"/> (an index into <see cref="GpuDevices"/>). Returns
    /// false — the session then runs on CPU — when the plugin is not registered or has no such GPU device. The plugin's
    /// CPU device is never used: an explicit OpenVino request means the GPU, as an explicit CUDA request does.
    /// </summary>
    internal static bool TryAppend(SessionOptions options, int deviceId)
    {
        var devices = GpuDevices();
        if (deviceId < 0 || deviceId >= devices.Count)
        {
            Trace.TraceWarning(IsRegistered
                ? $"[OpenVINO] No GPU device {deviceId} (the plugin lists {devices.Count}); the session runs on CPU."
                : "[OpenVINO] The plugin is not registered; the session runs on CPU.");
            return false;
        }

        try
        {
            options.AppendExecutionProvider(OrtEnv.Instance(), [devices[deviceId]], new Dictionary<string, string>
            {
                [LoadConfigOption] = GpuLoadConfig(ModelCacheDirectory()),
            });
            return true;
        }
        catch (Exception ex)
        {
            Trace.TraceWarning($"[OpenVINO] Failed to add the provider: {ex.Message}");
            return false;
        }
    }

    /// <summary>
    /// OpenVINO's model cache for the GPU: compiled kernels and, per model, the compiled model (a blob of the order of the
    /// model's size). A later load of the same model, in any process, skips GPU compilation — on first use that compilation
    /// takes tens of seconds. It sits next to the registered plugin library, inside its version directory, because
    /// compiled output belongs to one OpenVINO version and goes away with it when the runtime cache is cleared.
    /// </summary>
    internal static string ModelCacheDirectory()
    {
        var dir = Path.Combine(Path.GetDirectoryName(s_registeredLibrary!)!, "model-cache");
        Directory.CreateDirectory(dir);
        return dir;
    }

    /// <summary>The <see cref="LoadConfigOption"/> value that turns on the GPU model cache in <paramref name="cacheDirectory"/>.</summary>
    internal static string GpuLoadConfig(string cacheDirectory)
    {
        using var buffer = new MemoryStream();
        using (var json = new Utf8JsonWriter(buffer))
        {
            json.WriteStartObject();
            json.WriteStartObject("GPU");
            json.WriteString("CACHE_DIR", cacheDirectory);
            json.WriteEndObject();
            json.WriteEndObject();
        }
        return Encoding.UTF8.GetString(buffer.ToArray());
    }
}
