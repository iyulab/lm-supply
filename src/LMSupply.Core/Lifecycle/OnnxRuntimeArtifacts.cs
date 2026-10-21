using LMSupply.Inference;
using LMSupply.Runtime;

namespace LMSupply.Lifecycle;

/// <summary>
/// The ONNX Runtime native packages an ONNX model's session runs on, as a model kind's <see cref="IModelArtifacts"/> counts
/// them: whether they are on this machine, how much is still to download, and the download — for the provider a session of
/// the given <see cref="ExecutionProvider"/> is created on (Auto: CUDA when the hardware, driver and CUDA libraries allow it,
/// else CPU). The packages are fetched from the package feed by <see cref="RuntimeManager.Instance"/>, whose options decide
/// where they are cached and whether they may be downloaded.
/// </summary>
public static class OnnxRuntimeArtifacts
{
    /// <summary>Whether a session would find its runtime without downloading it. Reads the cache only.</summary>
    public static async Task<bool> IsAvailableOfflineAsync(ExecutionProvider provider, CancellationToken cancellationToken = default)
        => await IsAvailableOfflineAsync(await PackagesAsync(provider, cancellationToken), cancellationToken);

    /// <summary>
    /// Bytes a session would still download for its runtime: 0 when it is available offline. Reads the package feed for what
    /// is missing; downloads nothing. Null when that cannot be told (see <see cref="RuntimeManager.GetRuntimeDownloadBytesAsync"/>).
    /// </summary>
    public static async Task<long?> GetRemainingDownloadBytesAsync(ExecutionProvider provider, CancellationToken cancellationToken = default)
        => await GetRemainingDownloadBytesAsync(await PackagesAsync(provider, cancellationToken), cancellationToken);

    /// <summary>Downloads what a session's runtime needs into the cache, without loading it.</summary>
    public static async Task DownloadAsync(ExecutionProvider provider, IProgress<DownloadProgress>? progress = null, CancellationToken cancellationToken = default)
        => await DownloadAsync(await PackagesAsync(provider, cancellationToken), progress, cancellationToken);

    // The same for a given list of runtime packages — each a package, its provider slot and the version a load names (null:
    // the one it resolves) — for a backend that provisions others, such as ONNX Runtime GenAI beside ONNX Runtime.
    internal static async Task<bool> IsAvailableOfflineAsync(IReadOnlyList<RuntimePackage> packages, CancellationToken cancellationToken)
    {
        foreach (var p in packages)
        {
            if (!await Manager.IsRuntimeAvailableOfflineAsync(p.Package, p.Slot, p.Version, cancellationToken))
                return false;
        }

        return true;
    }

    internal static async Task<long?> GetRemainingDownloadBytesAsync(IReadOnlyList<RuntimePackage> packages, CancellationToken cancellationToken)
    {
        long total = 0;
        foreach (var p in packages)
        {
            if (await Manager.GetRuntimeDownloadBytesAsync(p.Package, p.Slot, p.Version, cancellationToken) is not { } bytes)
                return null;
            total += bytes;
        }

        return total;
    }

    internal static async Task DownloadAsync(IReadOnlyList<RuntimePackage> packages, IProgress<DownloadProgress>? progress, CancellationToken cancellationToken)
    {
        foreach (var p in packages)
            await Manager.DownloadRuntimeAsync(p.Package, p.Slot, p.Version, progress, cancellationToken);
    }

    internal readonly record struct RuntimePackage(string Package, string Slot, string? Version = null);

    private static readonly AsyncLocal<RuntimeManager?> s_manager = new();

    internal static RuntimeManager Manager => s_manager.Value ?? RuntimeManager.Instance;

    /// <summary>Answers from <paramref name="manager"/> instead of the process-wide one in this async flow, until disposed.</summary>
    internal static IDisposable UseManager(RuntimeManager manager)
    {
        var previous = s_manager.Value;
        s_manager.Value = manager;
        return new Restore(previous);
    }

    private sealed class Restore(RuntimeManager? previous) : IDisposable
    {
        public void Dispose() => s_manager.Value = previous;
    }

    // The packages OnnxSessionFactory provisions for a session of the provider, in the order it provisions them
    internal static async Task<IReadOnlyList<RuntimePackage>> PackagesAsync(ExecutionProvider provider, CancellationToken cancellationToken)
    {
        ExecutionProviderSupport.ThrowIfUnsupported(provider);
        switch (provider)
        {
            case ExecutionProvider.Cuda:
                return [new("onnxruntime", "cuda12")];
            case ExecutionProvider.OpenVino:
                return [new("onnxruntime", "cpu"), new("onnxruntime-ep", "openvino")];
            case ExecutionProvider.Auto:
                await Manager.InitializeAsync(cancellationToken);
                var chain = Manager.Gpu.GetFallbackProviders();
                var first = chain.FirstOrDefault(p => p != ExecutionProvider.Cuda || OnnxSessionFactory.CheckCudaRuntimeAvailability().Available);
                return [new("onnxruntime", first == ExecutionProvider.Cuda ? "cuda12" : "cpu")];
            default:
                return [new("onnxruntime", "cpu")];
        }
    }
}
