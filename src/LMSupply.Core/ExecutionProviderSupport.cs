using System.Diagnostics;
using LMSupply.Runtime;

namespace LMSupply;

/// <summary>
/// The one place that knows which <see cref="ExecutionProvider"/> values this build can serve, and
/// says so out loud. DirectML left with ONNX Runtime 1.25: the Microsoft.ML.OnnxRuntime.DirectML
/// package line ends at 1.24.4, the base Windows package carries no DirectML provider, and a managed
/// 1.30.0 runtime cannot bind a 1.24.4 native. So an explicit request cannot be honoured on any path
/// (ONNX session, GenAI, llama-server), and <see cref="ExecutionProvider.Auto"/> must not try it — a
/// machine that has cached the 1.24.4 native from an older release keeps a working provider only by
/// accident, and a fresh machine got a 404 on every session before this class existed.
/// </summary>
public static class ExecutionProviderSupport
{
    /// <summary>
    /// The reason DirectML is refused, in the words the caller sees.
    /// </summary>
    public const string DirectMLUnavailableMessage =
        "The DirectML execution provider is not available on this ONNX Runtime line: " +
        "Microsoft.ML.OnnxRuntime.DirectML ends at 1.24.4 and ONNX Runtime 1.25+ ships no DirectML provider. " +
        "Use ExecutionProvider.Auto (CUDA, CoreML or CPU for ONNX sessions; Vulkan for llama-server on AMD/Intel GPUs) " +
        "or ExecutionProvider.Cpu.";

    private const string DirectMLProviderName = "directml";
    private static int _directMLUnavailableTraced;

    /// <summary>
    /// Whether <paramref name="provider"/> can be served by this build.
    /// </summary>
    public static bool IsSupported(ExecutionProvider provider)
    {
#pragma warning disable CS0618 // the member is obsolete precisely because this method says so
        return provider != ExecutionProvider.DirectML;
#pragma warning restore CS0618
    }

    /// <summary>
    /// Throws <see cref="NotSupportedException"/> for a provider this build cannot serve.
    /// </summary>
    /// <param name="provider">The provider the caller asked for.</param>
    public static void ThrowIfUnsupported(ExecutionProvider provider)
    {
        if (!IsSupported(provider))
            throw new NotSupportedException(DirectMLUnavailableMessage);
    }

    /// <summary>
    /// Throws <see cref="NotSupportedException"/> for a provider name (as passed to the runtime
    /// manager, e.g. <c>"directml"</c>) this build cannot serve.
    /// </summary>
    /// <param name="providerName">The provider name the caller asked for.</param>
    public static void ThrowIfUnsupported(string providerName)
    {
        if (string.Equals(providerName, DirectMLProviderName, StringComparison.OrdinalIgnoreCase))
            throw new NotSupportedException(DirectMLUnavailableMessage);
    }

    /// <summary>
    /// Once per process: tells the operator that a Direct3D 12 capable GPU was found but ONNX
    /// sessions will run on CPU, because no GPU provider in <paramref name="fallbackChain"/> serves
    /// it. Called from the <see cref="ExecutionProvider.Auto"/> path, where both facts are in scope.
    /// </summary>
    internal static void TraceDirectMLUnavailableOnce(GpuInfo? gpu, IReadOnlyList<ExecutionProvider> fallbackChain)
    {
        if (gpu is null || !gpu.DirectMLSupported)
            return;
        if (fallbackChain.Any(p => p != ExecutionProvider.Cpu))
            return;
        if (Interlocked.Exchange(ref _directMLUnavailableTraced, 1) != 0)
            return;

        Trace.TraceInformation(
            $"[ExecutionProvider] A Direct3D 12 capable GPU was detected ({gpu.Vendor} {gpu.DeviceName ?? "n/a"}) but " +
            "the DirectML execution provider is unavailable on ONNX Runtime 1.25+ (package line ended at 1.24.4). " +
            "ONNX sessions run on CPU; llama-server (GGUF) paths still use Vulkan on this GPU.");
    }
}
