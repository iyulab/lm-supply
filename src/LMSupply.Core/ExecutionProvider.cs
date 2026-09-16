namespace LMSupply;

/// <summary>
/// Specifies the execution provider for ONNX Runtime inference.
/// </summary>
/// <remarks>
/// Members carry explicit values: a member can leave this enum (as <see cref="DirectML"/> will), and
/// a setting bound by number must keep meaning the same provider when that happens.
/// </remarks>
public enum ExecutionProvider
{
    /// <summary>
    /// Automatically select the best available provider.
    /// Tries GPU providers first, falls back to CPU.
    /// </summary>
    Auto = 0,

    /// <summary>
    /// NVIDIA CUDA execution provider.
    /// Requires Microsoft.ML.OnnxRuntime.Gpu package.
    /// </summary>
    Cuda = 1,

    /// <summary>
    /// The DirectML execution provider is no longer available (0.67.0). ONNX Runtime 1.25 and later
    /// ship no DirectML provider and the Microsoft.ML.OnnxRuntime.DirectML package line ends at
    /// 1.24.4, so no build of this library can honour the request. An explicit request throws
    /// <see cref="NotSupportedException"/> on every path (ONNX session, GenAI, llama-server); use
    /// <see cref="Auto"/> (which picks CUDA, CoreML or CPU for ONNX sessions and Vulkan for
    /// llama-server on AMD/Intel GPUs) or <see cref="Cpu"/>. See <see cref="ExecutionProviderSupport"/>.
    /// </summary>
    [Obsolete(ExecutionProviderSupport.DirectMLUnavailableMessage)]
    DirectML = 2,

    /// <summary>
    /// Apple CoreML execution provider for macOS/iOS.
    /// </summary>
    CoreML = 3,

    /// <summary>
    /// CPU execution provider.
    /// Always available, no additional packages required.
    /// </summary>
    Cpu = 4
}
