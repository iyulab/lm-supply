namespace LMSupply;

/// <summary>
/// Specifies the execution provider for ONNX Runtime inference.
/// </summary>
/// <remarks>
/// Members carry explicit values: a member can leave this enum, and a setting bound by number must keep meaning the
/// same provider when that happens. Value 2 was DirectML, removed in 0.111.0 (the provider left ONNX Runtime 1.25);
/// it is never reused, and <see cref="ExecutionProviderSupport"/> still refuses it with the reason.
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
    /// Apple CoreML execution provider for macOS/iOS.
    /// </summary>
    CoreML = 3,

    /// <summary>
    /// CPU execution provider.
    /// Always available, no additional packages required.
    /// </summary>
    Cpu = 4
}
