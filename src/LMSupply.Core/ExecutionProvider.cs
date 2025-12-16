namespace LMSupply;

/// <summary>
/// Specifies the execution provider for ONNX Runtime inference.
/// </summary>
public enum ExecutionProvider
{
    /// <summary>
    /// Automatically select the best available provider.
    /// Tries GPU providers first, falls back to CPU.
    /// </summary>
    Auto,

    /// <summary>
    /// NVIDIA CUDA execution provider.
    /// Requires Microsoft.ML.OnnxRuntime.Gpu package.
    /// </summary>
    Cuda,

    /// <summary>
    /// DirectML execution provider for Windows.
    /// Supports AMD, Intel, and NVIDIA GPUs.
    /// Requires Microsoft.ML.OnnxRuntime.DirectML package.
    /// </summary>
    DirectML,

    /// <summary>
    /// Apple CoreML execution provider for macOS/iOS.
    /// </summary>
    CoreML,

    /// <summary>
    /// CPU execution provider.
    /// Always available, no additional packages required.
    /// </summary>
    Cpu
}
