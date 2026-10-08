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
    Cpu = 4,

    /// <summary>
    /// Intel OpenVINO execution provider on an Intel GPU (Windows x64 and Linux x64), for ONNX sessions only.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Delivered as an ONNX Runtime plugin: on first use the <c>Intel.ML.OnnxRuntime.EP.OpenVINO</c> package (about
    /// 120 MB with the OpenVINO runtime it carries) is downloaded into the runtime cache and registered with ONNX Runtime.
    /// Its version is pinned by this library, independently of the ONNX Runtime version. Compiled GPU kernels are kept in
    /// the runtime cache, so only the first load of a model pays for compilation.
    /// </para>
    /// <para>
    /// Never chosen by <see cref="Auto"/>: select it explicitly. A host with no Intel GPU, or whose GPU the plugin does
    /// not expose, gets a CPU session, as an explicit <see cref="Cuda"/> request does on a host without CUDA. Text
    /// generation (ONNX Runtime GenAI) and llama-server do not support it and refuse it.
    /// </para>
    /// </remarks>
    OpenVino = 5
}
