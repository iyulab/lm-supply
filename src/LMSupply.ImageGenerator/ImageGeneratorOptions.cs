using LMSupply.Core;

namespace LMSupply.ImageGenerator;

/// <summary>
/// Options for configuring the image generator model.
/// </summary>
public sealed class ImageGeneratorOptions
{
    /// <summary>
    /// Directory for caching downloaded models.
    /// Default: ~/.cache/huggingface/hub/
    /// </summary>
    public string? CacheDirectory { get; set; }

    /// <summary>
    /// Execution provider for inference.
    /// Default: ExecutionProvider.Auto (auto-detect best available)
    /// </summary>
    public ExecutionProvider Provider { get; set; } = ExecutionProvider.Auto;

    /// <summary>
    /// Number of threads to use for CPU inference.
    /// Default: null (use system default)
    /// </summary>
    public int? ThreadCount { get; set; }

    /// <summary>
    /// Whether to disable automatic model downloading.
    /// This covers model files only. The native ONNX Runtime an ONNX model runs on is provisioned separately; to keep
    /// it off the network too, see <see cref="LMSupply.Runtime.RuntimeManager.Configure"/>.
    /// Default: false
    /// </summary>
    public bool DisableAutoDownload { get; set; }

    /// <summary>
    /// Whether to use FP16 precision for reduced memory usage.
    /// Default: true (recommended for GPU inference)
    /// </summary>
    public bool UseFp16 { get; set; } = true;

    /// <summary>
    /// Device ID for GPU inference (0-based index).
    /// Default: 0
    /// </summary>
    public int DeviceId { get; set; }

    /// <summary>
    /// ONNX Runtime log severity level.
    /// Default: <see cref="OrtLogLevel.Error"/> (suppresses provider fallback warnings)
    /// </summary>
    public OrtLogLevel LogLevel { get; set; } = OrtLogLevel.Error;

    /// <summary>A copy of these options, for an operation that must not modify the caller's instance.</summary>
    public ImageGeneratorOptions Clone() => (ImageGeneratorOptions)MemberwiseClone();
}
