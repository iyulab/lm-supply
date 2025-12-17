namespace LMSupply.Core.Download;

/// <summary>
/// User preferences for model selection during discovery.
/// </summary>
public sealed class ModelPreferences
{
    /// <summary>
    /// Default preferences: balanced between quality and memory.
    /// </summary>
    public static ModelPreferences Default { get; } = new();

    /// <summary>
    /// Preferences optimized for low memory usage.
    /// </summary>
    public static ModelPreferences LowMemory { get; } = new()
    {
        PreferLowMemory = true,
        QuantizationPriority = [Quantization.Int4, Quantization.Int8, Quantization.Fp16, Quantization.Default]
    };

    /// <summary>
    /// Preferences optimized for quality/accuracy.
    /// </summary>
    public static ModelPreferences HighQuality { get; } = new()
    {
        PreferLowMemory = false,
        QuantizationPriority = [Quantization.Default, Quantization.Fp16, Quantization.Int8, Quantization.Int4]
    };

    /// <summary>
    /// Whether to prefer smaller/quantized models for lower memory usage.
    /// </summary>
    public bool PreferLowMemory { get; init; }

    /// <summary>
    /// Preferred execution provider (CPU, CUDA, etc.).
    /// </summary>
    public ExecutionProvider PreferredProvider { get; init; } = ExecutionProvider.Cpu;

    /// <summary>
    /// Priority order for quantization selection.
    /// First matching variant will be selected.
    /// </summary>
    public IReadOnlyList<Quantization> QuantizationPriority { get; init; } =
        [Quantization.Default, Quantization.Fp16, Quantization.Int8, Quantization.Int4];

    /// <summary>
    /// Specific subfolder to use. If set, overrides auto-detection.
    /// </summary>
    public string? PreferredSubfolder { get; init; }

    /// <summary>
    /// Specific ONNX file names to look for (e.g., "model.onnx", "encoder_model.onnx").
    /// If empty, all ONNX files in the selected subfolder are included.
    /// </summary>
    public IReadOnlyList<string> PreferredOnnxFiles { get; init; } = [];
}

/// <summary>
/// Model quantization levels.
/// </summary>
public enum Quantization
{
    /// <summary>Default/full precision (FP32).</summary>
    Default,

    /// <summary>Half precision (FP16).</summary>
    Fp16,

    /// <summary>8-bit integer quantization.</summary>
    Int8,

    /// <summary>4-bit integer quantization.</summary>
    Int4
}
