namespace LMSupply.Inference;

/// <summary>
/// Model weight precision/quantization levels.
/// Determines memory usage and inference speed vs. accuracy trade-offs.
/// </summary>
public enum ModelPrecision
{
    /// <summary>
    /// Full 32-bit floating point precision (4 bytes per parameter).
    /// Highest accuracy, highest memory usage.
    /// </summary>
    FP32,

    /// <summary>
    /// 16-bit floating point precision (2 bytes per parameter).
    /// Good balance of accuracy and memory, recommended for GPU inference.
    /// </summary>
    FP16,

    /// <summary>
    /// 8-bit integer quantization (1 byte per parameter).
    /// Lower memory usage with minimal accuracy loss for most models.
    /// </summary>
    INT8,

    /// <summary>
    /// 4-bit integer quantization (0.5 bytes per parameter).
    /// Lowest memory usage, primarily used for large language models.
    /// </summary>
    INT4,

    /// <summary>
    /// Automatically select the best precision based on hardware capabilities.
    /// </summary>
    Auto
}

/// <summary>
/// Intermediate computation precision for inference operations.
/// </summary>
public enum ComputePrecision
{
    /// <summary>
    /// Use 32-bit floating point for all computations.
    /// </summary>
    FP32,

    /// <summary>
    /// Use 16-bit floating point for computations where supported.
    /// </summary>
    FP16,

    /// <summary>
    /// Use mixed precision (FP16 compute with FP32 accumulation).
    /// Best balance of speed and accuracy on modern GPUs.
    /// </summary>
    Mixed
}

/// <summary>
/// Configuration for model precision and optimization settings.
/// </summary>
public sealed class PrecisionConfig
{
    /// <summary>
    /// Default configuration using automatic precision selection.
    /// </summary>
    public static PrecisionConfig Default { get; } = new();

    /// <summary>
    /// High-accuracy configuration prioritizing quality over speed.
    /// </summary>
    public static PrecisionConfig HighAccuracy { get; } = new()
    {
        ModelPrecision = ModelPrecision.FP32,
        ComputePrecision = ComputePrecision.FP32,
        EnableGraphOptimization = true,
        OptimizationLevel = OptimizationLevel.Extended
    };

    /// <summary>
    /// High-performance configuration optimized for speed on GPU.
    /// </summary>
    public static PrecisionConfig HighPerformance { get; } = new()
    {
        ModelPrecision = ModelPrecision.FP16,
        ComputePrecision = ComputePrecision.Mixed,
        EnableGraphOptimization = true,
        OptimizationLevel = OptimizationLevel.All
    };

    /// <summary>
    /// Memory-efficient configuration for constrained environments.
    /// </summary>
    public static PrecisionConfig LowMemory { get; } = new()
    {
        ModelPrecision = ModelPrecision.INT8,
        ComputePrecision = ComputePrecision.FP16,
        EnableGraphOptimization = true,
        OptimizationLevel = OptimizationLevel.All
    };

    /// <summary>
    /// The model weight precision.
    /// </summary>
    public ModelPrecision ModelPrecision { get; init; } = ModelPrecision.Auto;

    /// <summary>
    /// The intermediate computation precision.
    /// </summary>
    public ComputePrecision ComputePrecision { get; init; } = ComputePrecision.Mixed;

    /// <summary>
    /// Whether to enable graph optimization.
    /// </summary>
    public bool EnableGraphOptimization { get; init; } = true;

    /// <summary>
    /// The graph optimization level.
    /// </summary>
    public OptimizationLevel OptimizationLevel { get; init; } = OptimizationLevel.Basic;

    /// <summary>
    /// Gets the bytes per parameter for the specified precision.
    /// </summary>
    public static double GetBytesPerParameter(ModelPrecision precision) => precision switch
    {
        ModelPrecision.FP32 => 4.0,
        ModelPrecision.FP16 => 2.0,
        ModelPrecision.INT8 => 1.0,
        ModelPrecision.INT4 => 0.5,
        ModelPrecision.Auto => 2.0, // Default to FP16 estimate
        _ => 2.0
    };

    /// <summary>
    /// Estimates memory usage for model weights.
    /// </summary>
    /// <param name="parameterCount">Number of model parameters.</param>
    /// <returns>Estimated memory in bytes.</returns>
    public long EstimateModelMemory(long parameterCount)
    {
        return (long)(parameterCount * GetBytesPerParameter(ModelPrecision));
    }

    /// <summary>
    /// Gets the recommended precision based on available GPU memory.
    /// </summary>
    /// <param name="availableMemoryBytes">Available GPU/system memory in bytes.</param>
    /// <param name="parameterCount">Number of model parameters.</param>
    /// <returns>Recommended model precision.</returns>
    public static ModelPrecision GetRecommendedPrecision(long availableMemoryBytes, long parameterCount)
    {
        // Include overhead factor (1.5x for activations, gradients, etc.)
        const double overheadFactor = 1.5;

        var fp32Memory = (long)(parameterCount * GetBytesPerParameter(ModelPrecision.FP32) * overheadFactor);
        var fp16Memory = (long)(parameterCount * GetBytesPerParameter(ModelPrecision.FP16) * overheadFactor);
        var int8Memory = (long)(parameterCount * GetBytesPerParameter(ModelPrecision.INT8) * overheadFactor);
        var int4Memory = (long)(parameterCount * GetBytesPerParameter(ModelPrecision.INT4) * overheadFactor);

        if (availableMemoryBytes >= fp32Memory)
            return ModelPrecision.FP32;
        if (availableMemoryBytes >= fp16Memory)
            return ModelPrecision.FP16;
        if (availableMemoryBytes >= int8Memory)
            return ModelPrecision.INT8;
        if (availableMemoryBytes >= int4Memory)
            return ModelPrecision.INT4;

        return ModelPrecision.INT4; // Use lowest precision if memory is very constrained
    }
}

/// <summary>
/// Graph optimization levels for ONNX Runtime.
/// </summary>
public enum OptimizationLevel
{
    /// <summary>
    /// No optimization, useful for debugging.
    /// </summary>
    None,

    /// <summary>
    /// Basic optimizations like constant folding.
    /// </summary>
    Basic,

    /// <summary>
    /// Extended optimizations including node fusions.
    /// </summary>
    Extended,

    /// <summary>
    /// All available optimizations.
    /// </summary>
    All
}
