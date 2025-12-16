namespace LMSupply.Generator;

/// <summary>
/// Estimates memory requirements for LLM inference.
/// Uses formulas from ONNX Runtime GenAI documentation.
/// </summary>
public static class MemoryEstimator
{
    /// <summary>
    /// Estimates total memory required for model inference.
    /// </summary>
    /// <param name="config">Model configuration parameters.</param>
    /// <returns>Memory estimate in bytes.</returns>
    public static MemoryEstimate Calculate(ModelMemoryConfig config)
    {
        var modelMemory = CalculateModelMemory(config.ParameterCount, config.Quantization);
        var kvCacheMemory = CalculateKvCacheMemory(
            config.BatchSize,
            config.ContextLength,
            config.NumLayers,
            config.HiddenSize,
            config.KvCachePrecision);
        var overhead = (long)((modelMemory + kvCacheMemory) * 0.1); // 10% overhead

        return new MemoryEstimate
        {
            ModelMemoryBytes = modelMemory,
            KvCacheMemoryBytes = kvCacheMemory,
            OverheadBytes = overhead,
            TotalBytes = modelMemory + kvCacheMemory + overhead,
            Config = config
        };
    }

    /// <summary>
    /// Estimates if a model can fit in available memory.
    /// </summary>
    /// <param name="config">Model configuration.</param>
    /// <param name="availableMemoryBytes">Available memory in bytes.</param>
    /// <param name="safetyMargin">Safety margin (0.0-1.0). Defaults to 0.2 (20%).</param>
    /// <returns>True if model fits within memory constraints.</returns>
    public static bool CanFitInMemory(
        ModelMemoryConfig config,
        long availableMemoryBytes,
        double safetyMargin = 0.2)
    {
        var estimate = Calculate(config);
        var requiredWithMargin = (long)(estimate.TotalBytes * (1 + safetyMargin));
        return requiredWithMargin <= availableMemoryBytes;
    }

    /// <summary>
    /// Calculates model weights memory based on parameter count and quantization.
    /// </summary>
    /// <remarks>
    /// Memory per parameter:
    /// - FP32: 4 bytes
    /// - FP16: 2 bytes
    /// - INT8: 1 byte
    /// - INT4: 0.5 bytes
    /// </remarks>
    public static long CalculateModelMemory(long parameterCount, Quantization quantization)
    {
        var bytesPerParam = quantization switch
        {
            Quantization.FP32 => 4.0,
            Quantization.FP16 => 2.0,
            Quantization.INT8 => 1.0,
            Quantization.INT4 => 0.5,
            _ => 2.0 // Default to FP16
        };

        return (long)(parameterCount * bytesPerParam);
    }

    /// <summary>
    /// Calculates KV cache memory for transformer models.
    /// </summary>
    /// <remarks>
    /// Formula: batch_size × seq_len × 2 × n_layers × d_model × bytes_per_value
    /// The '2' represents Key and Value tensors.
    /// </remarks>
    public static long CalculateKvCacheMemory(
        int batchSize,
        int contextLength,
        int numLayers,
        int hiddenSize,
        KvCachePrecision precision = KvCachePrecision.FP16)
    {
        var bytesPerValue = precision switch
        {
            KvCachePrecision.FP32 => 4,
            KvCachePrecision.FP16 => 2,
            KvCachePrecision.INT8 => 1,
            _ => 2
        };

        // KV cache = batch × seq_len × 2 (K+V) × layers × hidden_size × bytes
        return (long)batchSize * contextLength * 2 * numLayers * hiddenSize * bytesPerValue;
    }

    /// <summary>
    /// Gets default model configuration for common models.
    /// </summary>
    public static ModelMemoryConfig GetDefaultConfig(string modelFamily)
    {
        return modelFamily.ToLowerInvariant() switch
        {
            "phi-3.5-mini" or "phi-3-mini" => new ModelMemoryConfig
            {
                ParameterCount = 3_800_000_000,
                NumLayers = 32,
                HiddenSize = 3072,
                ContextLength = 4096,
                Quantization = Quantization.INT4
            },
            "phi-4" => new ModelMemoryConfig
            {
                ParameterCount = 14_000_000_000,
                NumLayers = 40,
                HiddenSize = 5120,
                ContextLength = 8192,
                Quantization = Quantization.INT4
            },
            "llama-3.2-1b" => new ModelMemoryConfig
            {
                ParameterCount = 1_000_000_000,
                NumLayers = 16,
                HiddenSize = 2048,
                ContextLength = 4096,
                Quantization = Quantization.INT4
            },
            "llama-3.2-3b" => new ModelMemoryConfig
            {
                ParameterCount = 3_000_000_000,
                NumLayers = 28,
                HiddenSize = 3072,
                ContextLength = 4096,
                Quantization = Quantization.INT4
            },
            _ => new ModelMemoryConfig
            {
                ParameterCount = 3_000_000_000,
                NumLayers = 32,
                HiddenSize = 2560,
                ContextLength = 4096,
                Quantization = Quantization.INT4
            }
        };
    }
}

/// <summary>
/// Model memory configuration parameters.
/// </summary>
public sealed record ModelMemoryConfig
{
    /// <summary>
    /// Total number of model parameters.
    /// </summary>
    public required long ParameterCount { get; init; }

    /// <summary>
    /// Number of transformer layers.
    /// </summary>
    public required int NumLayers { get; init; }

    /// <summary>
    /// Hidden dimension size (d_model).
    /// </summary>
    public required int HiddenSize { get; init; }

    /// <summary>
    /// Maximum context length (sequence length).
    /// </summary>
    public int ContextLength { get; init; } = 4096;

    /// <summary>
    /// Batch size for inference.
    /// </summary>
    public int BatchSize { get; init; } = 1;

    /// <summary>
    /// Model quantization level.
    /// </summary>
    public Quantization Quantization { get; init; } = Quantization.INT4;

    /// <summary>
    /// KV cache precision.
    /// </summary>
    public KvCachePrecision KvCachePrecision { get; init; } = KvCachePrecision.FP16;
}

/// <summary>
/// Memory estimation result.
/// </summary>
public sealed record MemoryEstimate
{
    /// <summary>
    /// Memory required for model weights.
    /// </summary>
    public required long ModelMemoryBytes { get; init; }

    /// <summary>
    /// Memory required for KV cache.
    /// </summary>
    public required long KvCacheMemoryBytes { get; init; }

    /// <summary>
    /// Estimated overhead (activations, temporary buffers).
    /// </summary>
    public required long OverheadBytes { get; init; }

    /// <summary>
    /// Total estimated memory requirement.
    /// </summary>
    public required long TotalBytes { get; init; }

    /// <summary>
    /// Configuration used for this estimate.
    /// </summary>
    public required ModelMemoryConfig Config { get; init; }

    /// <summary>
    /// Gets a human-readable summary.
    /// </summary>
    public string GetSummary()
    {
        var modelMB = ModelMemoryBytes / (1024.0 * 1024);
        var kvMB = KvCacheMemoryBytes / (1024.0 * 1024);
        var totalGB = TotalBytes / (1024.0 * 1024 * 1024);

        return $"""
            Model Weights: {modelMB:F0}MB ({Config.Quantization})
            KV Cache: {kvMB:F0}MB (ctx={Config.ContextLength}, batch={Config.BatchSize})
            Overhead: {OverheadBytes / (1024.0 * 1024):F0}MB
            Total: {totalGB:F2}GB
            """;
    }
}

/// <summary>
/// Model quantization levels.
/// </summary>
public enum Quantization
{
    /// <summary>32-bit floating point (4 bytes per parameter).</summary>
    FP32,
    /// <summary>16-bit floating point (2 bytes per parameter).</summary>
    FP16,
    /// <summary>8-bit integer (1 byte per parameter).</summary>
    INT8,
    /// <summary>4-bit integer (0.5 bytes per parameter).</summary>
    INT4
}

/// <summary>
/// KV cache precision levels.
/// </summary>
public enum KvCachePrecision
{
    /// <summary>32-bit floating point KV cache.</summary>
    FP32,
    /// <summary>16-bit floating point KV cache (default).</summary>
    FP16,
    /// <summary>8-bit integer KV cache (quantized).</summary>
    INT8
}
