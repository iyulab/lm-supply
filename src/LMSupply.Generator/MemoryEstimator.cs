using LMSupply.Generator.Internal.Llama;

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
    /// - Quant8: 1 byte
    /// - Quant4: 0.5 bytes
    /// </remarks>
    public static long CalculateModelMemory(long parameterCount, Quantization quantization)
    {
        var bytesPerParam = quantization switch
        {
            Quantization.FP32 => 4.0,
            Quantization.FP16 => 2.0,
            Quantization.Quant8 => 1.0,
            Quantization.Quant4 => 0.5,
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
            KvCachePrecision.Quant8 => 1,
            _ => 2
        };

        // KV cache = batch × seq_len × 2 (K+V) × layers × hidden_size × bytes
        return (long)batchSize * contextLength * 2 * numLayers * hiddenSize * bytesPerValue;
    }

    /// <summary>
    /// Estimates resource requirements for a GGUF model file.
    /// </summary>
    /// <param name="modelFileSizeBytes">Size of the GGUF file in bytes.</param>
    /// <param name="contextLength">Desired context length.</param>
    /// <param name="availableVramBytes">Available GPU VRAM in bytes (null to skip GPU check).</param>
    /// <param name="availableRamBytes">Available system RAM in bytes (null to skip RAM check).</param>
    /// <param name="estimatedLayers">Estimated number of layers (default: 32 for 7B-class models).</param>
    /// <returns>Resource estimation with GPU/RAM breakdown.</returns>
    public static ResourceEstimate EstimateForGguf(
        long modelFileSizeBytes,
        int contextLength = 4096,
        long? availableVramBytes = null,
        long? availableRamBytes = null,
        int estimatedLayers = 32)
    {
        // Estimate KV cache based on model size and context length
        // Rough formula: context_length * hidden_size * 2 (K+V) * layers * 2 bytes (FP16)
        // Estimate hidden_size from model size: smaller models ~2048, larger ~4096
        var estimatedHiddenSize = modelFileSizeBytes switch
        {
            < 2L * 1024 * 1024 * 1024 => 2048,   // <2GB: small model
            < 5L * 1024 * 1024 * 1024 => 3072,   // 2-5GB: medium model
            < 10L * 1024 * 1024 * 1024 => 4096,  // 5-10GB: large model
            _ => 5120                             // >10GB: very large model
        };

        var kvCacheMemory = (long)contextLength * estimatedHiddenSize * 2 * estimatedLayers * 2;
        return EstimateForGgufWithKvCache(modelFileSizeBytes, kvCacheMemory, availableVramBytes, availableRamBytes, estimatedLayers);
    }

    /// <summary>
    /// Same estimate as <see cref="EstimateForGguf(long, int, long?, long?, int)"/> with a known KV cache size
    /// (e.g. from the file's attention metadata) instead of the file-size heuristic.
    /// </summary>
    internal static ResourceEstimate EstimateForGgufWithKvCache(
        long modelFileSizeBytes,
        long kvCacheMemory,
        long? availableVramBytes,
        long? availableRamBytes,
        int estimatedLayers)
    {
        // GGUF file size is roughly equal to model weights in memory
        // Add ~10% overhead for model loading structures
        var modelMemory = (long)(modelFileSizeBytes * 1.1);

        // Total memory needed
        var totalMemory = modelMemory + kvCacheMemory;

        // Reserve 2GB for system overhead when calculating VRAM fit
        const long vramOverhead = 2L * 1024 * 1024 * 1024;
        var canFitInVram = availableVramBytes.HasValue &&
            totalMemory <= (availableVramBytes.Value - vramOverhead);

        // Reserve 4GB for system when calculating RAM fit
        const long ramOverhead = 4L * 1024 * 1024 * 1024;
        var canFitInRam = availableRamBytes.HasValue &&
            totalMemory <= (availableRamBytes.Value - ramOverhead);

        // Calculate recommended GPU layers based on available VRAM
        int recommendedLayers;
        long estimatedVram;
        long estimatedRam;

        if (canFitInVram)
        {
            // All layers on GPU
            recommendedLayers = estimatedLayers;
            estimatedVram = totalMemory;
            estimatedRam = 0;
        }
        else if (availableVramBytes.HasValue)
        {
            // Partial GPU offload
            var usableVram = Math.Max(0, availableVramBytes.Value - vramOverhead);
            var layerSize = modelMemory / estimatedLayers;
            recommendedLayers = (int)(usableVram / layerSize);
            recommendedLayers = Math.Clamp(recommendedLayers, 0, estimatedLayers);

            estimatedVram = recommendedLayers * layerSize;
            estimatedRam = totalMemory - estimatedVram + kvCacheMemory;
        }
        else
        {
            // CPU only
            recommendedLayers = 0;
            estimatedVram = 0;
            estimatedRam = totalMemory;
        }

        return new ResourceEstimate
        {
            EstimatedVramBytes = estimatedVram,
            EstimatedRamBytes = estimatedRam,
            RecommendedGpuLayers = recommendedLayers,
            TotalLayers = estimatedLayers,
            CanFitInVram = canFitInVram,
            CanFitInRam = canFitInRam || canFitInVram,
            Confidence = EstimationConfidence.Low, // File-size based estimation
            AvailableVramBytes = availableVramBytes,
            AvailableRamBytes = availableRamBytes,
            ModelFileSizeBytes = modelFileSizeBytes
        };
    }

    /// <summary>
    /// Estimates resource requirements by reading GGUF file metadata.
    /// Provides high-confidence estimation using actual model parameters.
    /// </summary>
    /// <param name="ggufFilePath">Path to the GGUF model file.</param>
    /// <param name="contextLength">Desired context length (uses model default if null).</param>
    /// <param name="availableVramBytes">Available GPU VRAM in bytes.</param>
    /// <param name="availableRamBytes">Available system RAM in bytes.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>High-confidence resource estimation, or null if metadata cannot be read.</returns>
    public static async Task<ResourceEstimate?> EstimateFromGgufFileAsync(
        string ggufFilePath,
        int? contextLength = null,
        long? availableVramBytes = null,
        long? availableRamBytes = null,
        CancellationToken cancellationToken = default)
    {
        var metadata = await GgufMetadataReader.ReadAsync(ggufFilePath, false, cancellationToken);
        if (metadata == null)
            return null;

        var config = metadata.ToMemoryConfig(contextLength);
        if (config == null)
        {
            // Fall back to file-size based estimation if metadata is incomplete
            var fileSize = new FileInfo(ggufFilePath).Length;
            return EstimateForGguf(
                fileSize,
                contextLength ?? 4096,
                availableVramBytes,
                availableRamBytes,
                metadata.LayerCount ?? 32);
        }

        var result = EstimateWithConfig(config, availableVramBytes, availableRamBytes);

        // Upgrade confidence to High since we used actual metadata
        return result with
        {
            Confidence = EstimationConfidence.High,
            TotalLayers = metadata.LayerCount ?? result.TotalLayers,
            ModelFileSizeBytes = new FileInfo(ggufFilePath).Length
        };
    }

    /// <summary>
    /// Reads GGUF file metadata and returns it directly.
    /// </summary>
    /// <param name="ggufFilePath">Path to the GGUF model file.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>GGUF metadata, or null if the file is not a valid GGUF.</returns>
    public static Task<GgufMetadata?> ReadGgufMetadataAsync(
        string ggufFilePath,
        CancellationToken cancellationToken = default)
    {
        return GgufMetadataReader.ReadAsync(ggufFilePath, false, cancellationToken);
    }

    /// <summary>
    /// Estimates resource requirements using detailed model configuration.
    /// </summary>
    public static ResourceEstimate EstimateWithConfig(
        ModelMemoryConfig config,
        long? availableVramBytes = null,
        long? availableRamBytes = null)
    {
        var estimate = Calculate(config);

        const long vramOverhead = 2L * 1024 * 1024 * 1024;
        const long ramOverhead = 4L * 1024 * 1024 * 1024;

        var canFitInVram = availableVramBytes.HasValue &&
            estimate.TotalBytes <= (availableVramBytes.Value - vramOverhead);

        var canFitInRam = availableRamBytes.HasValue &&
            estimate.TotalBytes <= (availableRamBytes.Value - ramOverhead);

        int recommendedLayers;
        long estimatedVram;
        long estimatedRam;

        if (canFitInVram)
        {
            recommendedLayers = config.NumLayers;
            estimatedVram = estimate.TotalBytes;
            estimatedRam = 0;
        }
        else if (availableVramBytes.HasValue)
        {
            var usableVram = Math.Max(0, availableVramBytes.Value - vramOverhead);
            var layerSize = estimate.ModelMemoryBytes / config.NumLayers;
            recommendedLayers = (int)(usableVram / layerSize);
            recommendedLayers = Math.Clamp(recommendedLayers, 0, config.NumLayers);

            estimatedVram = recommendedLayers * layerSize;
            estimatedRam = estimate.TotalBytes - estimatedVram + estimate.KvCacheMemoryBytes;
        }
        else
        {
            recommendedLayers = 0;
            estimatedVram = 0;
            estimatedRam = estimate.TotalBytes;
        }

        return new ResourceEstimate
        {
            EstimatedVramBytes = estimatedVram,
            EstimatedRamBytes = estimatedRam,
            RecommendedGpuLayers = recommendedLayers,
            TotalLayers = config.NumLayers,
            CanFitInVram = canFitInVram,
            CanFitInRam = canFitInRam || canFitInVram,
            Confidence = EstimationConfidence.Standard,
            AvailableVramBytes = availableVramBytes,
            AvailableRamBytes = availableRamBytes
        };
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
                Quantization = Quantization.Quant4
            },
            "phi-4" => new ModelMemoryConfig
            {
                ParameterCount = 14_000_000_000,
                NumLayers = 40,
                HiddenSize = 5120,
                ContextLength = 8192,
                Quantization = Quantization.Quant4
            },
            "llama-3.2-1b" => new ModelMemoryConfig
            {
                ParameterCount = 1_000_000_000,
                NumLayers = 16,
                HiddenSize = 2048,
                ContextLength = 4096,
                Quantization = Quantization.Quant4
            },
            "llama-3.2-3b" => new ModelMemoryConfig
            {
                ParameterCount = 3_000_000_000,
                NumLayers = 28,
                HiddenSize = 3072,
                ContextLength = 4096,
                Quantization = Quantization.Quant4
            },
            _ => new ModelMemoryConfig
            {
                ParameterCount = 3_000_000_000,
                NumLayers = 32,
                HiddenSize = 2560,
                ContextLength = 4096,
                Quantization = Quantization.Quant4
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
    public Quantization Quantization { get; init; } = Quantization.Quant4;

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
/// Extended resource estimation result for GGUF models with GPU/RAM breakdown.
/// </summary>
public sealed record ResourceEstimate
{
    /// <summary>
    /// Estimated GPU VRAM requirement in bytes.
    /// </summary>
    public required long EstimatedVramBytes { get; init; }

    /// <summary>
    /// Estimated system RAM requirement in bytes.
    /// </summary>
    public required long EstimatedRamBytes { get; init; }

    /// <summary>
    /// Total memory requirement (VRAM + RAM).
    /// </summary>
    public long TotalMemoryBytes => EstimatedVramBytes + EstimatedRamBytes;

    /// <summary>
    /// Recommended number of layers to offload to GPU.
    /// </summary>
    public int RecommendedGpuLayers { get; init; }

    /// <summary>
    /// Total number of layers in the model.
    /// </summary>
    public int TotalLayers { get; init; }

    /// <summary>
    /// Whether the model can fit entirely in GPU VRAM.
    /// </summary>
    public bool CanFitInVram { get; init; }

    /// <summary>
    /// Whether the model can fit in available system memory.
    /// </summary>
    public bool CanFitInRam { get; init; }

    /// <summary>
    /// Estimation confidence level.
    /// </summary>
    public EstimationConfidence Confidence { get; init; } = EstimationConfidence.Standard;

    /// <summary>
    /// Available GPU VRAM used for calculation.
    /// </summary>
    public long? AvailableVramBytes { get; init; }

    /// <summary>
    /// Available system RAM used for calculation.
    /// </summary>
    public long? AvailableRamBytes { get; init; }

    /// <summary>
    /// Model file size in bytes.
    /// </summary>
    public long? ModelFileSizeBytes { get; init; }

    /// <summary>
    /// Gets the GPU offload ratio (0.0-1.0) based on recommended layers.
    /// </summary>
    public float GpuOffloadRatio => TotalLayers > 0
        ? (float)RecommendedGpuLayers / TotalLayers
        : 0f;

    /// <summary>
    /// Gets a human-readable summary.
    /// </summary>
    public string GetSummary()
    {
        var vramGB = EstimatedVramBytes / (1024.0 * 1024 * 1024);
        var ramGB = EstimatedRamBytes / (1024.0 * 1024 * 1024);

        var status = (CanFitInVram, CanFitInRam) switch
        {
            (true, _) => "✓ Full GPU",
            (false, true) => "⚠ Partial GPU",
            (false, false) => "✗ Insufficient Memory"
        };

        return $"""
            GPU VRAM: {vramGB:F2} GB
            System RAM: {ramGB:F2} GB
            GPU Layers: {RecommendedGpuLayers}/{TotalLayers} ({GpuOffloadRatio:P0})
            Status: {status}
            Confidence: {Confidence}
            """;
    }
}

/// <summary>
/// Confidence level for resource estimation.
/// </summary>
public enum EstimationConfidence
{
    /// <summary>
    /// Low confidence - file size based estimation only.
    /// </summary>
    Low,

    /// <summary>
    /// Standard confidence - using known model architecture parameters.
    /// </summary>
    Standard,

    /// <summary>
    /// High confidence - using exact model metadata.
    /// </summary>
    High
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
    /// <summary>8-bit quantization (1 byte per parameter).</summary>
    Quant8,
    /// <summary>4-bit quantization (0.5 bytes per parameter).</summary>
    Quant4
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
    /// <summary>8-bit quantized KV cache.</summary>
    Quant8
}
