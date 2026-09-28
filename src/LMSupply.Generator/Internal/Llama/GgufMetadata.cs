namespace LMSupply.Generator.Internal.Llama;

/// <summary>
/// Metadata extracted from a GGUF model file.
/// </summary>
public sealed record GgufMetadata
{
    /// <summary>
    /// GGUF format version.
    /// </summary>
    public uint Version { get; init; }

    /// <summary>
    /// Model architecture (e.g., "llama", "gemma", "phi", "qwen2").
    /// </summary>
    public string? Architecture { get; init; }

    /// <summary>
    /// Model name from metadata.
    /// </summary>
    public string? Name { get; init; }

    /// <summary>
    /// Number of transformer layers (blocks).
    /// </summary>
    public int? LayerCount { get; init; }

    /// <summary>
    /// Maximum context length.
    /// </summary>
    public int? ContextLength { get; init; }

    /// <summary>
    /// Embedding/hidden dimension size.
    /// </summary>
    public int? EmbeddingLength { get; init; }

    /// <summary>
    /// Number of attention heads.
    /// </summary>
    public int? HeadCount { get; init; }

    /// <summary>
    /// Number of key-value heads (for GQA).
    /// </summary>
    public int? HeadCountKv { get; init; }

    /// <summary>
    /// Vocabulary size.
    /// </summary>
    public int? VocabSize { get; init; }

    /// <summary>
    /// Feed-forward network intermediate size.
    /// </summary>
    public int? FeedForwardLength { get; init; }

    /// <summary>
    /// RoPE frequency base.
    /// </summary>
    public float? RopeFreqBase { get; init; }

    /// <summary>
    /// Quantization type of the general model weights.
    /// </summary>
    public string? QuantizationType { get; init; }

    /// <summary>
    /// File type enumeration from GGUF (indicates quantization).
    /// </summary>
    public int? FileType { get; init; }

    /// <summary>
    /// Total tensor count in the file.
    /// </summary>
    public long TensorCount { get; init; }

    /// <summary>
    /// Estimated parameter count based on metadata.
    /// </summary>
    public long? EstimatedParameterCount { get; init; }

    /// <summary>
    /// Total number of MoE experts. Null or 0 for dense (non-MoE) models.
    /// Present in GGUF for architectures such as Mixtral, DeepSeek-V2, Gemma 4.
    /// </summary>
    public int? ExpertCount { get; init; }

    /// <summary>
    /// Number of MoE experts activated per token (top-K routing).
    /// Null for dense models.
    /// </summary>
    public int? ExpertUsedCount { get; init; }

    /// <summary>
    /// Per-layer KV head count, when the file stores <c>attention.head_count_kv</c> as an array
    /// (layers without attention carry 0). Null when it is a single value (see <see cref="HeadCountKv"/>,
    /// which holds the largest per-layer count in the array case).
    /// </summary>
    public IReadOnlyList<int>? HeadCountKvPerLayer { get; init; }

    /// <summary>Per-head key dimension (<c>attention.key_length</c>). Null → embedding length / head count.</summary>
    public int? KeyLength { get; init; }

    /// <summary>Per-head value dimension (<c>attention.value_length</c>). Null → the key dimension.</summary>
    public int? ValueLength { get; init; }

    /// <summary>Per-head key dimension of sliding-window layers, when it differs from the full-attention one.</summary>
    public int? KeyLengthSwa { get; init; }

    /// <summary>Per-head value dimension of sliding-window layers, when it differs from the full-attention one.</summary>
    public int? ValueLengthSwa { get; init; }

    /// <summary>Sliding-window attention width in tokens (<c>attention.sliding_window</c>).</summary>
    public int? SlidingWindow { get; init; }

    /// <summary>Per-layer flag: true for a sliding-window layer (<c>attention.sliding_window_pattern</c>).</summary>
    public IReadOnlyList<bool>? SlidingWindowPattern { get; init; }

    /// <summary>
    /// Number of trailing layers that reuse an earlier layer's KV cache instead of keeping their own
    /// (<c>attention.shared_kv_layers</c>).
    /// </summary>
    public int? SharedKvLayers { get; init; }

    /// <summary>
    /// For hybrid attention/recurrent models: every N-th layer uses full attention and keeps a KV cache;
    /// the others are recurrent (<c>full_attention_interval</c>).
    /// </summary>
    public int? FullAttentionInterval { get; init; }

    /// <summary>
    /// Raw metadata key-value pairs.
    /// </summary>
    public IReadOnlyDictionary<string, object?>? RawMetadata { get; init; }

    /// <summary>
    /// Creates a ModelMemoryConfig from this metadata.
    /// </summary>
    /// <param name="contextLength">Override context length (uses metadata value if null).</param>
    /// <returns>Memory configuration, or null if insufficient metadata.</returns>
    public ModelMemoryConfig? ToMemoryConfig(int? contextLength = null)
    {
        if (!LayerCount.HasValue || !EmbeddingLength.HasValue)
            return null;

        // Estimate parameter count if not available
        var paramCount = EstimatedParameterCount ?? EstimateParameterCount();

        return new ModelMemoryConfig
        {
            ParameterCount = paramCount,
            NumLayers = LayerCount.Value,
            HiddenSize = EmbeddingLength.Value,
            ContextLength = contextLength ?? ContextLength ?? 4096,
            Quantization = InferQuantization()
        };
    }

    /// <summary>
    /// Estimates parameter count from architecture dimensions.
    /// </summary>
    private long EstimateParameterCount()
    {
        if (!LayerCount.HasValue || !EmbeddingLength.HasValue)
            return 0;

        var layers = LayerCount.Value;
        var hidden = EmbeddingLength.Value;
        var ffn = FeedForwardLength ?? hidden * 4; // Default FFN ratio
        var vocab = VocabSize ?? 32000; // Default vocab size

        // Rough transformer parameter estimation:
        // - Embedding: vocab * hidden
        // - Per layer: 4 * hidden^2 (attention) + 3 * hidden * ffn (FFN)
        // - Output: vocab * hidden
        var embedding = (long)vocab * hidden;
        var perLayer = 4L * hidden * hidden + 3L * hidden * ffn;
        var output = (long)vocab * hidden;

        return embedding + (perLayer * layers) + output;
    }

    /// <summary>
    /// Infers quantization type from file type or quantization string.
    /// </summary>
    private Quantization InferQuantization()
    {
        // Infer from quantization type string
        if (!string.IsNullOrEmpty(QuantizationType))
        {
            var qt = QuantizationType.ToUpperInvariant();
            if (qt.Contains("Q4") || qt.Contains("INT4"))
                return Quantization.Quant4;
            if (qt.Contains("Q8") || qt.Contains("INT8"))
                return Quantization.Quant8;
            if (qt.Contains("F16") || qt.Contains("FP16"))
                return Quantization.FP16;
            if (qt.Contains("F32") || qt.Contains("FP32"))
                return Quantization.FP32;
        }

        // Infer from GGUF file type enum
        if (FileType.HasValue)
        {
            // GGUF file types (from llama.cpp/ggml.h):
            // 0: F32, 1: F16, 2-6: Q4_X, 7-8: Q8_X, etc.
            return FileType.Value switch
            {
                0 => Quantization.FP32,
                1 => Quantization.FP16,
                >= 2 and <= 6 => Quantization.Quant4,
                >= 7 and <= 8 => Quantization.Quant8,
                _ => Quantization.Quant4 // Default to Quant4 for quantized models
            };
        }

        return Quantization.Quant4; // Most GGUF files are quantized
    }

    /// <summary>
    /// Gets a human-readable summary of the metadata.
    /// </summary>
    public string GetSummary()
    {
        var parts = new List<string>();

        if (!string.IsNullOrEmpty(Name))
            parts.Add($"Name: {Name}");

        if (!string.IsNullOrEmpty(Architecture))
            parts.Add($"Architecture: {Architecture}");

        if (LayerCount.HasValue)
            parts.Add($"Layers: {LayerCount}");

        if (EmbeddingLength.HasValue)
            parts.Add($"Hidden Size: {EmbeddingLength}");

        if (ContextLength.HasValue)
            parts.Add($"Context: {ContextLength}");

        if (HeadCount.HasValue)
            parts.Add($"Heads: {HeadCount}" + (HeadCountKv.HasValue ? $" (KV: {HeadCountKv})" : ""));

        if (!string.IsNullOrEmpty(QuantizationType))
            parts.Add($"Quantization: {QuantizationType}");

        if (ExpertCount > 1)
            parts.Add($"Experts: {ExpertCount}" + (ExpertUsedCount.HasValue ? $" (top-{ExpertUsedCount})" : ""));

        if (EstimatedParameterCount.HasValue || (LayerCount.HasValue && EmbeddingLength.HasValue))
        {
            var paramB = (EstimatedParameterCount ?? EstimateParameterCount()) / 1_000_000_000.0;
            parts.Add($"Parameters: ~{paramB:F1}B");
        }

        return string.Join(", ", parts);
    }
}
