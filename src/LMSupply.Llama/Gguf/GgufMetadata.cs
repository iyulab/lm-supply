
namespace LMSupply.Llama.Gguf;

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
    /// Number of transformer layers (blocks); null when the file does not state a positive count.
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
    /// The decision head type of a native decision model (<c>&lt;arch&gt;.decision.type</c>, e.g. <c>laya</c>, <c>clef</c>).
    /// Null for any other model. llama-server serves such a model on <c>/v1/systemone</c>.
    /// </summary>
    public string? DecisionType { get; init; }

    /// <summary>
    /// The pooling the model was converted with (<c>&lt;arch&gt;.pooling_type</c>): 0 none, 1 mean, 2 CLS, 3 last, 4 rank.
    /// Null when the file does not say. An embedding server pools this way unless told otherwise.
    /// </summary>
    public int? PoolingType { get; init; }

    /// <summary>
    /// Raw metadata key-value pairs.
    /// </summary>
    public IReadOnlyDictionary<string, object?>? RawMetadata { get; init; }

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
    /// The parameter count the file states, or an estimate from its layer count, hidden size, feed-forward length and
    /// vocabulary when it states none; 0 when the shape is not in the file either.
    /// </summary>
    public long ParameterCountOrEstimate => EstimatedParameterCount ?? EstimateParameterCount();

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
