namespace LMSupply.Reranker;

/// <summary>
/// Configuration options for the Reranker.
/// </summary>
public sealed class RerankerOptions : LMSupplyOptionsBase
{
    /// <summary>
    /// Gets or sets the model identifier.
    /// <para>Supports:</para>
    /// <list type="bullet">
    /// <item>Preset aliases: "default", "quality", "fast", "multilingual"</item>
    /// <item>HuggingFace model IDs: "cross-encoder/ms-marco-MiniLM-L-6-v2"</item>
    /// <item>Local file paths: "/path/to/model.onnx"</item>
    /// </list>
    /// <para>Default: "default" (ms-marco-MiniLM-L-6-v2)</para>
    /// </summary>
    public string ModelId { get; set; } = "default";

    /// <summary>
    /// Gets or sets the maximum input sequence length.
    /// Longer inputs will be truncated.
    /// <para>Default: null (uses model's default, typically 512)</para>
    /// </summary>
    public int? MaxSequenceLength { get; set; }

    /// <summary>
    /// Gets or sets whether to disable automatic model download.
    /// When true, throws an exception if the model is not found locally.
    /// <para>Default: false</para>
    /// </summary>
    public bool DisableAutoDownload { get; set; } = false;

    /// <summary>
    /// Gets or sets the number of inference threads.
    /// <para>Default: null (uses Environment.ProcessorCount)</para>
    /// </summary>
    public int? ThreadCount { get; set; }

    /// <summary>
    /// Gets or sets the batch size for processing multiple documents.
    /// Larger values are faster but use more memory.
    /// <para>Default: 32</para>
    /// </summary>
    public int BatchSize { get; set; } = 32;

    /// <summary>
    /// Creates a copy of these options.
    /// </summary>
    public RerankerOptions Clone() => new()
    {
        ModelId = ModelId,
        MaxSequenceLength = MaxSequenceLength,
        CacheDirectory = CacheDirectory,
        Provider = Provider,
        DisableAutoDownload = DisableAutoDownload,
        ThreadCount = ThreadCount,
        BatchSize = BatchSize
    };
}
