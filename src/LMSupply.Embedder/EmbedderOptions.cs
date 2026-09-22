using LMSupply.Llama.Server;

namespace LMSupply.Embedder;

/// <summary>
/// Configuration options for the embedding model.
/// </summary>
public sealed class EmbedderOptions : LMSupplyOptionsBase
{
    /// <summary>The value <see cref="MaxSequenceLength"/> starts with.</summary>
    public const int DefaultMaxSequenceLength = 512;

    /// <summary>
    /// The sequence length to tokenize with, in tokens. <see langword="null"/> — the default — lets the
    /// model decide: its own <c>sentence_bert_config.json</c> <c>max_seq_length</c> first, then the
    /// catalog entry for a known alias, then <see cref="DefaultMaxSequenceLength"/>. A value is used as
    /// given. After loading, the property holds the length in effect.
    /// </summary>
    public int? MaxSequenceLength { get; set; }

    /// <summary>
    /// Gets or sets whether to normalize embeddings to unit vectors (L2 normalization).
    /// Defaults to true.
    /// </summary>
    public bool NormalizeEmbeddings { get; set; } = true;

    /// <summary>
    /// How token embeddings are pooled into one vector (ONNX models). <see langword="null"/> — the
    /// default — lets the model decide: its own <c>1_Pooling/config.json</c> first, then the catalog
    /// entry for a known alias, then <see cref="PoolingMode.Mean"/>. A value is used as given. After
    /// loading, the property holds the pooling in effect.
    /// </summary>
    public PoolingMode? PoolingMode { get; set; }

    /// <summary>
    /// Gets or sets whether to convert text to lowercase before tokenization.
    /// Defaults to true (for uncased models).
    /// </summary>
    public bool DoLowerCase { get; set; } = true;

    /// <summary>
    /// Gets or sets llama-server acquisition policy (version pinning, a pre-provisioned binary
    /// path, auto-update behavior) for GGUF model loading.
    /// If null, the process-wide default (<see cref="LlamaServerUpdateService.Instance"/>,
    /// unpinned "latest" resolution) is used. Only applies to GGUF models.
    /// </summary>
    public LlamaServerUpdateOptions? ServerUpdateOptions { get; set; }

    /// <summary>
    /// Gets or sets whether to disable automatic model download.
    /// When true, loading uses only the local cache and throws <see cref="LMSupply.Exceptions.ModelNotFoundException"/>
    /// if a model file is not there, writing nothing to the cache.
    /// <para>Default: false</para>
    /// </summary>
    public bool DisableAutoDownload { get; set; }
}

/// <summary>
/// Specifies the pooling strategy for sentence embeddings.
/// </summary>
public enum PoolingMode
{
    /// <summary>
    /// Mean pooling of all token embeddings (default, best for most models).
    /// </summary>
    Mean,

    /// <summary>
    /// Use the [CLS] token embedding (required for BGE models).
    /// </summary>
    Cls,

    /// <summary>
    /// Max pooling across all tokens.
    /// </summary>
    Max
}
