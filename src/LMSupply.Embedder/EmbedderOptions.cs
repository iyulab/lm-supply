using LMSupply.Llama.Server;

namespace LMSupply.Embedder;

/// <summary>
/// Configuration options for the embedding model.
/// </summary>
public sealed class EmbedderOptions : LMSupplyOptionsBase
{
    /// <summary>
    /// Gets or sets the maximum sequence length for tokenization.
    /// Defaults to 512.
    /// </summary>
    public int MaxSequenceLength { get; set; } = 512;

    /// <summary>
    /// Gets or sets whether to normalize embeddings to unit vectors (L2 normalization).
    /// Defaults to true.
    /// </summary>
    public bool NormalizeEmbeddings { get; set; } = true;

    /// <summary>
    /// Gets or sets the pooling mode for sentence embeddings.
    /// Defaults to Mean pooling.
    /// </summary>
    public PoolingMode PoolingMode { get; set; } = PoolingMode.Mean;

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
