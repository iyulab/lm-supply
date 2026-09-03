using LMSupply.Hardware;

namespace LMSupply.Embedder.Utils;

/// <summary>
/// Configuration information for a pre-configured embedding model.
/// </summary>
public sealed record ModelInfo : IModelInfoBase, IModelMemoryInfo
{
    /// <summary>
    /// Gets the unique identifier for this model (HuggingFace repository ID).
    /// </summary>
    public required string RepoId { get; init; }

    /// <summary>
    /// Gets the user-friendly alias name for this model (e.g., "default", "fast").
    /// Set internally by the registry when resolving.
    /// </summary>
    public string AliasName { get; init; } = string.Empty;

    /// <summary>
    /// Gets the embedding dimensions.
    /// </summary>
    public required int Dimensions { get; init; }

    /// <summary>
    /// Gets the maximum input sequence length.
    /// </summary>
    public required int MaxSequenceLength { get; init; }

    /// <summary>
    /// Gets the pooling mode for generating embeddings.
    /// </summary>
    public required PoolingMode PoolingMode { get; init; }

    /// <summary>
    /// Gets whether to lowercase input text.
    /// </summary>
    public required bool DoLowerCase { get; init; }

    /// <summary>
    /// Gets the model description.
    /// </summary>
    public string? Description { get; init; }

    /// <summary>
    /// Gets the text prefix this model's fine-tuning requires for search-query embeddings
    /// (e.g. "query: " for the E5 family, "search_query: " for Nomic). Null when the model
    /// needs no prefix convention. Applied automatically by <see cref="IEmbeddingModel.EmbedQueryAsync(string, System.Threading.CancellationToken)"/>.
    /// </summary>
    public string? QueryPrefix { get; init; }

    /// <summary>
    /// Gets the text prefix this model's fine-tuning requires for document/passage embeddings
    /// (e.g. "passage: " for the E5 family, "search_document: " for Nomic). Null when the model
    /// needs no prefix convention. Applied automatically by <see cref="IEmbeddingModel.EmbedPassageAsync(string, System.Threading.CancellationToken)"/>.
    /// </summary>
    public string? PassagePrefix { get; init; }

    /// <summary>
    /// Gets the subfolder within the HuggingFace repository.
    /// </summary>
    public string? Subfolder { get; init; }

    /// <summary>
    /// Gets the approximate model size in bytes.
    /// </summary>
    public long SizeBytes { get; init; }

    /// <summary>
    /// Gets the number of model parameters.
    /// </summary>
    public long Parameters { get; init; }

    // IModelInfoBase implementation
    string IModelInfoBase.Id => RepoId;
    int? IModelInfoBase.ContextLength => MaxSequenceLength;

    // IModelMemoryInfo explicit implementation
    long? IModelMemoryInfo.EstimatedSizeBytes => SizeBytes > 0 ? SizeBytes : null;
    long IModelMemoryInfo.ParameterCount => Parameters > 0 ? Parameters : SizeBytes / 2;
    string? IModelMemoryInfo.QuantizationType => null;
}
