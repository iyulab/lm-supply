using LMSupply.Embedder.Utils;

namespace LMSupply.Embedder;

/// <summary>
/// Represents a loaded embedding model that can generate text embeddings.
/// </summary>
public interface IEmbeddingModel : IAsyncDisposable
{
    /// <summary>
    /// Gets the model identifier.
    /// </summary>
    string ModelId { get; }

    /// <summary>
    /// Gets the embedding vector dimension.
    /// </summary>
    int Dimensions { get; }

    /// <summary>
    /// Gets whether GPU acceleration is being used for inference.
    /// </summary>
    bool IsGpuActive { get; }

    /// <summary>
    /// Gets the list of active execution providers.
    /// </summary>
    IReadOnlyList<string> ActiveProviders { get; }

    /// <summary>
    /// Gets the execution provider that was requested.
    /// </summary>
    ExecutionProvider RequestedProvider { get; }

    /// <summary>
    /// Gets the estimated memory usage of this model in bytes.
    /// Based on ONNX model file size with overhead factor.
    /// </summary>
    long? EstimatedMemoryBytes { get; }

    /// <summary>
    /// An opaque string that changes when, and only when, this library would produce different vectors
    /// for this model id: the tokenizer and its normalization convention, the pooling, L2
    /// normalization, the prompt prefixes, the sequence length in effect, the model file
    /// (quantization variant) and the epoch of each of this library's implementations of those
    /// steps. A consumer that stores vectors stores this next to them and, when a later load reports a
    /// different value, knows those vectors are stale for this model. The execution provider and GPU
    /// are not part of it. <see langword="null"/> when the implementation does not compute one.
    /// </summary>
    string? VectorSpaceRevision => null;

    /// <summary>
    /// Embeds the text exactly as given, with no prefix. For a model trained with prefixes (the E5 family, Nomic) this
    /// is not a vector the model was trained to produce: use it only when the text already carries the instruction the
    /// model expects (one the caller writes itself). <see cref="EmbedAsync(string, CancellationToken)"/>,
    /// <see cref="EmbedQueryAsync(string, CancellationToken)"/> and <see cref="EmbedPassageAsync(string, CancellationToken)"/>
    /// apply the model's own convention.
    /// </summary>
    ValueTask<float[]> EmbedRawAsync(string text, CancellationToken cancellationToken = default);

    /// <summary>
    /// Batch form of <see cref="EmbedRawAsync(string, CancellationToken)"/>.
    /// </summary>
    ValueTask<float[][]> EmbedRawAsync(IReadOnlyList<string> texts, CancellationToken cancellationToken = default);

    /// <summary>
    /// Matryoshka form of <see cref="EmbedRawAsync(string, CancellationToken)"/>: the full embedding, sliced to
    /// <paramref name="dimensions"/> and re-normalized.
    /// </summary>
    /// <param name="text">The input text to embed, as given.</param>
    /// <param name="dimensions">Target dimension count (1 ≤ dimensions ≤ model.Dimensions).</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    ValueTask<float[]> EmbedRawAsync(string text, int dimensions, CancellationToken cancellationToken = default);

    /// <summary>
    /// Batch, Matryoshka form of <see cref="EmbedRawAsync(string, CancellationToken)"/>.
    /// </summary>
    /// <param name="texts">The input texts to embed, as given.</param>
    /// <param name="dimensions">Target dimension count (1 ≤ dimensions ≤ model.Dimensions).</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    ValueTask<float[][]> EmbedRawAsync(IReadOnlyList<string> texts, int dimensions, CancellationToken cancellationToken = default);

    /// <summary>
    /// Pre-loads the model to avoid cold start latency on first inference.
    /// </summary>
    /// <param name="cancellationToken">Cancellation token.</param>
    Task WarmupAsync(CancellationToken cancellationToken = default);

    /// <summary>
    /// Gets information about the loaded model.
    /// </summary>
    /// <returns>Model information, or null if not available.</returns>
    ModelInfo? GetModelInfo();

    /// <summary>
    /// Generates an embedding for text with no retrieval role — semantic similarity, clustering, features — applying
    /// the model's <see cref="ModelInfo.DefaultPrefix"/> when it has one (e.g. the E5 family's "query: ", which its
    /// model card prescribes for every task other than retrieval). For a model without one this is
    /// <see cref="EmbedRawAsync(string, CancellationToken)"/>. For retrieval, embed the two sides with
    /// <see cref="EmbedQueryAsync(string, CancellationToken)"/> and <see cref="EmbedPassageAsync(string, CancellationToken)"/>.
    /// </summary>
    ValueTask<float[]> EmbedAsync(string text, CancellationToken cancellationToken = default) =>
        EmbedRawAsync(WithPrefix(text, GetModelInfo()?.DefaultPrefix), cancellationToken);

    /// <summary>
    /// Batch form of <see cref="EmbedAsync(string, CancellationToken)"/>.
    /// </summary>
    ValueTask<float[][]> EmbedAsync(IReadOnlyList<string> texts, CancellationToken cancellationToken = default) =>
        EmbedRawAsync(WithPrefix(texts, GetModelInfo()?.DefaultPrefix), cancellationToken);

    /// <summary>
    /// Matryoshka-truncated form of <see cref="EmbedAsync(string, CancellationToken)"/>: the full embedding, sliced to
    /// <paramref name="dimensions"/> and re-normalized.
    /// </summary>
    /// <param name="text">The input text to embed.</param>
    /// <param name="dimensions">Target dimension count (1 ≤ dimensions ≤ model.Dimensions).</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    ValueTask<float[]> EmbedAsync(string text, int dimensions, CancellationToken cancellationToken = default) =>
        EmbedRawAsync(WithPrefix(text, GetModelInfo()?.DefaultPrefix), dimensions, cancellationToken);

    /// <summary>
    /// Batch, Matryoshka-truncated form of <see cref="EmbedAsync(string, CancellationToken)"/>.
    /// </summary>
    /// <param name="texts">The input texts to embed.</param>
    /// <param name="dimensions">Target dimension count (1 ≤ dimensions ≤ model.Dimensions).</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    ValueTask<float[][]> EmbedAsync(IReadOnlyList<string> texts, int dimensions, CancellationToken cancellationToken = default) =>
        EmbedRawAsync(WithPrefix(texts, GetModelInfo()?.DefaultPrefix), dimensions, cancellationToken);

    /// <summary>
    /// Generates a query embedding, applying the model's <see cref="ModelInfo.QueryPrefix"/>
    /// automatically when it has one (e.g. the E5 family's "query: " convention); the same as
    /// <see cref="EmbedRawAsync(string, CancellationToken)"/> when the model has no query prefix or no
    /// <see cref="ModelInfo"/> at all.
    /// </summary>
    ValueTask<float[]> EmbedQueryAsync(string text, CancellationToken cancellationToken = default) =>
        EmbedRawAsync(WithPrefix(text, GetModelInfo()?.QueryPrefix), cancellationToken);

    /// <summary>
    /// Batch form of <see cref="EmbedQueryAsync(string, CancellationToken)"/>.
    /// </summary>
    ValueTask<float[][]> EmbedQueryAsync(IReadOnlyList<string> texts, CancellationToken cancellationToken = default) =>
        EmbedRawAsync(WithPrefix(texts, GetModelInfo()?.QueryPrefix), cancellationToken);

    /// <summary>
    /// Matryoshka-truncated form of <see cref="EmbedQueryAsync(string, CancellationToken)"/>.
    /// </summary>
    ValueTask<float[]> EmbedQueryAsync(string text, int dimensions, CancellationToken cancellationToken = default) =>
        EmbedRawAsync(WithPrefix(text, GetModelInfo()?.QueryPrefix), dimensions, cancellationToken);

    /// <summary>
    /// Batch, Matryoshka-truncated form of <see cref="EmbedQueryAsync(string, CancellationToken)"/>.
    /// </summary>
    ValueTask<float[][]> EmbedQueryAsync(IReadOnlyList<string> texts, int dimensions, CancellationToken cancellationToken = default) =>
        EmbedRawAsync(WithPrefix(texts, GetModelInfo()?.QueryPrefix), dimensions, cancellationToken);

    /// <summary>
    /// Generates a passage/document embedding, applying the model's
    /// <see cref="ModelInfo.PassagePrefix"/> automatically when it has one (e.g. the E5 family's
    /// "passage: " convention); the same as
    /// <see cref="EmbedRawAsync(string, CancellationToken)"/> when the model has no passage prefix or no
    /// <see cref="ModelInfo"/> at all.
    /// </summary>
    ValueTask<float[]> EmbedPassageAsync(string text, CancellationToken cancellationToken = default) =>
        EmbedRawAsync(WithPrefix(text, GetModelInfo()?.PassagePrefix), cancellationToken);

    /// <summary>
    /// Batch form of <see cref="EmbedPassageAsync(string, CancellationToken)"/>.
    /// </summary>
    ValueTask<float[][]> EmbedPassageAsync(IReadOnlyList<string> texts, CancellationToken cancellationToken = default) =>
        EmbedRawAsync(WithPrefix(texts, GetModelInfo()?.PassagePrefix), cancellationToken);

    /// <summary>
    /// Matryoshka-truncated form of <see cref="EmbedPassageAsync(string, CancellationToken)"/>.
    /// </summary>
    ValueTask<float[]> EmbedPassageAsync(string text, int dimensions, CancellationToken cancellationToken = default) =>
        EmbedRawAsync(WithPrefix(text, GetModelInfo()?.PassagePrefix), dimensions, cancellationToken);

    /// <summary>
    /// Batch, Matryoshka-truncated form of <see cref="EmbedPassageAsync(string, CancellationToken)"/>.
    /// </summary>
    ValueTask<float[][]> EmbedPassageAsync(IReadOnlyList<string> texts, int dimensions, CancellationToken cancellationToken = default) =>
        EmbedRawAsync(WithPrefix(texts, GetModelInfo()?.PassagePrefix), dimensions, cancellationToken);

    private static string WithPrefix(string text, string? prefix) =>
        string.IsNullOrEmpty(prefix) ? text : prefix + text;

    private static IReadOnlyList<string> WithPrefix(IReadOnlyList<string> texts, string? prefix) =>
        string.IsNullOrEmpty(prefix) ? texts : [.. texts.Select(t => prefix + t)];
}
