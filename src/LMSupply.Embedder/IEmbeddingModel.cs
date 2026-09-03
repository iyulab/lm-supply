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
    /// Generates an embedding for a single text.
    /// </summary>
    ValueTask<float[]> EmbedAsync(string text, CancellationToken cancellationToken = default);

    /// <summary>
    /// Generates embeddings for multiple texts in batch.
    /// </summary>
    ValueTask<float[][]> EmbedAsync(IReadOnlyList<string> texts, CancellationToken cancellationToken = default);

    /// <summary>
    /// Generates a Matryoshka embedding truncated to the specified dimension count.
    /// Computes the full embedding then slices and re-normalizes to the requested size.
    /// </summary>
    /// <param name="text">The input text to embed.</param>
    /// <param name="dimensions">Target dimension count (1 ≤ dimensions ≤ model.Dimensions).</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    ValueTask<float[]> EmbedAsync(string text, int dimensions, CancellationToken cancellationToken = default);

    /// <summary>
    /// Generates Matryoshka embeddings for multiple texts truncated to the specified dimension count.
    /// </summary>
    /// <param name="texts">The input texts to embed.</param>
    /// <param name="dimensions">Target dimension count (1 ≤ dimensions ≤ model.Dimensions).</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    ValueTask<float[][]> EmbedAsync(IReadOnlyList<string> texts, int dimensions, CancellationToken cancellationToken = default);

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
    /// Generates a query embedding, applying the model's <see cref="ModelInfo.QueryPrefix"/>
    /// automatically when it has one (e.g. the E5 family's "query: " convention). A no-op
    /// passthrough to <see cref="EmbedAsync(string, CancellationToken)"/> when the model has no
    /// query prefix or no <see cref="ModelInfo"/> at all.
    /// </summary>
    ValueTask<float[]> EmbedQueryAsync(string text, CancellationToken cancellationToken = default) =>
        EmbedAsync(WithPrefix(text, GetModelInfo()?.QueryPrefix), cancellationToken);

    /// <summary>
    /// Batch form of <see cref="EmbedQueryAsync(string, CancellationToken)"/>.
    /// </summary>
    ValueTask<float[][]> EmbedQueryAsync(IReadOnlyList<string> texts, CancellationToken cancellationToken = default) =>
        EmbedAsync(WithPrefix(texts, GetModelInfo()?.QueryPrefix), cancellationToken);

    /// <summary>
    /// Matryoshka-truncated form of <see cref="EmbedQueryAsync(string, CancellationToken)"/>.
    /// </summary>
    ValueTask<float[]> EmbedQueryAsync(string text, int dimensions, CancellationToken cancellationToken = default) =>
        EmbedAsync(WithPrefix(text, GetModelInfo()?.QueryPrefix), dimensions, cancellationToken);

    /// <summary>
    /// Batch, Matryoshka-truncated form of <see cref="EmbedQueryAsync(string, CancellationToken)"/>.
    /// </summary>
    ValueTask<float[][]> EmbedQueryAsync(IReadOnlyList<string> texts, int dimensions, CancellationToken cancellationToken = default) =>
        EmbedAsync(WithPrefix(texts, GetModelInfo()?.QueryPrefix), dimensions, cancellationToken);

    /// <summary>
    /// Generates a passage/document embedding, applying the model's
    /// <see cref="ModelInfo.PassagePrefix"/> automatically when it has one (e.g. the E5 family's
    /// "passage: " convention). A no-op passthrough to
    /// <see cref="EmbedAsync(string, CancellationToken)"/> when the model has no passage prefix
    /// or no <see cref="ModelInfo"/> at all.
    /// </summary>
    ValueTask<float[]> EmbedPassageAsync(string text, CancellationToken cancellationToken = default) =>
        EmbedAsync(WithPrefix(text, GetModelInfo()?.PassagePrefix), cancellationToken);

    /// <summary>
    /// Batch form of <see cref="EmbedPassageAsync(string, CancellationToken)"/>.
    /// </summary>
    ValueTask<float[][]> EmbedPassageAsync(IReadOnlyList<string> texts, CancellationToken cancellationToken = default) =>
        EmbedAsync(WithPrefix(texts, GetModelInfo()?.PassagePrefix), cancellationToken);

    /// <summary>
    /// Matryoshka-truncated form of <see cref="EmbedPassageAsync(string, CancellationToken)"/>.
    /// </summary>
    ValueTask<float[]> EmbedPassageAsync(string text, int dimensions, CancellationToken cancellationToken = default) =>
        EmbedAsync(WithPrefix(text, GetModelInfo()?.PassagePrefix), dimensions, cancellationToken);

    /// <summary>
    /// Batch, Matryoshka-truncated form of <see cref="EmbedPassageAsync(string, CancellationToken)"/>.
    /// </summary>
    ValueTask<float[][]> EmbedPassageAsync(IReadOnlyList<string> texts, int dimensions, CancellationToken cancellationToken = default) =>
        EmbedAsync(WithPrefix(texts, GetModelInfo()?.PassagePrefix), dimensions, cancellationToken);

    private static string WithPrefix(string text, string? prefix) =>
        string.IsNullOrEmpty(prefix) ? text : prefix + text;

    private static IReadOnlyList<string> WithPrefix(IReadOnlyList<string> texts, string? prefix) =>
        string.IsNullOrEmpty(prefix) ? texts : [.. texts.Select(t => prefix + t)];
}
