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
    /// Generates an embedding for a single text.
    /// </summary>
    ValueTask<float[]> EmbedAsync(string text, CancellationToken cancellationToken = default);

    /// <summary>
    /// Generates embeddings for multiple texts in batch.
    /// </summary>
    ValueTask<float[][]> EmbedAsync(IReadOnlyList<string> texts, CancellationToken cancellationToken = default);

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
}
