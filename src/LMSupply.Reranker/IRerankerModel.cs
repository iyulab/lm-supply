using LMSupply.Reranker.Models;

namespace LMSupply.Reranker;

/// <summary>
/// Interface for semantic document reranking.
/// </summary>
public interface IRerankerModel : IAsyncDisposable
{
    /// <summary>
    /// Gets the model identifier.
    /// </summary>
    string ModelId { get; }

    /// <summary>
    /// Reranks documents by relevance to a query.
    /// </summary>
    /// <param name="query">The search query.</param>
    /// <param name="documents">The documents to rerank.</param>
    /// <param name="topK">Maximum number of results to return. Null returns all documents.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Documents sorted by relevance score (highest first).</returns>
    /// <exception cref="ArgumentNullException">Query or documents is null.</exception>
    /// <exception cref="ArgumentException">Query is empty or documents collection is empty.</exception>
    Task<IReadOnlyList<RankedResult>> RerankAsync(
        string query,
        IEnumerable<string> documents,
        int? topK = null,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Calculates relevance scores for documents without sorting.
    /// </summary>
    /// <param name="query">The search query.</param>
    /// <param name="documents">The documents to score.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Relevance scores in the same order as input documents (0.0 to 1.0).</returns>
    /// <exception cref="ArgumentNullException">Query or documents is null.</exception>
    /// <exception cref="ArgumentException">Query is empty or documents collection is empty.</exception>
    Task<float[]> ScoreAsync(
        string query,
        IEnumerable<string> documents,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Reranks multiple query-document sets in batch.
    /// </summary>
    /// <param name="queries">The search queries.</param>
    /// <param name="documentSets">Document sets corresponding to each query.</param>
    /// <param name="topK">Maximum number of results per query. Null returns all documents.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Ranked results for each query.</returns>
    Task<IReadOnlyList<IReadOnlyList<RankedResult>>> RerankBatchAsync(
        IEnumerable<string> queries,
        IEnumerable<IEnumerable<string>> documentSets,
        int? topK = null,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Pre-loads the model to avoid cold start latency on first inference.
    /// </summary>
    /// <param name="cancellationToken">Cancellation token.</param>
    Task WarmupAsync(CancellationToken cancellationToken = default);

    /// <summary>
    /// Gets information about the loaded model.
    /// </summary>
    /// <returns>Model information, or null if not yet loaded.</returns>
    ModelInfo? GetModelInfo();
}
