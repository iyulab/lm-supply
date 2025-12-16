using LMSupply.Reranker.Core;
using LMSupply.Reranker.Infrastructure;
using LMSupply.Reranker.Models;
using LMSupply.Text;

namespace LMSupply.Reranker;

/// <summary>
/// A lightweight, zero-configuration semantic reranker for .NET.
/// </summary>
/// <remarks>
/// <para>
/// This class is thread-safe and can be shared across multiple threads.
/// For best performance, create a single instance and reuse it.
/// </para>
/// <example>
/// Basic usage:
/// <code>
/// var reranker = new Reranker();
/// var results = await reranker.RerankAsync("What is AI?", documents);
/// </code>
/// </example>
/// </remarks>
public sealed class Reranker : IRerankerModel
{
    private readonly RerankerOptions _options;
    private readonly ModelRegistry _registry;
    private readonly Lazy<Task<RerankerState>> _stateLazy;
    private readonly SemaphoreSlim _initLock = new(1, 1);
    private bool _disposed;

    /// <inheritdoc />
    public string ModelId => _options.ModelId;

    /// <summary>
    /// Initializes a new Reranker with default settings.
    /// Uses the default model (ms-marco-MiniLM-L-6-v2) with automatic download.
    /// </summary>
    public Reranker() : this(new RerankerOptions())
    {
    }

    /// <summary>
    /// Initializes a new Reranker with custom options.
    /// </summary>
    /// <param name="options">Configuration options.</param>
    /// <exception cref="ArgumentNullException">Options is null.</exception>
    public Reranker(RerankerOptions options)
    {
        _options = options?.Clone() ?? throw new ArgumentNullException(nameof(options));
        _registry = ModelRegistry.Default;
        _stateLazy = new Lazy<Task<RerankerState>>(InitializeAsync);
    }

    /// <inheritdoc />
    public async Task<IReadOnlyList<RankedResult>> RerankAsync(
        string query,
        IEnumerable<string> documents,
        int? topK = null,
        CancellationToken cancellationToken = default)
    {
        ValidateInputs(query, documents, out var docList);

        var scores = await ScoreInternalAsync(query, docList, cancellationToken);

        var results = new RankedResult[docList.Count];
        for (var i = 0; i < docList.Count; i++)
        {
            results[i] = new RankedResult(i, scores[i], docList[i]);
        }

        Array.Sort(results);

        if (topK.HasValue && topK.Value < results.Length)
        {
            return results[..topK.Value];
        }

        return results;
    }

    /// <inheritdoc />
    public async Task<float[]> ScoreAsync(
        string query,
        IEnumerable<string> documents,
        CancellationToken cancellationToken = default)
    {
        ValidateInputs(query, documents, out var docList);
        return await ScoreInternalAsync(query, docList, cancellationToken);
    }

    /// <inheritdoc />
    public async Task<IReadOnlyList<IReadOnlyList<RankedResult>>> RerankBatchAsync(
        IEnumerable<string> queries,
        IEnumerable<IEnumerable<string>> documentSets,
        int? topK = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(queries);
        ArgumentNullException.ThrowIfNull(documentSets);

        var queryList = queries.ToList();
        var docSetList = documentSets.Select(d => d.ToList()).ToList();

        if (queryList.Count != docSetList.Count)
        {
            throw new ArgumentException(
                "Number of queries must match number of document sets.",
                nameof(documentSets));
        }

        var results = new List<IReadOnlyList<RankedResult>>(queryList.Count);

        for (var i = 0; i < queryList.Count; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var ranked = await RerankAsync(queryList[i], docSetList[i], topK, cancellationToken);
            results.Add(ranked);
        }

        return results;
    }

    /// <inheritdoc />
    public async Task WarmupAsync(CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        _ = await _stateLazy.Value;
    }

    /// <inheritdoc />
    public ModelInfo? GetModelInfo()
    {
        if (_stateLazy.IsValueCreated && _stateLazy.Value.IsCompletedSuccessfully)
        {
            return _stateLazy.Value.Result.ModelInfo;
        }
        return null;
    }

    private async Task<float[]> ScoreInternalAsync(
        string query,
        List<string> documents,
        CancellationToken cancellationToken)
    {
        var state = await _stateLazy.Value;
        cancellationToken.ThrowIfCancellationRequested();

        var allScores = new float[documents.Count];
        var batchSize = _options.BatchSize;

        for (var offset = 0; offset < documents.Count; offset += batchSize)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var batchDocs = documents
                .Skip(offset)
                .Take(Math.Min(batchSize, documents.Count - offset))
                .ToList();

            var batch = state.Tokenizer.EncodePairBatch(query, batchDocs);
            var scores = state.Inference.Infer(batch);

            Array.Copy(scores, 0, allScores, offset, scores.Length);
        }

        return allScores;
    }

    private async Task<RerankerState> InitializeAsync()
    {
        await _initLock.WaitAsync();
        try
        {
            // Resolve model
            var modelInfo = _registry.Resolve(_options.ModelId);

            // Ensure model files are available
            using var modelManager = new ModelManager(
                _options.CacheDirectory,
                !_options.DisableAutoDownload);

            var modelPaths = await modelManager.EnsureModelAsync(modelInfo);

            // Determine max sequence length
            var maxLength = _options.MaxSequenceLength ?? modelInfo.MaxSequenceLength;

            // Initialize tokenizer using Text.Core
            var modelDir = Path.GetDirectoryName(modelPaths.TokenizerPath)!;
            var tokenizer = await TokenizerFactory.CreateWordPiecePairAsync(modelDir, maxLength);

            // Initialize inference engine
            var inference = CrossEncoderInference.Create(
                modelPaths.ModelPath,
                modelInfo,
                _options.Provider,
                _options.ThreadCount);

            return new RerankerState(modelInfo, tokenizer, inference);
        }
        finally
        {
            _initLock.Release();
        }
    }

    private static void ValidateInputs(string query, IEnumerable<string> documents, out List<string> docList)
    {
        ArgumentNullException.ThrowIfNull(query);
        ArgumentNullException.ThrowIfNull(documents);

        if (string.IsNullOrWhiteSpace(query))
        {
            throw new ArgumentException("Query cannot be empty or whitespace.", nameof(query));
        }

        docList = documents.ToList();

        if (docList.Count == 0)
        {
            throw new ArgumentException("Documents collection cannot be empty.", nameof(documents));
        }
    }

    /// <inheritdoc />
    public async ValueTask DisposeAsync()
    {
        if (_disposed) return;

        if (_stateLazy.IsValueCreated)
        {
            try
            {
                var state = await _stateLazy.Value;
                state.Tokenizer.Dispose();
                state.Inference.Dispose();
            }
            catch
            {
                // Ignore initialization errors during disposal
            }
        }

        _initLock.Dispose();
        _disposed = true;
    }

    /// <summary>
    /// Internal state holding initialized components.
    /// </summary>
    private sealed record RerankerState(
        ModelInfo ModelInfo,
        IPairTokenizer Tokenizer,
        CrossEncoderInference Inference);
}
