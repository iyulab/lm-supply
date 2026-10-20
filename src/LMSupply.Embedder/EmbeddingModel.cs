using LMSupply.Download;
using LMSupply.Embedder.Inference;
using LMSupply.Embedder.Pooling;
using LMSupply.Embedder.Utils;
using LMSupply.Exceptions;
using LMSupply.Inference;
using LMSupply.Text;

namespace LMSupply.Embedder;

/// <summary>
/// Implementation of IEmbeddingModel with memory pooling and batch parallelization.
/// </summary>
internal sealed class EmbeddingModel : IEmbeddingModel
{
    private readonly OnnxInferenceEngine _engine;
    private readonly ISequenceTokenizer _tokenizer;
    private readonly IPoolingStrategy _poolingStrategy;
    private readonly EmbedderOptions _options;
    private readonly EmbedderModelInfo? _modelInfo;
    private readonly string? _modelPath;
    private readonly VectorSpaceDescriptor? _vectorSpace;
    private bool _disposed;
    private bool _warmedUp;

    /// <summary>
    /// Padded tokens one batched inference run may hold when the model's maximum sequence length is shorter: a
    /// 512-token model runs up to 16 full-length texts, or 64 texts of 128 tokens, per call.
    /// </summary>
    internal const int BatchTokenBudget = 8192;

    public string ModelId { get; }
    public int Dimensions => _engine.HiddenSize;

    /// <inheritdoc />
    public long? EstimatedMemoryBytes => _modelPath is not null && File.Exists(_modelPath)
        ? WeightBytes(_modelPath) * 2
        : null;

    // A large ONNX model keeps its weights in an external data file beside the graph (model.onnx_data or
    // model.onnx.data); the graph file alone is a few hundred kilobytes.
    private static long WeightBytes(string modelPath) =>
        new[] { modelPath, modelPath + "_data", modelPath + ".data" }
            .Sum(CacheManager.GetContentLengthOrZero);

    /// <inheritdoc />
    public bool IsGpuActive => _engine.IsGpuActive;

    /// <inheritdoc />
    public IReadOnlyList<string> ActiveProviders => _engine.ActiveProviders;

    /// <inheritdoc />
    public ExecutionProvider RequestedProvider => _engine.RequestedProvider;

    /// <inheritdoc />
    public string? VectorSpaceRevision => _vectorSpace?.Revision;

    /// <inheritdoc />
    public string? RetrievalVectorSpaceRevision => _vectorSpace?.RetrievalRevision;

    internal EmbeddingModel(
        string modelId,
        OnnxInferenceEngine engine,
        ISequenceTokenizer tokenizer,
        IPoolingStrategy poolingStrategy,
        EmbedderOptions options,
        EmbedderModelInfo? modelInfo = null,
        string? modelPath = null,
        VectorSpaceDescriptor? vectorSpace = null)
    {
        ModelId = modelId;
        _engine = engine;
        _tokenizer = tokenizer;
        _poolingStrategy = poolingStrategy;
        _options = options;
        _modelInfo = modelInfo;
        _modelPath = modelPath;
        _vectorSpace = vectorSpace;
    }

    /// <summary>
    /// Runs <paramref name="work"/> under <see cref="CancellableInference"/>'s bound. If the bound
    /// elapses while a GPU provider is active, the engine is moved to the next provider in its
    /// fallback chain (same chain a run-time provider crash uses) and the work is retried exactly
    /// once. A timeout on CPU, or a second timeout after recovery, surfaces unchanged.
    /// </summary>
    private async Task<T> RunWithTimeoutRecoveryAsync<T>(Func<T> work, CancellationToken cancellationToken)
    {
        try
        {
            return await CancellableInference.RunAsync(work, cancellationToken);
        }
        catch (InferenceTimeoutException ex) when (_engine.TryRecoverAfterTimeout(ex))
        {
            return await CancellableInference.RunAsync(work, cancellationToken);
        }
    }

    public async ValueTask<float[]> EmbedRawAsync(string text, CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        // Tokenize
        var encoded = _tokenizer.EncodeSequence(text, _options.MaxSequenceLength ?? EmbedderOptions.DefaultMaxSequenceLength);

        // Run inference. CancellableInference guarantees control returns to the caller if the
        // token is cancelled, even when the native ONNX call (e.g. a cold GPU kernel init) ignores it.
        var tokenEmbeddings = await RunWithTimeoutRecoveryAsync(
            () => _engine.RunInference(encoded.InputIds, encoded.AttentionMask, cancellationToken),
            cancellationToken);

        // Pool to sentence embedding using pooled buffer
        int seqLength = encoded.InputIds.Length;
        var result = new float[Dimensions];

        _poolingStrategy.Pool(
            tokenEmbeddings.AsSpan(),
            encoded.AttentionMask.AsSpan(),
            result.AsSpan(),
            seqLength,
            Dimensions);

        // Normalize if requested
        if (_options.NormalizeEmbeddings)
        {
            VectorOperations.NormalizeL2(result.AsSpan());
        }

        return result;
    }

    public async ValueTask<float[][]> EmbedRawAsync(IReadOnlyList<string> texts, CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        if (texts.Count == 0)
            return [];

        var maxLength = _options.MaxSequenceLength ?? EmbedderOptions.DefaultMaxSequenceLength;

        // Group the texts into runs of similar length: sorted by token count, each run takes texts while its padded
        // size (texts x longest) stays within the budget. One run is one [batch, length] inference, so short texts
        // share a call instead of each paying a call, and no text is padded to a much longer one. The budget is at
        // least one sequence of the longest length the model takes, so a run never costs more padded tokens than a
        // single maximum-length text, which the per-run timeout bound already allows for.
        var lengths = new int[texts.Count];
        for (int i = 0; i < texts.Count; i++)
            lengths[i] = _tokenizer.EncodeSequence(texts[i], maxLength).InputIds.Length;

        var results = new float[texts.Count][];
        int hiddenDim = Dimensions;
        bool normalize = _options.NormalizeEmbeddings;

        foreach (var group in PlanRuns(lengths, Math.Max(maxLength, BatchTokenBudget)))
        {
            var encoded = _tokenizer.EncodeBatch(group.Select(i => texts[i]).ToArray(), maxLength);
            var masks = encoded.GetAttentionMasksJagged();
            int seqLength = encoded.SequenceLength;

            var tokenEmbeddings = await RunWithTimeoutRecoveryAsync(
                () => _engine.RunBatchInference(encoded.InputIds, encoded.AttentionMask, cancellationToken),
                cancellationToken);

            int itemSize = seqLength * hiddenDim;
            for (int g = 0; g < group.Length; g++)
            {
                var result = new float[hiddenDim];
                _poolingStrategy.Pool(
                    tokenEmbeddings.AsSpan(g * itemSize, itemSize),
                    masks[g].AsSpan(),
                    result.AsSpan(),
                    seqLength,
                    hiddenDim);

                if (normalize)
                    VectorOperations.NormalizeL2(result.AsSpan());

                results[group[g]] = result;
            }
        }

        return results;
    }

    /// <summary>
    /// Splits texts of the given token <paramref name="lengths"/> into inference runs: indexes in ascending length,
    /// each run as long as its size times its longest length stays within <paramref name="budget"/>. A text longer
    /// than the budget runs alone.
    /// </summary>
    internal static List<int[]> PlanRuns(IReadOnlyList<int> lengths, int budget)
    {
        var order = Enumerable.Range(0, lengths.Count).OrderBy(i => lengths[i]).ToArray();
        var runs = new List<int[]>();
        for (int start = 0; start < order.Length;)
        {
            int end = start + 1;
            while (end < order.Length && (end - start + 1) * lengths[order[end]] <= budget)
                end++;

            runs.Add(order[start..end]);
            start = end;
        }

        return runs;
    }

    public async ValueTask<float[]> EmbedRawAsync(
        string text,
        int dimensions,
        CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (dimensions <= 0 || dimensions > Dimensions)
            throw new ArgumentOutOfRangeException(
                nameof(dimensions), $"dimensions must be between 1 and {Dimensions}.");

        var full = await EmbedRawAsync(text, cancellationToken);
        if (dimensions == Dimensions)
            return full;

        var truncated = full[..dimensions];
        VectorOperations.NormalizeL2(truncated.AsSpan());
        return truncated;
    }

    public async ValueTask<float[][]> EmbedRawAsync(
        IReadOnlyList<string> texts,
        int dimensions,
        CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (dimensions <= 0 || dimensions > Dimensions)
            throw new ArgumentOutOfRangeException(
                nameof(dimensions), $"dimensions must be between 1 and {Dimensions}.");

        var full = await EmbedRawAsync(texts, cancellationToken);
        if (dimensions == Dimensions)
            return full;

        var result = new float[full.Length][];
        for (int i = 0; i < full.Length; i++)
        {
            result[i] = full[i][..dimensions];
            VectorOperations.NormalizeL2(result[i].AsSpan());
        }
        return result;
    }

    public async Task WarmupAsync(CancellationToken cancellationToken = default)
    {
        if (_warmedUp) return;

        ObjectDisposedException.ThrowIf(_disposed, this);

        // Perform a dummy inference to warm up the model
        await EmbedRawAsync("warmup", cancellationToken);
        _warmedUp = true;
    }

    public EmbedderModelInfo? GetModelInfo() => _modelInfo;

    public async ValueTask DisposeAsync()
    {
        if (_disposed)
            return;

        _disposed = true;

        // Dispose ONNX Runtime resources
        _engine?.Dispose();

        // Satisfy async contract (currently no async cleanup needed)
        await Task.CompletedTask;

        // Suppress finalization since we've cleaned up
        GC.SuppressFinalize(this);
    }
}
