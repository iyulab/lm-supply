using System.Buffers;
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
    private readonly ModelInfo? _modelInfo;
    private readonly string? _modelPath;
    private bool _disposed;
    private bool _warmedUp;

    public string ModelId { get; }
    public int Dimensions => _engine.HiddenSize;

    /// <inheritdoc />
    public long? EstimatedMemoryBytes => _modelPath is not null && File.Exists(_modelPath)
        ? new FileInfo(_modelPath).Length * 2
        : null;

    /// <inheritdoc />
    public bool IsGpuActive => _engine.IsGpuActive;

    /// <inheritdoc />
    public IReadOnlyList<string> ActiveProviders => _engine.ActiveProviders;

    /// <inheritdoc />
    public ExecutionProvider RequestedProvider => _engine.RequestedProvider;

    internal EmbeddingModel(
        string modelId,
        OnnxInferenceEngine engine,
        ISequenceTokenizer tokenizer,
        IPoolingStrategy poolingStrategy,
        EmbedderOptions options,
        ModelInfo? modelInfo = null,
        string? modelPath = null)
    {
        ModelId = modelId;
        _engine = engine;
        _tokenizer = tokenizer;
        _poolingStrategy = poolingStrategy;
        _options = options;
        _modelInfo = modelInfo;
        _modelPath = modelPath;
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

    public async ValueTask<float[]> EmbedAsync(string text, CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        // Tokenize
        var encoded = _tokenizer.EncodeSequence(text, _options.MaxSequenceLength);

        // Run inference. CancellableInference guarantees control returns to the caller if the
        // token is cancelled, even when the native ONNX call (e.g. cold DirectML init) ignores it.
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

    public async ValueTask<float[][]> EmbedAsync(IReadOnlyList<string> texts, CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        if (texts.Count == 0)
            return [];

        // Tokenize all texts
        var encodedBatch = _tokenizer.EncodeBatch(texts, _options.MaxSequenceLength);

        // Get jagged arrays for batch inference
        var allInputIds = encodedBatch.GetInputIdsJagged();
        var allAttentionMasks = encodedBatch.GetAttentionMasksJagged();

        // Run inference one item at a time, each under its own CancellableInference bound. The
        // bound exists to catch a hung native call, and a hang shows up on a single item — whereas
        // a whole batch on CPU can legitimately run well past one item's bound (observed: a 5-item
        // bge-m3 batch on CPU exceeded the 60s default and was reported as a hang). Per-item bounds
        // also let timeout recovery retry only the item that timed out, not the entire batch.
        var allTokenEmbeddings = new float[texts.Count][];
        for (int i = 0; i < texts.Count; i++)
        {
            var inputIds = allInputIds[i];
            var attentionMask = allAttentionMasks[i];
            allTokenEmbeddings[i] = await RunWithTimeoutRecoveryAsync(
                () => _engine.RunInference(inputIds, attentionMask, cancellationToken),
                cancellationToken);
        }

        // Pool each to sentence embedding with parallel processing
        var results = new float[texts.Count][];
        int seqLength = encodedBatch.SequenceLength;
        int hiddenDim = Dimensions;
        bool normalize = _options.NormalizeEmbeddings;

        Parallel.For(0, texts.Count, i =>
        {
            // Rent buffer from pool for intermediate work
            var resultBuffer = ArrayPool<float>.Shared.Rent(hiddenDim);
            try
            {
                var resultSpan = resultBuffer.AsSpan(0, hiddenDim);

                _poolingStrategy.Pool(
                    allTokenEmbeddings[i].AsSpan(),
                    allAttentionMasks[i].AsSpan(),
                    resultSpan,
                    seqLength,
                    hiddenDim);

                if (normalize)
                {
                    VectorOperations.NormalizeL2(resultSpan);
                }

                // Copy to final result array
                results[i] = resultSpan.ToArray();
            }
            finally
            {
                ArrayPool<float>.Shared.Return(resultBuffer);
            }
        });

        return results;
    }

    public async ValueTask<float[]> EmbedAsync(
        string text,
        int dimensions,
        CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (dimensions <= 0 || dimensions > Dimensions)
            throw new ArgumentOutOfRangeException(
                nameof(dimensions), $"dimensions must be between 1 and {Dimensions}.");

        var full = await EmbedAsync(text, cancellationToken);
        if (dimensions == Dimensions)
            return full;

        var truncated = full[..dimensions];
        VectorOperations.NormalizeL2(truncated.AsSpan());
        return truncated;
    }

    public async ValueTask<float[][]> EmbedAsync(
        IReadOnlyList<string> texts,
        int dimensions,
        CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (dimensions <= 0 || dimensions > Dimensions)
            throw new ArgumentOutOfRangeException(
                nameof(dimensions), $"dimensions must be between 1 and {Dimensions}.");

        var full = await EmbedAsync(texts, cancellationToken);
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
        await EmbedAsync("warmup", cancellationToken);
        _warmedUp = true;
    }

    public ModelInfo? GetModelInfo() => _modelInfo;

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
