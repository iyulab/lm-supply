using System.Diagnostics;
using LMSupply.Download;
using LMSupply.Llama.Server;
using LMSupply.Reranker.Models;

namespace LMSupply.Reranker.Inference;

/// <summary>
/// GGUF reranker model implementation using llama-server with rank pooling.
/// </summary>
internal sealed class LlamaServerRerankerModel : IRerankerModel
{
    private readonly ServerLease _serverLease;
    private readonly RerankerOptions _options;
    private readonly string _modelPath;
    private bool _disposed;

    private LlamaServerRerankerModel(
        string modelId,
        string modelPath,
        ServerLease serverLease,
        RerankerOptions options)
    {
        ModelId = modelId;
        _modelPath = modelPath;
        _serverLease = serverLease;
        _options = options;
    }

    /// <summary>
    /// Loads a GGUF reranker model using llama-server.
    /// </summary>
    public static async Task<LlamaServerRerankerModel> LoadAsync(
        string modelId,
        string modelPath,
        RerankerOptions options,
        IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        // 1. Get llama-server via update service
        progress?.Report(new DownloadProgress
        {
            FileName = "llama-server",
            BytesDownloaded = 0,
            TotalBytes = 0,
            Phase = DownloadPhase.Downloading
        });

        var preferredBackend = global::LMSupply.Llama.LlamaBackendSelector.MapProvider(
            options.Provider, Hardware.HardwareProfile.Current.GpuInfo);
        var updateService = LlamaServerUpdateService.Resolve(options.ServerUpdateOptions);
        var updateResult = await updateService.GetServerPathAsync(
            preferredBackend,
            progress,
            cancellationToken);

        if (!updateResult.Success)
        {
            throw new InvalidOperationException(
                $"Failed to get llama-server: {updateResult.Error}");
        }

        var serverPath = updateResult.ServerPath;
        var backend = updateResult.Backend;

        // 2. Configure server for reranking mode (embedding + rank pooling)
        progress?.Report(new DownloadProgress
        {
            FileName = Path.GetFileName(modelPath),
            BytesDownloaded = 30,
            TotalBytes = 100,
            Phase = DownloadPhase.Extracting
        });

        var contextSize = options.MaxSequenceLength ?? 512;

        var serverConfig = new LlamaServerConfig
        {
            ModelPath = modelPath,
            Port = 0, // Auto-assign
            ContextSize = contextSize,
            GpuLayers = backend == LlamaServerBackend.Cpu ? 0 : -1,
            BatchSize = 512,
            Parallel = 1,
            Mode = ServerMode.Reranking, // Enables --embedding and --pooling rank
            Pooling = PoolingType.Rank,
            ShutdownTimeout = TimeSpan.FromSeconds(10)
        };

        // 3. Lease server from pool
        var serverLease = await LlamaServerPool.Instance.LeaseAsync(
            serverPath,
            serverConfig,
            backend,
            progress,
            cancellationToken);

        progress?.Report(new DownloadProgress
        {
            FileName = Path.GetFileName(modelPath),
            BytesDownloaded = 100,
            TotalBytes = 100,
            Phase = DownloadPhase.Complete
        });

        return new LlamaServerRerankerModel(
            modelId,
            modelPath,
            serverLease,
            options);
    }

    /// <inheritdoc />
    public string ModelId { get; }

    /// <inheritdoc />
    public long? EstimatedMemoryBytes => File.Exists(_modelPath) ? new FileInfo(_modelPath).Length * 2 : null;

    /// <inheritdoc />
    public bool IsGpuActive => _serverLease.Backend != LlamaServerBackend.Cpu;

    /// <inheritdoc />
    public IReadOnlyList<string> ActiveProviders => IsGpuActive
        ? [$"llama-server-{_serverLease.Backend}", "CPU"]
        : ["llama-server-CPU"];

    /// <inheritdoc />
    public ExecutionProvider RequestedProvider => _options.Provider;

    /// <inheritdoc />
    public async Task<IReadOnlyList<RankedResult>> RerankAsync(
        string query,
        IEnumerable<string> documents,
        int? topK = null,
        CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();

        var docList = documents.ToList();
        if (docList.Count == 0)
        {
            return [];
        }

        var results = await _serverLease.Client.RerankAsync(
            query,
            docList,
            topK ?? docList.Count,
            cancellationToken);

        // Convert to RankedResult with documents
        var rankedResults = results
            .Select(r => new RankedResult(r.Index, r.RelevanceScore, docList[r.Index]))
            .OrderByDescending(r => r.Score)
            .ToList();

        // Warn if all scores are near-zero — likely model incompatibility with --pooling rank
        // (e.g., generative rerankers like Qwen3-Reranker require prompt-based scoring, not rank pooling)
        if (rankedResults.Count > 0 && rankedResults.All(r => MathF.Abs(r.Score) < 1e-4f))
        {
            Trace.TraceWarning(
                $"[LlamaServerReranker] All rerank scores are near-zero (max={rankedResults.Max(r => r.Score):E2}). " +
                $"Model '{ModelId}' may be incompatible with llama-server's rank pooling mode. " +
                "Generative rerankers (e.g., Qwen3-Reranker) require prompt-based scoring, not --pooling rank. " +
                "Consider using a cross-encoder model (e.g., BAAI/bge-reranker-v2-m3-GGUF).");
        }

        if (topK.HasValue && topK.Value < rankedResults.Count)
        {
            return rankedResults.Take(topK.Value).ToList();
        }

        return rankedResults;
    }

    /// <inheritdoc />
    public async Task<float[]> ScoreAsync(
        string query,
        IEnumerable<string> documents,
        CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();

        var docList = documents.ToList();
        if (docList.Count == 0)
        {
            return [];
        }

        var results = await _serverLease.Client.RerankAsync(
            query,
            docList,
            docList.Count,
            cancellationToken);

        // Return scores in original document order
        var scores = new float[docList.Count];
        foreach (var result in results)
        {
            scores[result.Index] = result.RelevanceScore;
        }

        return scores;
    }

    /// <inheritdoc />
    public async Task<IReadOnlyList<IReadOnlyList<RankedResult>>> RerankBatchAsync(
        IEnumerable<string> queries,
        IEnumerable<IEnumerable<string>> documentSets,
        int? topK = null,
        CancellationToken cancellationToken = default)
    {
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
        // Perform a minimal rerank to warm up
        await RerankAsync("warmup query", ["warmup document"], cancellationToken: cancellationToken);
    }

    /// <inheritdoc />
    public ModelInfo? GetModelInfo() => new()
    {
        Id = ModelId,
        AliasName = "gguf",
        DisplayName = Path.GetFileNameWithoutExtension(_modelPath),
        Parameters = 0, // Unknown for GGUF
        MaxSequenceLength = _options.MaxSequenceLength ?? 512,
        SizeBytes = EstimatedMemoryBytes ?? 0,
        OnnxFile = "", // N/A for GGUF
        TokenizerFile = "", // N/A - llama-server handles tokenization
        Description = $"GGUF reranker model via llama-server-{_serverLease.Backend}"
    };

    private void ThrowIfDisposed()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
    }

    public async ValueTask DisposeAsync()
    {
        if (_disposed)
            return;

        _disposed = true;
        await _serverLease.DisposeAsync();
    }
}
