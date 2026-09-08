using LMSupply.Download;
using LMSupply.Embedder.Utils;
using LMSupply.Llama.Server;

namespace LMSupply.Embedder.Inference;

/// <summary>
/// GGUF embedding model implementation using llama-server.
/// Uses llama-server with --embedding flag for GGUF model inference.
/// </summary>
internal sealed class LlamaServerEmbeddingModel : IEmbeddingModel
{
    private readonly ServerLease _serverLease;
    private readonly EmbedderOptions _options;
    private readonly string _modelPath;
    private bool _disposed;

    private LlamaServerEmbeddingModel(
        string modelId,
        string modelPath,
        ServerLease serverLease,
        int dimensions,
        EmbedderOptions options)
    {
        ModelId = modelId;
        _modelPath = modelPath;
        _serverLease = serverLease;
        Dimensions = dimensions;
        _options = options;
    }

    /// <summary>
    /// Loads a GGUF embedding model using llama-server.
    /// </summary>
    public static async Task<LlamaServerEmbeddingModel> LoadAsync(
        string modelId,
        string modelPath,
        EmbedderOptions options,
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

        // 2. Configure server for embedding mode
        progress?.Report(new DownloadProgress
        {
            FileName = Path.GetFileName(modelPath),
            BytesDownloaded = 30,
            TotalBytes = 100,
            Phase = DownloadPhase.Extracting
        });

        // Map pooling mode
        var poolingType = options.PoolingMode switch
        {
            PoolingMode.Cls => PoolingType.Cls,
            PoolingMode.Mean => PoolingType.Mean,
            PoolingMode.Max => PoolingType.Last, // Max isn't directly supported, use Last as fallback
            _ => PoolingType.Mean
        };

        var serverConfig = new LlamaServerConfig
        {
            ModelPath = modelPath,
            Port = 0, // Auto-assign
            ContextSize = options.MaxSequenceLength,
            GpuLayers = backend == LlamaServerBackend.Cpu ? 0 : -1,
            BatchSize = 512,
            Parallel = 1,
            Mode = ServerMode.Embedding,
            Pooling = poolingType,
            ServerVersion = updateResult.NewVersion ?? updateResult.PreviousVersion,
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
            BytesDownloaded = 80,
            TotalBytes = 100,
            Phase = DownloadPhase.Extracting
        });

        // 4. Determine embedding dimensions by running a test embedding
        int dimensions;
        try
        {
            var testEmbedding = await serverLease.Client.GenerateEmbeddingAsync("test", cancellationToken);
            dimensions = testEmbedding.Length;
        }
        catch (Exception ex)
        {
            await serverLease.DisposeAsync();
            throw new InvalidOperationException(
                $"Failed to determine embedding dimensions: {ex.Message}", ex);
        }

        progress?.Report(new DownloadProgress
        {
            FileName = Path.GetFileName(modelPath),
            BytesDownloaded = 100,
            TotalBytes = 100,
            Phase = DownloadPhase.Complete
        });

        return new LlamaServerEmbeddingModel(
            modelId,
            modelPath,
            serverLease,
            dimensions,
            options);
    }

    /// <inheritdoc />
    public string ModelId { get; }

    /// <inheritdoc />
    public int Dimensions { get; }

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
    public async ValueTask<float[]> EmbedAsync(string text, CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();

        var embedding = await _serverLease.Client.GenerateEmbeddingAsync(text, cancellationToken);

        if (_options.NormalizeEmbeddings)
        {
            NormalizeVector(embedding);
        }

        return embedding;
    }

    /// <inheritdoc />
    public async ValueTask<float[][]> EmbedAsync(IReadOnlyList<string> texts, CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();

        var embeddings = await _serverLease.Client.GenerateEmbeddingsBatchAsync(texts, cancellationToken);

        if (_options.NormalizeEmbeddings)
        {
            foreach (var embedding in embeddings)
            {
                NormalizeVector(embedding);
            }
        }

        return embeddings;
    }

    /// <inheritdoc />
    public async ValueTask<float[]> EmbedAsync(
        string text,
        int dimensions,
        CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();
        if (dimensions <= 0 || dimensions > Dimensions)
            throw new ArgumentOutOfRangeException(
                nameof(dimensions), $"dimensions must be between 1 and {Dimensions}.");

        var full = await EmbedAsync(text, cancellationToken);
        if (dimensions == Dimensions)
            return full;

        var truncated = full[..dimensions];
        NormalizeVector(truncated);
        return truncated;
    }

    /// <inheritdoc />
    public async ValueTask<float[][]> EmbedAsync(
        IReadOnlyList<string> texts,
        int dimensions,
        CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();
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
            NormalizeVector(result[i]);
        }
        return result;
    }

    /// <inheritdoc />
    public async Task WarmupAsync(CancellationToken cancellationToken = default)
    {
        // Server is already warmed up during LoadAsync (test embedding)
        // Perform another embedding to ensure everything is ready
        await EmbedAsync("warmup", cancellationToken);
    }

    /// <inheritdoc />
    public ModelInfo? GetModelInfo() => new()
    {
        RepoId = ModelId,
        Dimensions = Dimensions,
        MaxSequenceLength = _options.MaxSequenceLength,
        PoolingMode = _options.PoolingMode,
        DoLowerCase = _options.DoLowerCase,
        Description = $"GGUF embedding model via llama-server-{_serverLease.Backend}"
    };

    /// <summary>
    /// Normalizes a vector to unit length (L2 normalization).
    /// </summary>
    private static void NormalizeVector(float[] vector)
    {
        var sumSquares = 0f;
        for (var i = 0; i < vector.Length; i++)
        {
            sumSquares += vector[i] * vector[i];
        }

        if (sumSquares <= 0)
            return;

        var norm = MathF.Sqrt(sumSquares);
        for (var i = 0; i < vector.Length; i++)
        {
            vector[i] /= norm;
        }
    }

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
