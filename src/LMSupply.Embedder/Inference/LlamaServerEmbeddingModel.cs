using LMSupply.Llama.Gguf;
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
    // The server lease, replaced with a new server (same configuration) when the held one has exited.
    private readonly RestartingLease<ServerLease> _lease;
    private readonly EmbedderOptions _options;
    private readonly string _modelPath;
    private bool _disposed;

    private readonly PromptPrefixes _prompts;

    private LlamaServerEmbeddingModel(
        string modelId,
        string modelPath,
        ServerLease serverLease,
        Func<CancellationToken, Task<ServerLease>> relet,
        int dimensions,
        EmbedderOptions options,
        VectorSpaceDescriptor vectorSpace,
        PromptPrefixes prompts)
    {
        _prompts = prompts;
        ModelId = modelId;
        VectorSpaceRevision = vectorSpace.Revision;
        RetrievalVectorSpaceRevision = vectorSpace.RetrievalRevision;
        _modelPath = modelPath;
        _lease = RestartingServerLease.Create(serverLease, relet, $"LlamaServerEmbeddingModel '{modelId}'");
        Dimensions = dimensions;
        _options = options;
    }

    /// <summary>
    /// Loads a GGUF embedding model using llama-server.
    /// </summary>
    public static Task<LlamaServerEmbeddingModel> LoadAsync(
        string modelId,
        string modelPath,
        EmbedderOptions options,
        IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
        => LoadAsync(modelId, modelPath, options, PromptPrefixes.None, progress, cancellationToken);

    /// <summary>
    /// Loads a GGUF embedding model whose prompt prefixes are known (from the catalog entry of the model the GGUF file
    /// was converted from). <see cref="IEmbeddingModel.EmbedAsync(string, CancellationToken)"/>,
    /// <see cref="IEmbeddingModel.EmbedQueryAsync(string, CancellationToken)"/> and
    /// <see cref="IEmbeddingModel.EmbedPassageAsync(string, CancellationToken)"/> apply them, as they do on the ONNX path.
    /// </summary>
    internal static async Task<LlamaServerEmbeddingModel> LoadAsync(
        string modelId,
        string modelPath,
        EmbedderOptions options,
        PromptPrefixes prompts,
        IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        // 1. Get llama-server via update service
        progress?.Report(new DownloadProgress
        {
            FileName = "llama-server",
            Kind = DownloadKind.Runtime,
            Phase = DownloadPhase.Preparing
        });

        var preferredBackend = global::LMSupply.Llama.LlamaBackendSelector.MapProvider(
            options.Provider, Hardware.HardwareProfile.For(options.Provider).GpuInfo);
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

        // 2. Configure server for embedding mode. The pool reports Loading when it starts a server.
        // Pooling and length come from the file when the options leave them unset: a GGUF embedder records the pooling
        // it was converted with (<arch>.pooling_type) and the length it was trained to (<arch>.context_length). The
        // previous defaults - mean pooling, 512 tokens - turned a CLS model such as bge-m3 into a different vector space
        // and rejected any passage longer than 512 tokens.
        var metadata = await GgufMetadataReader.ReadAsync(modelPath, false, cancellationToken);
        options.PoolingMode ??= PoolingFromFile(metadata?.PoolingType) ?? PoolingMode.Mean;
        options.MaxSequenceLength ??= metadata?.ContextLength is { } trained
            ? Math.Min(trained, MaxAutomaticContextLength)
            : EmbedderOptions.DefaultMaxSequenceLength;
        var poolingType = ServerPooling(options.PoolingMode.Value);

        var serverConfig = new LlamaServerConfig
        {
            ModelPath = modelPath,
            Port = 0, // Auto-assign
            ContextSize = options.MaxSequenceLength.Value,
            GpuLayers = backend == LlamaServerBackend.Cpu ? 0 : -1,
            // An embedding model reads each input in one physical batch, so both batch sizes hold the whole context;
            // with 512 an input longer than 512 tokens was refused.
            BatchSize = options.MaxSequenceLength.Value,
            UBatchSize = options.MaxSequenceLength.Value,
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

            if (ex is OperationCanceledException && cancellationToken.IsCancellationRequested)
            {
                throw;
            }

            throw new InvalidOperationException(
                $"Failed to determine embedding dimensions: {ex.Message}", ex);
        }

        // What the server was actually asked to do — not the option as requested (Max is sent as Last).
        // Tokenization happens inside llama-server and is not visible here; the server binary's version
        // is deliberately outside the revision (see VectorSpaceDescriptor).
        var vectorSpace = new VectorSpaceDescriptor(
            Backend: "gguf",
            ModelFile: Path.GetFileName(modelPath),
            Tokenizer: "llama-server",
            Pooling: poolingType.ToString(),
            Normalize: options.NormalizeEmbeddings,
            MaxSequenceLength: options.MaxSequenceLength.Value,
            Dimensions: dimensions,
            QueryPrefix: prompts.Query,
            PassagePrefix: prompts.Passage,
            DefaultPrefix: prompts.Default);
        System.Diagnostics.Trace.TraceInformation($"[LocalEmbedder.vectorspace] {modelId}: {vectorSpace.Canonical} -> {vectorSpace.Revision}");

        return new LlamaServerEmbeddingModel(
            modelId,
            modelPath,
            serverLease,
            ct => LlamaServerPool.Instance.LeaseAsync(serverPath, serverConfig, backend, null, ct),
            dimensions,
            options,
            vectorSpace,
            prompts);
    }

    /// <summary>
    /// The longest context taken from a file's trained length when the options set none. Embedders trained to 32K
    /// tokens exist; a context that size costs memory most passages never use, and 8,192 covers the common long-context
    /// embedders (bge-m3). Set <see cref="EmbedderOptions.MaxSequenceLength"/> to go higher.
    /// </summary>
    internal const int MaxAutomaticContextLength = 8192;

    /// <summary>The pooling a GGUF <c>pooling_type</c> value names, or null for none, rank or a value it does not know.</summary>
    internal static PoolingMode? PoolingFromFile(int? poolingType) => poolingType switch
    {
        1 => PoolingMode.Mean,
        2 => PoolingMode.Cls,
        3 => PoolingMode.Last,
        _ => null,
    };

    /// <summary>
    /// The llama-server pooling for an embedder pooling mode. llama-server has no max pooling; asking for it is refused
    /// rather than sent as another pooling (it used to be sent as last-token, a different vector space without a word).
    /// </summary>
    internal static PoolingType ServerPooling(PoolingMode mode) => mode switch
    {
        PoolingMode.Mean => PoolingType.Mean,
        PoolingMode.Cls => PoolingType.Cls,
        PoolingMode.Last => PoolingType.Last,
        _ => throw new NotSupportedException(
            $"PoolingMode.{mode} is not available for GGUF embedders: llama-server pools by mean, CLS or last token."),
    };

    /// <inheritdoc />
    public string ModelId { get; }

    /// <inheritdoc />
    public int Dimensions { get; }

    /// <inheritdoc />
    public string VectorSpaceRevision { get; }

    /// <inheritdoc />
    public string RetrievalVectorSpaceRevision { get; }

    /// <inheritdoc />
    public long? EstimatedMemoryBytes => File.Exists(_modelPath) ? new FileInfo(_modelPath).Length * 2 : null;

    /// <inheritdoc />
    public bool IsGpuActive => _lease.Current.Backend != LlamaServerBackend.Cpu;

    /// <inheritdoc />
    public IReadOnlyList<string> ActiveProviders => IsGpuActive
        ? [$"llama-server-{_lease.Current.Backend}", "CPU"]
        : ["llama-server-CPU"];

    /// <inheritdoc />
    public ExecutionProvider RequestedProvider => _options.Provider;

    /// <inheritdoc />
    public async ValueTask<float[]> EmbedRawAsync(string text, CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();

        var embedding = await (await _lease.GetAsync(cancellationToken).ConfigureAwait(false)).Client.GenerateEmbeddingAsync(text, cancellationToken);

        if (_options.NormalizeEmbeddings)
        {
            NormalizeVector(embedding);
        }

        return embedding;
    }

    /// <inheritdoc />
    public async ValueTask<float[][]> EmbedRawAsync(IReadOnlyList<string> texts, CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();

        var embeddings = await (await _lease.GetAsync(cancellationToken).ConfigureAwait(false)).Client.GenerateEmbeddingsBatchAsync(texts, cancellationToken);

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
    public async ValueTask<float[]> EmbedRawAsync(
        string text,
        int dimensions,
        CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();
        if (dimensions <= 0 || dimensions > Dimensions)
            throw new ArgumentOutOfRangeException(
                nameof(dimensions), $"dimensions must be between 1 and {Dimensions}.");

        var full = await EmbedRawAsync(text, cancellationToken);
        if (dimensions == Dimensions)
            return full;

        var truncated = full[..dimensions];
        NormalizeVector(truncated);
        return truncated;
    }

    /// <inheritdoc />
    public async ValueTask<float[][]> EmbedRawAsync(
        IReadOnlyList<string> texts,
        int dimensions,
        CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();
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
            NormalizeVector(result[i]);
        }
        return result;
    }

    /// <inheritdoc />
    public async Task WarmupAsync(CancellationToken cancellationToken = default)
    {
        // Server is already warmed up during LoadAsync (test embedding)
        // Perform another embedding to ensure everything is ready
        await EmbedRawAsync("warmup", cancellationToken);
    }

    /// <inheritdoc />
    public EmbedderModelInfo? GetModelInfo() => new()
    {
        RepoId = ModelId,
        Dimensions = Dimensions,
        MaxSequenceLength = _options.MaxSequenceLength ?? EmbedderOptions.DefaultMaxSequenceLength,
        PoolingMode = _options.PoolingMode ?? PoolingMode.Mean,
        DoLowerCase = _options.DoLowerCase,
        QueryPrefix = _prompts.Query,
        PassagePrefix = _prompts.Passage,
        DefaultPrefix = _prompts.Default,
        Description = $"GGUF embedding model via llama-server-{_lease.Current.Backend}"
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
        await _lease.DisposeAsync();
    }
}
