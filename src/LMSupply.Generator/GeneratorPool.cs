using System.Collections.Concurrent;
using LMSupply.Generator.Abstractions;

namespace LMSupply.Generator;

/// <summary>
/// Pool for managing multiple generator model instances with memory protection.
/// Provides model caching and automatic resource management.
/// </summary>
public sealed class GeneratorPool : IAsyncDisposable
{
    private readonly ConcurrentDictionary<string, PooledModel> _models = new();
    private readonly SemaphoreSlim _loadLock = new(1, 1);
    private readonly IGeneratorModelFactory _factory;
    private readonly GeneratorPoolOptions _options;
    private readonly long _availableMemory;
    private long _allocatedMemory;
    private bool _disposed;

    /// <summary>
    /// Creates a new generator pool with the specified options.
    /// </summary>
    /// <param name="factory">Factory for creating generator models.</param>
    /// <param name="options">Pool configuration options.</param>
    public GeneratorPool(IGeneratorModelFactory factory, GeneratorPoolOptions? options = null)
    {
        _factory = factory ?? throw new ArgumentNullException(nameof(factory));
        _options = options ?? new GeneratorPoolOptions();

        var recommendation = HardwareDetector.GetRecommendation();
        _availableMemory = _options.MaxMemoryBytes
            ?? recommendation.GpuInfo.TotalMemoryBytes
            ?? recommendation.SystemMemoryBytes;
    }

    /// <summary>
    /// Gets the number of loaded models.
    /// </summary>
    public int LoadedModelCount => _models.Count;

    /// <summary>
    /// Gets the total allocated memory across all loaded models.
    /// </summary>
    public long AllocatedMemoryBytes => _allocatedMemory;

    /// <summary>
    /// Gets the available memory for model loading.
    /// </summary>
    public long AvailableMemoryBytes => _availableMemory - _allocatedMemory;

    /// <summary>
    /// Gets or loads a generator model by ID.
    /// </summary>
    /// <param name="modelId">The model identifier.</param>
    /// <param name="options">Optional model loading options.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>The generator model.</returns>
    /// <exception cref="InvalidOperationException">Thrown when memory is insufficient.</exception>
    public async Task<IGeneratorModel> GetOrLoadAsync(
        string modelId,
        GeneratorOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();

        // Check if already loaded
        if (_models.TryGetValue(modelId, out var pooled))
        {
            pooled.UpdateLastAccess();
            return pooled.Model;
        }

        // Acquire load lock for thread-safe loading
        await _loadLock.WaitAsync(cancellationToken);
        try
        {
            // Double-check after acquiring lock
            if (_models.TryGetValue(modelId, out pooled))
            {
                pooled.UpdateLastAccess();
                return pooled.Model;
            }

            // Calculate memory requirement
            var memoryRequired = EstimateModelMemory(modelId, options);

            // Check memory availability
            if (!CanAllocate(memoryRequired))
            {
                // Try to evict least recently used models
                await EvictModelsAsync(memoryRequired, cancellationToken);

                if (!CanAllocate(memoryRequired))
                {
                    throw new InvalidOperationException(
                        $"Insufficient memory to load model '{modelId}'. " +
                        $"Required: {memoryRequired / (1024.0 * 1024 * 1024):F2}GB, " +
                        $"Available: {AvailableMemoryBytes / (1024.0 * 1024 * 1024):F2}GB");
                }
            }

            // Load the model
            var model = await _factory.LoadAsync(modelId, options, cancellationToken);

            pooled = new PooledModel(modelId, model, memoryRequired);
            _models[modelId] = pooled;
            Interlocked.Add(ref _allocatedMemory, memoryRequired);

            return model;
        }
        finally
        {
            _loadLock.Release();
        }
    }

    /// <summary>
    /// Checks if a model is currently loaded.
    /// </summary>
    public bool IsLoaded(string modelId) => _models.ContainsKey(modelId);

    /// <summary>
    /// Unloads a specific model and releases its resources.
    /// </summary>
    public async Task UnloadAsync(string modelId, CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();

        if (_models.TryRemove(modelId, out var pooled))
        {
            Interlocked.Add(ref _allocatedMemory, -pooled.AllocatedMemory);
            await pooled.Model.DisposeAsync();
        }
    }

    /// <summary>
    /// Unloads all models and releases resources.
    /// </summary>
    public async Task UnloadAllAsync(CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();

        var models = _models.Values.ToList();
        _models.Clear();
        _allocatedMemory = 0;

        foreach (var pooled in models)
        {
            await pooled.Model.DisposeAsync();
        }
    }

    /// <summary>
    /// Gets information about all loaded models.
    /// </summary>
    public IReadOnlyList<LoadedModelInfo> GetLoadedModels() =>
        _models.Values
            .Select(p => new LoadedModelInfo(
                p.ModelId,
                p.AllocatedMemory,
                p.LoadedAt,
                p.LastAccessedAt))
            .ToList();

    private bool CanAllocate(long requiredBytes)
    {
        var withSafetyMargin = (long)(requiredBytes * (1 + _options.MemorySafetyMargin));
        return _allocatedMemory + withSafetyMargin <= _availableMemory;
    }

    private async Task EvictModelsAsync(long requiredBytes, CancellationToken cancellationToken)
    {
        // Get models sorted by last access (oldest first)
        var candidates = _models.Values
            .OrderBy(p => p.LastAccessedAt)
            .ToList();

        foreach (var candidate in candidates)
        {
            if (CanAllocate(requiredBytes))
                break;

            await UnloadAsync(candidate.ModelId, cancellationToken);
        }
    }

    private long EstimateModelMemory(string modelId, GeneratorOptions? options)
    {
        var modelInfo = ModelRegistry.GetModel(modelId);
        if (modelInfo != null)
        {
            var config = modelInfo.GetMemoryConfig(options?.MaxContextLength);
            var estimate = MemoryEstimator.Calculate(config);
            return estimate.TotalBytes;
        }

        // Default estimate for unknown models
        return MemoryEstimator.Calculate(new ModelMemoryConfig
        {
            ParameterCount = 3_000_000_000,
            NumLayers = 32,
            HiddenSize = 2560,
            ContextLength = options?.MaxContextLength ?? 4096,
            Quantization = Quantization.INT4
        }).TotalBytes;
    }

    private void ThrowIfDisposed()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
    }

    /// <inheritdoc />
    public async ValueTask DisposeAsync()
    {
        if (_disposed)
            return;

        _disposed = true;

        // Directly dispose all models without checking disposed state
        var models = _models.Values.ToList();
        _models.Clear();
        _allocatedMemory = 0;

        foreach (var pooled in models)
        {
            await pooled.Model.DisposeAsync();
        }

        _loadLock.Dispose();
    }

    private sealed class PooledModel
    {
        public string ModelId { get; }
        public IGeneratorModel Model { get; }
        public long AllocatedMemory { get; }
        public DateTime LoadedAt { get; }
        public DateTime LastAccessedAt { get; private set; }

        public PooledModel(string modelId, IGeneratorModel model, long allocatedMemory)
        {
            ModelId = modelId;
            Model = model;
            AllocatedMemory = allocatedMemory;
            LoadedAt = DateTime.UtcNow;
            LastAccessedAt = DateTime.UtcNow;
        }

        public void UpdateLastAccess() => LastAccessedAt = DateTime.UtcNow;
    }
}

/// <summary>
/// Configuration options for the generator pool.
/// </summary>
public sealed class GeneratorPoolOptions
{
    /// <summary>
    /// Maximum memory to allocate for models (in bytes).
    /// If null, uses available GPU or system memory.
    /// </summary>
    public long? MaxMemoryBytes { get; set; }

    /// <summary>
    /// Safety margin for memory calculations (0.0-1.0).
    /// Defaults to 0.2 (20% buffer).
    /// </summary>
    public double MemorySafetyMargin { get; set; } = 0.2;

    /// <summary>
    /// Maximum number of models to keep loaded simultaneously.
    /// Defaults to 2.
    /// </summary>
    public int MaxLoadedModels { get; set; } = 2;
}

/// <summary>
/// Information about a loaded model in the pool.
/// </summary>
/// <param name="ModelId">The model identifier.</param>
/// <param name="AllocatedMemoryBytes">Memory allocated for this model.</param>
/// <param name="LoadedAt">When the model was loaded.</param>
/// <param name="LastAccessedAt">When the model was last accessed.</param>
public sealed record LoadedModelInfo(
    string ModelId,
    long AllocatedMemoryBytes,
    DateTime LoadedAt,
    DateTime LastAccessedAt);
