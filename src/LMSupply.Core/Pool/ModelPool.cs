using System.Collections.Concurrent;
using LMSupply.Hardware;

namespace LMSupply.Pool;

/// <summary>
/// Generic pool for managing multiple model instances with LRU eviction and memory protection.
/// </summary>
/// <typeparam name="TModel">The model type.</typeparam>
/// <typeparam name="TOptions">The options type.</typeparam>
public sealed class ModelPool<TModel, TOptions> : IAsyncDisposable
    where TModel : IAsyncDisposable
    where TOptions : class
{
    private readonly ConcurrentDictionary<string, PooledModel> _models = new();
    private readonly SemaphoreSlim _loadLock = new(1, 1);
    private readonly IModelLoader<TModel, TOptions> _loader;
    private readonly ModelPoolOptions _options;
    private readonly long _availableMemory;
    private long _allocatedMemory;
    private long _accessClock;
    private bool _disposed;

    /// <summary>
    /// Creates a new model pool with the specified loader and options.
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException"><see cref="ModelPoolOptions.MaxLoadedModels"/> is less than 1.</exception>
    public ModelPool(IModelLoader<TModel, TOptions> loader, ModelPoolOptions? options = null)
    {
        _loader = loader ?? throw new ArgumentNullException(nameof(loader));
        _options = options ?? new ModelPoolOptions();
        ArgumentOutOfRangeException.ThrowIfLessThan(_options.MaxLoadedModels, 1, "options.MaxLoadedModels");

        var profile = HardwareProfile.Current;
        _availableMemory = _options.MaxMemoryBytes
            ?? profile.GpuInfo.TotalMemoryBytes
            ?? profile.SystemMemoryBytes;
    }

    /// <summary>Gets the number of loaded models.</summary>
    public int LoadedModelCount => _models.Count;

    /// <summary>Gets the total allocated memory across all loaded models.</summary>
    public long AllocatedMemoryBytes => _allocatedMemory;

    /// <summary>Gets the available memory for model loading.</summary>
    public long AvailableMemoryBytes => _availableMemory - _allocatedMemory;

    /// <summary>Gets or loads a model by ID.</summary>
    /// <exception cref="InvalidOperationException">Thrown when memory is insufficient.</exception>
    public async Task<TModel> GetOrLoadAsync(
        string modelId,
        TOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();

        if (_models.TryGetValue(modelId, out var pooled))
        {
            pooled.UpdateLastAccess(Interlocked.Increment(ref _accessClock));
            return pooled.Model;
        }

        await _loadLock.WaitAsync(cancellationToken);
        try
        {
            if (_models.TryGetValue(modelId, out pooled))
            {
                pooled.UpdateLastAccess(Interlocked.Increment(ref _accessClock));
                return pooled.Model;
            }

            // Make room by count first: a new model is one more loaded model whatever its size.
            if (_models.Count >= _options.MaxLoadedModels)
            {
                await EvictLeastRecentlyUsedAsync(() => _models.Count < _options.MaxLoadedModels, cancellationToken);
            }

            var memoryRequired = _loader.EstimateMemoryBytes(modelId, options);

            if (!CanAllocate(memoryRequired))
            {
                await EvictLeastRecentlyUsedAsync(() => CanAllocate(memoryRequired), cancellationToken);

                if (!CanAllocate(memoryRequired))
                {
                    throw new InvalidOperationException(
                        $"Insufficient memory to load model '{modelId}'. " +
                        $"Required: {memoryRequired / (1024.0 * 1024 * 1024):F2} GB, " +
                        $"Available: {AvailableMemoryBytes / (1024.0 * 1024 * 1024):F2} GB");
                }
            }

            var model = await _loader.LoadAsync(modelId, options, cancellationToken);

            pooled = new PooledModel(modelId, model, memoryRequired, Interlocked.Increment(ref _accessClock));
            _models[modelId] = pooled;
            Interlocked.Add(ref _allocatedMemory, memoryRequired);

            return model;
        }
        finally
        {
            _loadLock.Release();
        }
    }

    /// <summary>Checks if a model is currently loaded.</summary>
    public bool IsLoaded(string modelId) => _models.ContainsKey(modelId);

    /// <summary>Unloads a specific model and releases its resources.</summary>
    public async Task UnloadAsync(string modelId, CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();

        if (_models.TryRemove(modelId, out var pooled))
        {
            Interlocked.Add(ref _allocatedMemory, -pooled.AllocatedMemory);
            await pooled.Model.DisposeAsync();
        }
    }

    /// <summary>Unloads all models and releases resources.</summary>
    public async Task UnloadAllAsync(CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();

        var models = _models.Values.ToList();
        _models.Clear();
        _allocatedMemory = 0;

        foreach (var m in models)
            await m.Model.DisposeAsync();
    }

    /// <summary>Gets information about all loaded models.</summary>
    public IReadOnlyList<LoadedModelInfo> GetLoadedModels() =>
        _models.Values
            .Select(p => new LoadedModelInfo(p.ModelId, p.AllocatedMemory, p.LoadedAt, p.LastAccessedAt))
            .ToList();

    private bool CanAllocate(long requiredBytes)
    {
        var withMargin = (long)(requiredBytes * (1 + _options.MemorySafetyMargin));
        return _allocatedMemory + withMargin <= _availableMemory;
    }

    // Least recently used first, by an access counter rather than a timestamp: two accesses inside the
    // clock's resolution would otherwise tie, and the order among ties is whatever the dictionary yields.
    private async Task EvictLeastRecentlyUsedAsync(Func<bool> enough, CancellationToken cancellationToken)
    {
        var candidates = _models.Values.OrderBy(p => p.AccessOrder).ToList();
        foreach (var candidate in candidates)
        {
            if (enough())
                break;
            await UnloadAsync(candidate.ModelId, cancellationToken);
        }
    }

    private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(_disposed, this);

    /// <inheritdoc />
    public async ValueTask DisposeAsync()
    {
        if (_disposed) return;
        _disposed = true;

        var models = _models.Values.ToList();
        _models.Clear();
        _allocatedMemory = 0;

        foreach (var m in models)
            await m.Model.DisposeAsync();

        _loadLock.Dispose();
    }

    private sealed class PooledModel
    {
        public string ModelId { get; }
        public TModel Model { get; }
        public long AllocatedMemory { get; }
        public DateTime LoadedAt { get; }
        public DateTime LastAccessedAt { get; private set; }
        public long AccessOrder { get; private set; }

        public PooledModel(string modelId, TModel model, long allocatedMemory, long accessOrder)
        {
            ModelId = modelId;
            Model = model;
            AllocatedMemory = allocatedMemory;
            LoadedAt = DateTime.UtcNow;
            LastAccessedAt = DateTime.UtcNow;
            AccessOrder = accessOrder;
        }

        public void UpdateLastAccess(long accessOrder)
        {
            LastAccessedAt = DateTime.UtcNow;
            AccessOrder = accessOrder;
        }
    }
}
