using System.Collections.Concurrent;
using System.Diagnostics;
using LMSupply.Captioner;
using LMSupply.Detector;
using LMSupply.Embedder;
using LMSupply.Generator;
using LMSupply.Generator.Abstractions;
using LMSupply.ImageGenerator;
using LMSupply.Ocr;
using LMSupply.Reranker;
using LMSupply.Segmenter;
using LMSupply.Synthesizer;
using LMSupply.Transcriber;
using LMSupply.Translator;
using LMSupply.Console.Host.Models.Requests;
using LMSupply.Console.Host.Models.Responses;
using HostLoadedModelInfo = LMSupply.Console.Host.Models.Responses.LoadedModelInfo;

namespace LMSupply.Console.Host.Services;

/// <summary>
/// Model lifecycle management service (singleton)
/// - On-demand loading
/// - LRU caching
/// - Concurrency control
/// </summary>
public sealed partial class ModelManagerService : IAsyncDisposable
{
    private readonly ConcurrentDictionary<string, LoadedModelEntry> _loadedModels = new(StringComparer.OrdinalIgnoreCase);
    private readonly SemaphoreSlim _loadLock = new(1, 1);
    private readonly ILogger<ModelManagerService> _logger;
    private readonly CacheService _cacheService;
    private readonly SystemMonitorService _systemMonitor;
    private readonly int _maxLoadedModels;
    private readonly int _memoryPressureThreshold;
    private readonly int _defaultEstimatedMemoryMB;
    private readonly TimeSpan _idleTimeout;
    private readonly Timer _cleanupTimer;
    private bool _disposed;

    public ModelManagerService(
        IConfiguration configuration,
        CacheService cacheService,
        SystemMonitorService systemMonitor,
        ILogger<ModelManagerService> logger)
    {
        _cacheService = cacheService;
        _systemMonitor = systemMonitor;
        _logger = logger;
        _maxLoadedModels = configuration.GetValue("ModelManager:MaxLoadedModels", 20);
        _idleTimeout = TimeSpan.FromMinutes(configuration.GetValue("ModelManager:IdleTimeoutMinutes", 30));
        _memoryPressureThreshold = configuration.GetValue("ModelManager:MemoryPressureThreshold", 80);
        _defaultEstimatedMemoryMB = configuration.GetValue("ModelManager:DefaultEstimatedMemoryMB", 500);

        // Periodic cleanup timer (every 5 minutes)
        _cleanupTimer = new Timer(CleanupIdleModels, null, TimeSpan.FromMinutes(5), TimeSpan.FromMinutes(5));

        LogManagerInitialized(_logger, _maxLoadedModels, _idleTimeout);
    }

    /// <summary>
    /// Gets the Generator model (loads it if not already loaded)
    /// </summary>
    public async Task<ModelScope<IGeneratorModel>> GetGeneratorAsync(
        string modelId,
        CancellationToken cancellationToken = default)
    {
        var key = $"generator:{modelId}";
        var model = (IGeneratorModel)await GetOrLoadModelAsync(key, async () =>
        {
            LogLoadingModel(_logger, "Generator", modelId);
            var m = await LocalGenerator.LoadAsync(modelId, cancellationToken: cancellationToken);
            await m.WarmupAsync(cancellationToken);
            return m;
        }, "generator", modelId, cancellationToken);
        return new ModelScope<IGeneratorModel>(this, key, model);
    }

    /// <summary>
    /// Gets the Embedder model (loads it if not already loaded)
    /// </summary>
    public async Task<ModelScope<IEmbeddingModel>> GetEmbedderAsync(
        string modelId,
        CancellationToken cancellationToken = default)
    {
        var key = $"embedder:{modelId}";
        var model = (IEmbeddingModel)await GetOrLoadModelAsync(key, async () =>
        {
            LogLoadingModel(_logger, "Embedder", modelId);
            var m = await LocalEmbedder.LoadAsync(modelId, cancellationToken: cancellationToken);
            await m.WarmupAsync(cancellationToken);
            return m;
        }, "embedder", modelId, cancellationToken);
        return new ModelScope<IEmbeddingModel>(this, key, model);
    }

    /// <summary>
    /// Gets the Reranker model (loads it if not already loaded)
    /// </summary>
    public async Task<ModelScope<IRerankerModel>> GetRerankerAsync(
        string modelId,
        CancellationToken cancellationToken = default)
    {
        var key = $"reranker:{modelId}";
        var model = (IRerankerModel)await GetOrLoadModelAsync(key, async () =>
        {
            LogLoadingModel(_logger, "Reranker", modelId);
            var m = await LocalReranker.LoadAsync(modelId, cancellationToken: cancellationToken);
            await m.WarmupAsync(cancellationToken);
            return m;
        }, "reranker", modelId, cancellationToken);
        return new ModelScope<IRerankerModel>(this, key, model);
    }

    /// <summary>
    /// Gets the Transcriber model (loads it if not already loaded)
    /// </summary>
    public async Task<ModelScope<ITranscriberModel>> GetTranscriberAsync(
        string modelId,
        CancellationToken cancellationToken = default)
    {
        var key = $"transcriber:{modelId}";
        var model = (ITranscriberModel)await GetOrLoadModelAsync(key, async () =>
        {
            LogLoadingModel(_logger, "Transcriber", modelId);
            var m = await LocalTranscriber.LoadAsync(modelId, cancellationToken: cancellationToken);
            await m.WarmupAsync(cancellationToken);
            return m;
        }, "transcriber", modelId, cancellationToken);
        return new ModelScope<ITranscriberModel>(this, key, model);
    }

    /// <summary>
    /// Gets the Synthesizer model (loads it if not already loaded)
    /// </summary>
    public async Task<ModelScope<ISynthesizerModel>> GetSynthesizerAsync(
        string modelId,
        CancellationToken cancellationToken = default)
    {
        var key = $"synthesizer:{modelId}";
        var model = (ISynthesizerModel)await GetOrLoadModelAsync(key, async () =>
        {
            LogLoadingModel(_logger, "Synthesizer", modelId);
            var m = await LocalSynthesizer.LoadAsync(modelId, cancellationToken: cancellationToken);
            await m.WarmupAsync(cancellationToken);
            return m;
        }, "synthesizer", modelId, cancellationToken);
        return new ModelScope<ISynthesizerModel>(this, key, model);
    }

    /// <summary>
    /// Gets the Captioner model (loads it if not already loaded)
    /// </summary>
    public async Task<ModelScope<ICaptionerModel>> GetCaptionerAsync(
        string modelId,
        CancellationToken cancellationToken = default)
    {
        var key = $"captioner:{modelId}";
        var model = (ICaptionerModel)await GetOrLoadModelAsync(key, async () =>
        {
            LogLoadingModel(_logger, "Captioner", modelId);
            var m = await LocalCaptioner.LoadAsync(modelId, cancellationToken: cancellationToken);
            await m.WarmupAsync(cancellationToken);
            return m;
        }, "captioner", modelId, cancellationToken);
        return new ModelScope<ICaptionerModel>(this, key, model);
    }

    /// <summary>
    /// Gets the OCR model (loads it if not already loaded)
    /// </summary>
    public async Task<ModelScope<IOcr>> GetOcrAsync(
        string languageHint = "en",
        CancellationToken cancellationToken = default)
    {
        var key = $"ocr:{languageHint}";
        var model = (IOcr)await GetOrLoadModelAsync(key, async () =>
        {
            LogLoadingOcrModel(_logger, languageHint);
            var m = await LocalOcr.LoadForLanguageAsync(languageHint, cancellationToken: cancellationToken);
            await m.WarmupAsync(cancellationToken);
            return m;
        }, "ocr", languageHint, cancellationToken);
        return new ModelScope<IOcr>(this, key, model);
    }

    /// <summary>
    /// Gets the Detector model (loads it if not already loaded)
    /// </summary>
    public async Task<ModelScope<IDetectorModel>> GetDetectorAsync(
        string modelId,
        CancellationToken cancellationToken = default)
    {
        var key = $"detector:{modelId}";
        var model = (IDetectorModel)await GetOrLoadModelAsync(key, async () =>
        {
            LogLoadingModel(_logger, "Detector", modelId);
            var m = await LocalDetector.LoadAsync(modelId, cancellationToken: cancellationToken);
            // LocalDetector.LoadAsync already calls WarmupAsync internally
            return m;
        }, "detector", modelId, cancellationToken);
        return new ModelScope<IDetectorModel>(this, key, model);
    }

    /// <summary>
    /// Gets the Segmenter model (loads it if not already loaded)
    /// </summary>
    public async Task<ModelScope<ISegmenterModel>> GetSegmenterAsync(
        string modelId,
        CancellationToken cancellationToken = default)
    {
        var key = $"segmenter:{modelId}";
        var model = (ISegmenterModel)await GetOrLoadModelAsync(key, async () =>
        {
            LogLoadingModel(_logger, "Segmenter", modelId);
            var m = await LocalSegmenter.LoadAsync(modelId, cancellationToken: cancellationToken);
            // LocalSegmenter.LoadAsync already calls WarmupAsync internally
            return m;
        }, "segmenter", modelId, cancellationToken);
        return new ModelScope<ISegmenterModel>(this, key, model);
    }

    /// <summary>
    /// Gets the Translator model (loads it if not already loaded)
    /// </summary>
    public async Task<ModelScope<ITranslatorModel>> GetTranslatorAsync(
        string modelId,
        CancellationToken cancellationToken = default)
    {
        var key = $"translator:{modelId}";
        var model = (ITranslatorModel)await GetOrLoadModelAsync(key, async () =>
        {
            LogLoadingModel(_logger, "Translator", modelId);
            var m = await LocalTranslator.LoadAsync(modelId, cancellationToken: cancellationToken);
            // LocalTranslator.LoadAsync already calls WarmupAsync internally
            return m;
        }, "translator", modelId, cancellationToken);
        return new ModelScope<ITranslatorModel>(this, key, model);
    }

    /// <summary>
    /// Gets the ImageGenerator model (loads it if not already loaded)
    /// </summary>
    public async Task<ModelScope<IImageGeneratorModel>> GetImageGeneratorAsync(
        string modelId,
        CancellationToken cancellationToken = default)
    {
        var key = $"imagegen:{modelId}";
        var model = (IImageGeneratorModel)await GetOrLoadModelAsync(key, async () =>
        {
            LogLoadingModel(_logger, "ImageGenerator", modelId);
            var m = await LocalImageGenerator.LoadAsync(modelId, cancellationToken: cancellationToken);
            await m.WarmupAsync(cancellationToken);
            return m;
        }, "imagegen", modelId, cancellationToken);
        return new ModelScope<IImageGeneratorModel>(this, key, model);
    }

    /// <summary>
    /// Pre-load a model with explicit options.
    /// Returns the cache key (e.g., "generator:default").
    /// </summary>
    public async Task<string> LoadModelAsync(ModelLoadRequest request, CancellationToken ct = default)
    {
        var type = request.Type.ToLowerInvariant();
        var modelId = request.ModelId;
        var key = $"{type}:{modelId}";

        // If already loaded, just touch it
        if (_loadedModels.TryGetValue(key, out var existing))
        {
            existing.LastUsedAt = DateTime.UtcNow;
            return key;
        }

        // Build common options helper
        void ApplyBase(LMSupplyOptionsBase opts)
        {
            if (request.Provider is not null &&
                Enum.TryParse<ExecutionProvider>(request.Provider, ignoreCase: true, out var provider))
            {
                opts.Provider = provider;
            }
            if (request.ThreadCount is not null)
                opts.ThreadCount = request.ThreadCount;
        }

        Func<Task<IAsyncDisposable>> loadFunc = type switch
        {
            "generator" => async () =>
            {
                var opts = new GeneratorOptions();
                ApplyBase(opts);
                if (request.MaxContextLength is not null) opts.MaxContextLength = request.MaxContextLength;
                if (request.MaxConcurrentRequests is not null) opts.MaxConcurrentRequests = request.MaxConcurrentRequests.Value;
                LogLoadingModel(_logger, "Generator", modelId);
                var model = await LocalGenerator.LoadAsync(modelId, opts, cancellationToken: ct);
                await model.WarmupAsync(ct);
                return model;
            },
            "embedder" => async () =>
            {
                var opts = new EmbedderOptions();
                ApplyBase(opts);
                if (request.MaxSequenceLength is not null) opts.MaxSequenceLength = request.MaxSequenceLength.Value;
                LogLoadingModel(_logger, "Embedder", modelId);
                var model = await LocalEmbedder.LoadAsync(modelId, opts, cancellationToken: ct);
                await model.WarmupAsync(ct);
                return model;
            },
            "reranker" => async () =>
            {
                var opts = new RerankerOptions();
                ApplyBase(opts);
                LogLoadingModel(_logger, "Reranker", modelId);
                var model = await LocalReranker.LoadAsync(modelId, opts, cancellationToken: ct);
                await model.WarmupAsync(ct);
                return model;
            },
            "transcriber" => async () =>
            {
                var opts = new TranscriberOptions();
                ApplyBase(opts);
                LogLoadingModel(_logger, "Transcriber", modelId);
                var model = await LocalTranscriber.LoadAsync(modelId, opts, cancellationToken: ct);
                await model.WarmupAsync(ct);
                return model;
            },
            "synthesizer" => async () =>
            {
                var opts = new SynthesizerOptions();
                ApplyBase(opts);
                LogLoadingModel(_logger, "Synthesizer", modelId);
                var model = await LocalSynthesizer.LoadAsync(modelId, opts, cancellationToken: ct);
                await model.WarmupAsync(ct);
                return model;
            },
            "translator" => async () =>
            {
                var opts = new TranslatorOptions();
                ApplyBase(opts);
                LogLoadingModel(_logger, "Translator", modelId);
                var model = await LocalTranslator.LoadAsync(modelId, opts, cancellationToken: ct);
                return model;
            },
            "captioner" => async () =>
            {
                var opts = new CaptionerOptions();
                ApplyBase(opts);
                LogLoadingModel(_logger, "Captioner", modelId);
                var model = await LocalCaptioner.LoadAsync(modelId, opts, cancellationToken: ct);
                await model.WarmupAsync(ct);
                return model;
            },
            "ocr" => async () =>
            {
                var opts = new OcrOptions();
                ApplyBase(opts);
                LogLoadingOcrModel(_logger, modelId);
                var model = await LocalOcr.LoadForLanguageAsync(modelId, opts, cancellationToken: ct);
                await model.WarmupAsync(ct);
                return model;
            },
            "detector" => async () =>
            {
                var opts = new DetectorOptions();
                ApplyBase(opts);
                LogLoadingModel(_logger, "Detector", modelId);
                var model = await LocalDetector.LoadAsync(modelId, opts, cancellationToken: ct);
                return model;
            },
            "segmenter" => async () =>
            {
                var opts = new SegmenterOptions();
                ApplyBase(opts);
                LogLoadingModel(_logger, "Segmenter", modelId);
                var model = await LocalSegmenter.LoadAsync(modelId, opts, cancellationToken: ct);
                return model;
            },
            "imagegen" => async () =>
            {
                var opts = new ImageGeneratorOptions();
                if (request.Provider is not null &&
                    Enum.TryParse<ExecutionProvider>(request.Provider, ignoreCase: true, out var imgProvider))
                {
                    opts.Provider = imgProvider;
                }
                if (request.ThreadCount is not null) opts.ThreadCount = request.ThreadCount;
                LogLoadingModel(_logger, "ImageGenerator", modelId);
                var model = await LocalImageGenerator.LoadAsync(modelId, opts, cancellationToken: ct);
                await model.WarmupAsync(ct);
                return model;
            },
            _ => throw new ArgumentException($"Unknown model type: {request.Type}")
        };

        await GetOrLoadModelAsync(key, loadFunc, type, modelId, ct);
        return key;
    }

    /// <summary>
    /// Lists loaded models
    /// </summary>
    public IReadOnlyList<HostLoadedModelInfo> GetLoadedModels()
    {
        return _loadedModels.Values
            .Select(e => new HostLoadedModelInfo
            {
                ModelId = e.ModelId,
                ModelType = e.ModelType,
                LoadedAt = e.LoadedAt,
                LastUsedAt = e.LastUsedAt
            })
            .OrderByDescending(m => m.LastUsedAt)
            .ToList();
    }

    /// <summary>
    /// Unloads a specific model
    /// </summary>
    public async Task UnloadModelAsync(string key)
    {
        if (_loadedModels.TryRemove(key, out var entry))
        {
            LogUnloadingModel(_logger, key);
            await entry.Model.DisposeAsync();
        }
    }

    /// <summary>
    /// Unloads all models
    /// </summary>

    /// <summary>
    /// Marks a model as in-use to prevent cleanup during active operations.
    /// Call EndUse when the operation completes.
    /// </summary>
    public void BeginUse(string key)
    {
        if (_loadedModels.TryGetValue(key, out var entry))
        {
            Interlocked.Increment(ref entry.ActiveUsageCount);
            entry.LastUsedAt = DateTime.UtcNow;
        }
    }

    /// <summary>
    /// Marks a model as no longer in-use, allowing cleanup if idle.
    /// </summary>
    public void EndUse(string key)
    {
        if (_loadedModels.TryGetValue(key, out var entry))
        {
            Interlocked.Decrement(ref entry.ActiveUsageCount);
            entry.LastUsedAt = DateTime.UtcNow;
        }
    }

    public async Task UnloadAllAsync()
    {
        var entries = _loadedModels.Values.ToList();
        _loadedModels.Clear();

        foreach (var entry in entries)
        {
            await entry.Model.DisposeAsync();
        }

        LogUnloadedAllModels(_logger, entries.Count);
    }

    private async Task<IAsyncDisposable> GetOrLoadModelAsync(
        string key,
        Func<Task<IAsyncDisposable>> loadFunc,
        string modelType,
        string modelId,
        CancellationToken cancellationToken)
    {
        // Already loaded
        if (_loadedModels.TryGetValue(key, out var entry))
        {
            entry.LastUsedAt = DateTime.UtcNow;
            return entry.Model;
        }

        await _loadLock.WaitAsync(cancellationToken);
        try
        {
            // Double-check
            if (_loadedModels.TryGetValue(key, out entry))
            {
                entry.LastUsedAt = DateTime.UtcNow;
                return entry.Model;
            }

            // Evict the oldest models when over capacity
            await EnsureCapacityAsync();

            // Load the model
            var model = await loadFunc();
            var newEntry = new LoadedModelEntry
            {
                Key = key,
                Model = model,
                ModelType = modelType,
                ModelId = modelId,
                LoadedAt = DateTime.UtcNow,
                LastUsedAt = DateTime.UtcNow,
                EstimatedMemoryMB = _defaultEstimatedMemoryMB
            };

            _loadedModels[key] = newEntry;
            LogModelLoaded(_logger, key, _loadedModels.Count);

            return model;
        }
        finally
        {
            _loadLock.Release();
        }
    }

    private async Task EnsureCapacityAsync()
    {
        // Hard cap check
        while (_loadedModels.Count >= _maxLoadedModels)
        {
            if (!await EvictOldestUnusedAsync()) break;
        }

        // Memory pressure check
        var metrics = _systemMonitor.GetMemoryMetrics();
        while (metrics.UsagePercent > _memoryPressureThreshold && !_loadedModels.IsEmpty)
        {
            if (!await EvictOldestUnusedAsync()) break;
            metrics = _systemMonitor.GetMemoryMetrics();
        }
    }

    private async Task<bool> EvictOldestUnusedAsync()
    {
        var candidates = _loadedModels.Values
            .Where(e => !e.IsInUse)
            .OrderBy(e => e.LastUsedAt)
            .ThenByDescending(e => e.EstimatedMemoryMB)
            .ToList();

        foreach (var candidate in candidates)
        {
            // Atomically remove — if another thread marked it in-use, skip
            if (_loadedModels.TryRemove(candidate.Key, out var removed))
            {
                // Re-check: if it became in-use between LINQ and TryRemove, re-add it
                if (removed.IsInUse)
                {
                    _loadedModels.TryAdd(removed.Key, removed);
                    continue;
                }

                LogEvictingModel(_logger, removed.Key, removed.EstimatedMemoryMB);
                await removed.Model.DisposeAsync();
                return true;
            }
        }

        return false;
    }

    private void CleanupIdleModels(object? state)
    {
        _ = CleanupIdleModelsAsync();
    }

    private async Task CleanupIdleModelsAsync()
    {
        try
        {
            var threshold = DateTime.UtcNow - _idleTimeout;
            var idleModels = _loadedModels.Values
                .Where(e => !e.IsInUse && e.LastUsedAt < threshold)
                .ToList();

            foreach (var entry in idleModels)
            {
                try
                {
                    await UnloadModelAsync(entry.Key);
                }
                catch (Exception ex)
                {
                    Trace.TraceInformation($"[ModelManagerService] Cleanup failed for {entry.Key}: {ex.Message}");
                }
            }

            if (idleModels.Count > 0)
                LogCleanedUpIdleModels(_logger, idleModels.Count);
        }
        catch (Exception ex)
        {
            Trace.TraceInformation($"[ModelManagerService] Cleanup error: {ex.Message}");
        }
    }

    public async ValueTask DisposeAsync()
    {
        if (_disposed) return;
        _disposed = true;

        await _cleanupTimer.DisposeAsync();
        await UnloadAllAsync();
        _loadLock.Dispose();
    }

    private sealed class LoadedModelEntry
    {
        public required string Key { get; init; }
        public required IAsyncDisposable Model { get; init; }
        public required string ModelType { get; init; }
        public required string ModelId { get; init; }
        public required DateTime LoadedAt { get; init; }
        public DateTime LastUsedAt { get; set; }
        public int ActiveUsageCount;
        public double EstimatedMemoryMB { get; init; }

        public bool IsInUse => ActiveUsageCount > 0;
    }

    [LoggerMessage(Level = LogLevel.Information, Message = "ModelManager initialized: MaxModels={Max}, IdleTimeout={Timeout}")]
    private static partial void LogManagerInitialized(ILogger logger, int max, TimeSpan timeout);

    [LoggerMessage(Level = LogLevel.Information, Message = "Loading {ModelType} model: {ModelId}")]
    private static partial void LogLoadingModel(ILogger logger, string modelType, string modelId);

    [LoggerMessage(Level = LogLevel.Information, Message = "Loading OCR model for language: {Language}")]
    private static partial void LogLoadingOcrModel(ILogger logger, string language);

    [LoggerMessage(Level = LogLevel.Information, Message = "Unloading model: {Key}")]
    private static partial void LogUnloadingModel(ILogger logger, string key);

    [LoggerMessage(Level = LogLevel.Information, Message = "Unloaded all {Count} models")]
    private static partial void LogUnloadedAllModels(ILogger logger, int count);

    [LoggerMessage(Level = LogLevel.Information, Message = "Model loaded: {Key} (Total: {Count})")]
    private static partial void LogModelLoaded(ILogger logger, string key, int count);

    [LoggerMessage(Level = LogLevel.Information, Message = "Cleaned up {Count} idle models")]
    private static partial void LogCleanedUpIdleModels(ILogger logger, int count);

    [LoggerMessage(Level = LogLevel.Information, Message = "Evicting model {Key} (~{EstimatedMB:F0}MB) due to capacity/memory pressure")]
    private static partial void LogEvictingModel(ILogger logger, string key, double estimatedMB);
}
