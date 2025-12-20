using System.Collections.Concurrent;
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
using LMSupply.Console.Host.Models.Responses;
using HostLoadedModelInfo = LMSupply.Console.Host.Models.Responses.LoadedModelInfo;

namespace LMSupply.Console.Host.Services;

/// <summary>
/// 모델 생명주기 관리 서비스 (싱글톤)
/// - On-Demand 로딩
/// - LRU 캐싱
/// - 동시성 제어
/// </summary>
public sealed class ModelManagerService : IAsyncDisposable
{
    private readonly ConcurrentDictionary<string, LoadedModelEntry> _loadedModels = new();
    private readonly SemaphoreSlim _loadLock = new(1, 1);
    private readonly ILogger<ModelManagerService> _logger;
    private readonly CacheService _cacheService;
    private readonly int _maxLoadedModels;
    private readonly TimeSpan _idleTimeout;
    private readonly Timer _cleanupTimer;
    private bool _disposed;

    public ModelManagerService(
        IConfiguration configuration,
        CacheService cacheService,
        ILogger<ModelManagerService> logger)
    {
        _cacheService = cacheService;
        _logger = logger;
        _maxLoadedModels = configuration.GetValue("ModelManager:MaxLoadedModels", 3);
        _idleTimeout = TimeSpan.FromMinutes(configuration.GetValue("ModelManager:IdleTimeoutMinutes", 30));

        // 주기적 정리 타이머 (5분마다)
        _cleanupTimer = new Timer(CleanupIdleModels, null, TimeSpan.FromMinutes(5), TimeSpan.FromMinutes(5));

        _logger.LogInformation("ModelManager initialized: MaxModels={Max}, IdleTimeout={Timeout}",
            _maxLoadedModels, _idleTimeout);
    }

    /// <summary>
    /// Generator 모델 조회 (없으면 로드)
    /// </summary>
    public async Task<IGeneratorModel> GetGeneratorAsync(
        string modelId,
        CancellationToken cancellationToken = default)
    {
        var key = $"generator:{modelId}";
        return (IGeneratorModel)await GetOrLoadModelAsync(key, async () =>
        {
            _logger.LogInformation("Loading Generator model: {ModelId}", modelId);
            var model = await LocalGenerator.LoadAsync(modelId, cancellationToken: cancellationToken);
            await model.WarmupAsync(cancellationToken);
            return model;
        }, "generator", modelId, cancellationToken);
    }

    /// <summary>
    /// Embedder 모델 조회 (없으면 로드)
    /// </summary>
    public async Task<IEmbeddingModel> GetEmbedderAsync(
        string modelId,
        CancellationToken cancellationToken = default)
    {
        var key = $"embedder:{modelId}";
        return (IEmbeddingModel)await GetOrLoadModelAsync(key, async () =>
        {
            _logger.LogInformation("Loading Embedder model: {ModelId}", modelId);
            var model = await LocalEmbedder.LoadAsync(modelId, cancellationToken: cancellationToken);
            await model.WarmupAsync(cancellationToken);
            return model;
        }, "embedder", modelId, cancellationToken);
    }

    /// <summary>
    /// Reranker 모델 조회 (없으면 로드)
    /// </summary>
    public async Task<IRerankerModel> GetRerankerAsync(
        string modelId,
        CancellationToken cancellationToken = default)
    {
        var key = $"reranker:{modelId}";
        return (IRerankerModel)await GetOrLoadModelAsync(key, async () =>
        {
            _logger.LogInformation("Loading Reranker model: {ModelId}", modelId);
            var model = await LocalReranker.LoadAsync(modelId, cancellationToken: cancellationToken);
            await model.WarmupAsync(cancellationToken);
            return model;
        }, "reranker", modelId, cancellationToken);
    }

    /// <summary>
    /// Transcriber 모델 조회 (없으면 로드)
    /// </summary>
    public async Task<ITranscriberModel> GetTranscriberAsync(
        string modelId,
        CancellationToken cancellationToken = default)
    {
        var key = $"transcriber:{modelId}";
        return (ITranscriberModel)await GetOrLoadModelAsync(key, async () =>
        {
            _logger.LogInformation("Loading Transcriber model: {ModelId}", modelId);
            var model = await LocalTranscriber.LoadAsync(modelId, cancellationToken: cancellationToken);
            await model.WarmupAsync(cancellationToken);
            return model;
        }, "transcriber", modelId, cancellationToken);
    }

    /// <summary>
    /// Synthesizer 모델 조회 (없으면 로드)
    /// </summary>
    public async Task<ISynthesizerModel> GetSynthesizerAsync(
        string modelId,
        CancellationToken cancellationToken = default)
    {
        var key = $"synthesizer:{modelId}";
        return (ISynthesizerModel)await GetOrLoadModelAsync(key, async () =>
        {
            _logger.LogInformation("Loading Synthesizer model: {ModelId}", modelId);
            var model = await LocalSynthesizer.LoadAsync(modelId, cancellationToken: cancellationToken);
            await model.WarmupAsync(cancellationToken);
            return model;
        }, "synthesizer", modelId, cancellationToken);
    }

    /// <summary>
    /// Captioner 모델 조회 (없으면 로드)
    /// </summary>
    public async Task<ICaptionerModel> GetCaptionerAsync(
        string modelId,
        CancellationToken cancellationToken = default)
    {
        var key = $"captioner:{modelId}";
        return (ICaptionerModel)await GetOrLoadModelAsync(key, async () =>
        {
            _logger.LogInformation("Loading Captioner model: {ModelId}", modelId);
            var model = await LocalCaptioner.LoadAsync(modelId, cancellationToken: cancellationToken);
            await model.WarmupAsync(cancellationToken);
            return model;
        }, "captioner", modelId, cancellationToken);
    }

    /// <summary>
    /// OCR 모델 조회 (없으면 로드)
    /// </summary>
    public async Task<IOcr> GetOcrAsync(
        string languageHint = "en",
        CancellationToken cancellationToken = default)
    {
        var key = $"ocr:{languageHint}";
        return (IOcr)await GetOrLoadModelAsync(key, async () =>
        {
            _logger.LogInformation("Loading OCR model for language: {Language}", languageHint);
            var model = await LocalOcr.LoadForLanguageAsync(languageHint, cancellationToken: cancellationToken);
            await model.WarmupAsync(cancellationToken);
            return model;
        }, "ocr", languageHint, cancellationToken);
    }

    /// <summary>
    /// Detector 모델 조회 (없으면 로드)
    /// </summary>
    public async Task<IDetectorModel> GetDetectorAsync(
        string modelId,
        CancellationToken cancellationToken = default)
    {
        var key = $"detector:{modelId}";
        return (IDetectorModel)await GetOrLoadModelAsync(key, async () =>
        {
            _logger.LogInformation("Loading Detector model: {ModelId}", modelId);
            var model = await LocalDetector.LoadAsync(modelId, cancellationToken: cancellationToken);
            // LocalDetector.LoadAsync already calls WarmupAsync internally
            return model;
        }, "detector", modelId, cancellationToken);
    }

    /// <summary>
    /// Segmenter 모델 조회 (없으면 로드)
    /// </summary>
    public async Task<ISegmenterModel> GetSegmenterAsync(
        string modelId,
        CancellationToken cancellationToken = default)
    {
        var key = $"segmenter:{modelId}";
        return (ISegmenterModel)await GetOrLoadModelAsync(key, async () =>
        {
            _logger.LogInformation("Loading Segmenter model: {ModelId}", modelId);
            var model = await LocalSegmenter.LoadAsync(modelId, cancellationToken: cancellationToken);
            // LocalSegmenter.LoadAsync already calls WarmupAsync internally
            return model;
        }, "segmenter", modelId, cancellationToken);
    }

    /// <summary>
    /// Translator 모델 조회 (없으면 로드)
    /// </summary>
    public async Task<ITranslatorModel> GetTranslatorAsync(
        string modelId,
        CancellationToken cancellationToken = default)
    {
        var key = $"translator:{modelId}";
        return (ITranslatorModel)await GetOrLoadModelAsync(key, async () =>
        {
            _logger.LogInformation("Loading Translator model: {ModelId}", modelId);
            var model = await LocalTranslator.LoadAsync(modelId, cancellationToken: cancellationToken);
            // LocalTranslator.LoadAsync already calls WarmupAsync internally
            return model;
        }, "translator", modelId, cancellationToken);
    }

    /// <summary>
    /// ImageGenerator 모델 조회 (없으면 로드)
    /// </summary>
    public async Task<IImageGeneratorModel> GetImageGeneratorAsync(
        string modelId,
        CancellationToken cancellationToken = default)
    {
        var key = $"imagegen:{modelId}";
        return (IImageGeneratorModel)await GetOrLoadModelAsync(key, async () =>
        {
            _logger.LogInformation("Loading ImageGenerator model: {ModelId}", modelId);
            var model = await LocalImageGenerator.LoadAsync(modelId, cancellationToken: cancellationToken);
            await model.WarmupAsync(cancellationToken);
            return model;
        }, "imagegen", modelId, cancellationToken);
    }

    /// <summary>
    /// 로드된 모델 목록
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
    /// 특정 모델 언로드
    /// </summary>
    public async Task UnloadModelAsync(string key)
    {
        if (_loadedModels.TryRemove(key, out var entry))
        {
            _logger.LogInformation("Unloading model: {Key}", key);
            await entry.Model.DisposeAsync();
        }
    }

    /// <summary>
    /// 모든 모델 언로드
    /// </summary>
    public async Task UnloadAllAsync()
    {
        var entries = _loadedModels.Values.ToList();
        _loadedModels.Clear();

        foreach (var entry in entries)
        {
            await entry.Model.DisposeAsync();
        }

        _logger.LogInformation("Unloaded all {Count} models", entries.Count);
    }

    private async Task<IAsyncDisposable> GetOrLoadModelAsync(
        string key,
        Func<Task<IAsyncDisposable>> loadFunc,
        string modelType,
        string modelId,
        CancellationToken cancellationToken)
    {
        // 이미 로드된 경우
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

            // 용량 초과 시 가장 오래된 모델 제거
            await EnsureCapacityAsync();

            // 모델 로드
            var model = await loadFunc();
            var newEntry = new LoadedModelEntry
            {
                Key = key,
                Model = model,
                ModelType = modelType,
                ModelId = modelId,
                LoadedAt = DateTime.UtcNow,
                LastUsedAt = DateTime.UtcNow
            };

            _loadedModels[key] = newEntry;
            _logger.LogInformation("Model loaded: {Key} (Total: {Count})", key, _loadedModels.Count);

            return model;
        }
        finally
        {
            _loadLock.Release();
        }
    }

    private async Task EnsureCapacityAsync()
    {
        while (_loadedModels.Count >= _maxLoadedModels)
        {
            // LRU: 가장 오래 사용하지 않은 모델 제거
            var oldest = _loadedModels.Values
                .OrderBy(e => e.LastUsedAt)
                .FirstOrDefault();

            if (oldest != null)
            {
                await UnloadModelAsync(oldest.Key);
            }
            else
            {
                break;
            }
        }
    }

    private void CleanupIdleModels(object? state)
    {
        var threshold = DateTime.UtcNow - _idleTimeout;
        var idleModels = _loadedModels.Values
            .Where(e => e.LastUsedAt < threshold)
            .ToList();

        foreach (var entry in idleModels)
        {
            _ = UnloadModelAsync(entry.Key);
        }

        if (idleModels.Count > 0)
        {
            _logger.LogInformation("Cleaned up {Count} idle models", idleModels.Count);
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
    }
}
