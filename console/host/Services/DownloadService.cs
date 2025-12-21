using LMSupply.Core.Download;
using LMSupply.Download;

namespace LMSupply.Console.Host.Services;

/// <summary>
/// Model download service (thin wrapper over HuggingFaceDownloader).
/// </summary>
public sealed class DownloadService : IDisposable
{
    private readonly HuggingFaceDownloader _downloader;
    private readonly ModelDiscoveryService _discoveryService;
    private readonly ILogger<DownloadService> _logger;
    private bool _disposed;

    public DownloadService(CacheService cacheService, ILogger<DownloadService> logger)
    {
        _downloader = new HuggingFaceDownloader(cacheService.CacheDirectory);
        _discoveryService = new ModelDiscoveryService(cacheService.CacheDirectory);
        _logger = logger;
    }

    /// <summary>
    /// Checks if a model exists on HuggingFace and returns its information.
    /// </summary>
    public async Task<ModelCheckResult> CheckModelAsync(string repoId, CancellationToken cancellationToken = default)
    {
        try
        {
            var files = await _discoveryService.ListRepositoryFilesAsync(repoId, "main", cancellationToken);

            // Detect model type based on files
            var fileNames = files.Select(f => f.FileName).ToList();
            var detectedType = CacheManager.DetectModelType(fileNames, repoId);

            // Calculate total size
            var totalSize = files.Sum(f => f.Size);

            return new ModelCheckResult
            {
                Exists = true,
                RepoId = repoId,
                DetectedType = detectedType,
                FileCount = files.Count,
                TotalSizeBytes = totalSize,
                Files = fileNames
            };
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Model check failed for {RepoId}", repoId);
            return new ModelCheckResult
            {
                Exists = false,
                RepoId = repoId,
                Error = ex.Message
            };
        }
    }

    /// <summary>
    /// Downloads a model with async progress reporting via callback.
    /// </summary>
    public async Task DownloadModelAsync(
        string repoId,
        Func<DownloadProgress, Task> onProgressAsync,
        CancellationToken cancellationToken = default)
    {
        _logger.LogInformation("Starting download: {RepoId}", repoId);

        // Use AsyncProgress with cancellation token to properly handle async callbacks
        var progress = new AsyncProgress<DownloadProgress>(onProgressAsync, cancellationToken);

        try
        {
            await _downloader.DownloadWithDiscoveryAsync(
                repoId,
                preferences: null,
                revision: "main",
                progress: progress,
                cancellationToken: cancellationToken);

            _logger.LogInformation("Download completed: {RepoId}", repoId);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Download failed: {RepoId}", repoId);
            throw;
        }
    }

    public void Dispose()
    {
        if (_disposed) return;
        _downloader.Dispose();
        _discoveryService.Dispose();
        _disposed = true;
    }
}

/// <summary>
/// Thread-safe async progress reporter that serializes async callbacks.
/// </summary>
internal sealed class AsyncProgress<T> : IProgress<T>
{
    private readonly Func<T, Task> _handler;
    private readonly SemaphoreSlim _writeLock = new(1, 1);
    private readonly CancellationToken _cancellationToken;

    public AsyncProgress(Func<T, Task> handler, CancellationToken cancellationToken = default)
    {
        _handler = handler ?? throw new ArgumentNullException(nameof(handler));
        _cancellationToken = cancellationToken;
    }

    public void Report(T value)
    {
        if (_cancellationToken.IsCancellationRequested)
            return;

        // Synchronously wait for lock then execute async handler
        // This ensures writes are serialized and not interleaved
        try
        {
            _writeLock.Wait(_cancellationToken);
            try
            {
                _handler(value).GetAwaiter().GetResult();
            }
            finally
            {
                _writeLock.Release();
            }
        }
        catch (OperationCanceledException)
        {
            // Cancelled - ignore
        }
        catch
        {
            // Swallow exceptions in progress reporting to avoid disrupting download
        }
    }
}

/// <summary>
/// Result of model validation check.
/// </summary>
public record ModelCheckResult
{
    public bool Exists { get; init; }
    public required string RepoId { get; init; }
    public ModelType DetectedType { get; init; }
    public int FileCount { get; init; }
    public long TotalSizeBytes { get; init; }
    public double TotalSizeMB => TotalSizeBytes / (1024.0 * 1024.0);
    public IReadOnlyList<string> Files { get; init; } = [];
    public string? Error { get; init; }
}
