namespace LMSupply.Console.Host.Services;

public sealed partial class TempFileService(ILogger<TempFileService> logger) : IHostedService, IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "lmsupply-images");
    private readonly Dictionary<string, (string Path, DateTime Expires)> _files = new();
    private readonly object _lock = new();
    private Timer? _timer;

    public string Save(byte[] data, string extension = "png")
    {
        Directory.CreateDirectory(_dir);
        var id = Guid.NewGuid().ToString("N");
        var path = Path.Combine(_dir, $"{id}.{extension}");
        File.WriteAllBytes(path, data);
        lock (_lock) _files[id] = (path, DateTime.UtcNow.AddHours(1));
        return id;
    }

    public bool TryGetFile(string id, out string? path)
    {
        lock (_lock)
        {
            if (_files.TryGetValue(id, out var entry) && entry.Expires > DateTime.UtcNow)
            {
                path = entry.Path;
                return true;
            }
        }
        path = null;
        return false;
    }

    public Task StartAsync(CancellationToken cancellationToken)
    {
        _timer = new Timer(Cleanup, null, TimeSpan.FromMinutes(15), TimeSpan.FromMinutes(15));
        return Task.CompletedTask;
    }

    public Task StopAsync(CancellationToken cancellationToken)
    {
        _timer?.Change(Timeout.Infinite, 0);
        return Task.CompletedTask;
    }

    private void Cleanup(object? state)
    {
        List<string> expiredPaths;
        lock (_lock)
        {
            var expired = _files
                .Where(kv => kv.Value.Expires < DateTime.UtcNow)
                .ToList();
            expiredPaths = expired.Select(kv => kv.Value.Path).ToList();
            foreach (var kv in expired) _files.Remove(kv.Key);
        }
        foreach (var path in expiredPaths)
        {
            // A file still open elsewhere, or no longer ours to delete, is left for the next sweep or the OS.
            try { File.Delete(path); } catch (Exception e) when (e is IOException or UnauthorizedAccessException) { }
        }
        LogCleanup(logger, expiredPaths.Count);
    }

    [LoggerMessage(Level = LogLevel.Debug, Message = "[TempFileService] Cleaned up {Count} expired images")]
    static partial void LogCleanup(ILogger logger, int count);

    public void Dispose() => _timer?.Dispose();
}
