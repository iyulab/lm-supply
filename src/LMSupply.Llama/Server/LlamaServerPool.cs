using System.Collections.Concurrent;
using System.Diagnostics;
using LMSupply.Download;

namespace LMSupply.Llama.Server;

/// <summary>
/// Pool for managing llama-server instances.
/// Reuses servers for the same model/backend/context configuration.
/// Automatically cleans up on process exit.
/// </summary>
public sealed class LlamaServerPool : IAsyncDisposable
{
    private static readonly Lazy<LlamaServerPool> _instance = new(
        () =>
        {
            var pool = new LlamaServerPool();
            RegisterForCleanup(pool);
            return pool;
        },
        LazyThreadSafetyMode.ExecutionAndPublication);

    private static bool _cleanupRegistered;

    /// <summary>
    /// Gets the singleton instance.
    /// </summary>
    public static LlamaServerPool Instance => _instance.Value;

    private readonly ConcurrentDictionary<string, PooledServer> _servers = new();
    private readonly SemaphoreSlim _createLock = new(1, 1);
    private readonly Timer _cleanupTimer;
    private bool _disposed;

    private static void RegisterForCleanup(LlamaServerPool pool)
    {
        if (_cleanupRegistered)
            return;

        _cleanupRegistered = true;

        // Register for process exit cleanup
        AppDomain.CurrentDomain.ProcessExit += (_, _) =>
        {
            // Synchronous cleanup on process exit
            pool.DisposeAsync().AsTask().GetAwaiter().GetResult();
        };

        // Also register for console cancel (Ctrl+C)
        try
        {
            Console.CancelKeyPress += (_, e) =>
            {
                // Allow cancellation to proceed after cleanup
                e.Cancel = false;
                pool.DisposeAsync().AsTask().GetAwaiter().GetResult();
            };
        }
        catch (Exception ex)
        {
            Trace.TraceInformation($"[LlamaServerPool] Console handler registration failed: {ex.Message}");
        }
    }

    /// <summary>
    /// Disposes the singleton instance and all pooled servers.
    /// Call this for explicit cleanup before application exit.
    /// </summary>
    public static async ValueTask DisposeInstanceAsync()
    {
        if (_instance.IsValueCreated)
        {
            await _instance.Value.DisposeAsync();
        }
    }

    /// <summary>
    /// Options for the server pool.
    /// </summary>
    public LlamaServerPoolOptions Options { get; }

    /// <summary>
    /// Creates a new server pool with default options.
    /// </summary>
    public LlamaServerPool() : this(new LlamaServerPoolOptions())
    {
    }

    /// <summary>
    /// Creates a new server pool with custom options.
    /// </summary>
    public LlamaServerPool(LlamaServerPoolOptions options)
    {
        Options = options;

        // Start cleanup timer
        _cleanupTimer = new Timer(
            CleanupIdleServers,
            null,
            options.CleanupInterval,
            options.CleanupInterval);
    }

    /// <summary>
    /// Leases a server for the specified configuration.
    /// Returns an existing server if available, or creates a new one.
    /// </summary>
    public async Task<ServerLease> LeaseAsync(
        string serverPath,
        LlamaServerConfig config,
        LlamaServerBackend backend,
        IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        var key = MakeKey(config.ModelPath, backend, config.ContextSize, config.Mode);

        // Try to get existing server
        if (_servers.TryGetValue(key, out var pooledServer))
        {
            if (pooledServer.TryLease())
            {
                return new ServerLease(pooledServer, this);
            }

            // Server is busy or dead, try to create new one
        }

        // Create new server
        await _createLock.WaitAsync(cancellationToken);
        try
        {
            // Double-check after acquiring lock
            if (_servers.TryGetValue(key, out pooledServer))
            {
                if (pooledServer.TryLease())
                {
                    return new ServerLease(pooledServer, this);
                }
            }

            // Check server limit
            var activeCount = _servers.Values.Count(s => s.IsAlive);
            if (activeCount >= Options.MaxServers)
            {
                // Evict oldest idle server
                await EvictOldestIdleServerAsync();
            }

            // Start new server
            progress?.Report(new DownloadProgress
            {
                FileName = Path.GetFileName(config.ModelPath),
                Phase = DownloadPhase.Extracting
            });

            var serverProcess = await LlamaServerProcess.StartAsync(
                serverPath,
                config,
                backend,
                cancellationToken);

            var client = new LlamaServerClient(
                serverProcess.Info!.BaseUrl,
                maxContextLength: config.ContextSize,
                requestTimeout: config.RequestTimeout);

            var newPooledServer = new PooledServer(key, serverProcess, client, config.ModelPath, backend);
            newPooledServer.TryLease();

            _servers[key] = newPooledServer;

            return new ServerLease(newPooledServer, this);
        }
        finally
        {
            _createLock.Release();
        }
    }

    /// <summary>
    /// Returns a server to the pool.
    /// </summary>
    internal static void Release(PooledServer server)
    {
        server.Release();
    }

    /// <summary>
    /// Gets the current pool status.
    /// </summary>
    public PoolStatus GetStatus()
    {
        var servers = _servers.Values.ToList();

        return new PoolStatus
        {
            TotalServers = servers.Count,
            ActiveServers = servers.Count(s => s.IsInUse),
            IdleServers = servers.Count(s => s.IsAlive && !s.IsInUse),
            Entries = servers.Select(s => new PoolEntry
            {
                Key = s.Key,
                ModelPath = s.ModelPath,
                Backend = s.Backend,
                IsInUse = s.IsInUse,
                LastUsed = s.LastUsed,
                ProcessId = s.Server.Info?.ProcessId ?? 0
            }).ToList()
        };
    }

    /// <summary>
    /// Stops every pooled server that no model is using, now rather than after
    /// <see cref="LlamaServerPoolOptions.IdleTimeout"/>, and returns how many were stopped.
    /// </summary>
    /// <remarks>
    /// A server stays in the pool after the last model using it is disposed so that loading the same
    /// model again is fast. On a single GPU that idle server keeps its memory, and the next model to
    /// load gets less. A host that switches models calls this after disposing the old one. Loading a
    /// generator on a GPU already stops the idle GPU servers of other models before it measures memory.
    /// A server a model is using is never stopped.
    /// </remarks>
    public Task<int> ReleaseIdleAsync() => ReleaseIdleAsync(static _ => true);

    /// <summary>
    /// <see cref="ReleaseIdleAsync()"/> for the idle servers <paramref name="filter"/> selects.
    /// </summary>
    internal async Task<int> ReleaseIdleAsync(Func<PooledServer, bool> filter)
    {
        var released = 0;
        foreach (var server in _servers.Values.Where(filter).ToList())
        {
            if (await TryRetireAsync(server).ConfigureAwait(false))
                released++;
        }
        return released;
    }

    /// <summary>
    /// Removes and stops <paramref name="server"/> if no lease holds it. The retire is atomic with
    /// <see cref="PooledServer.TryLease"/>: a lease taken concurrently either wins (and the server stays)
    /// or fails (and the caller starts a new server) — a leased server is never stopped under its model.
    /// </summary>
    private async Task<bool> TryRetireAsync(PooledServer server)
    {
        if (!server.TryRetire())
            return false;

        _servers.TryRemove(new KeyValuePair<string, PooledServer>(server.Key, server));
        await server.DisposeAsync().ConfigureAwait(false);
        return true;
    }

    private async Task EvictOldestIdleServerAsync()
    {
        foreach (var candidate in _servers.Values
            .Where(s => s.IsAlive && !s.IsInUse)
            .OrderBy(s => s.LastUsed)
            .ToList())
        {
            if (await TryRetireAsync(candidate))
                return;
        }
    }

    private async void CleanupIdleServers(object? state)
    {
        if (_disposed)
            return;

        var now = DateTimeOffset.UtcNow;
        await ReleaseIdleAsync(s => (now - s.LastUsed) > Options.IdleTimeout).ConfigureAwait(false);

        // Also remove dead servers
        var deadServers = _servers.Values.Where(s => !s.IsAlive).ToList();
        foreach (var server in deadServers)
        {
            if (_servers.TryRemove(new KeyValuePair<string, PooledServer>(server.Key, server)))
            {
                await server.DisposeAsync().ConfigureAwait(false);
            }
        }
    }

    private static string MakeKey(string modelPath, LlamaServerBackend backend, int contextSize, ServerMode mode = ServerMode.Generation)
        => $"{modelPath}|{backend}|{contextSize}|{mode}";

    public async ValueTask DisposeAsync()
    {
        if (_disposed)
            return;

        _disposed = true;

        await _cleanupTimer.DisposeAsync();
        _createLock.Dispose();

        foreach (var server in _servers.Values)
        {
            await server.DisposeAsync();
        }

        _servers.Clear();
    }
}

/// <summary>
/// Options for the server pool.
/// </summary>
public sealed class LlamaServerPoolOptions
{
    /// <summary>
    /// Maximum number of servers to keep in the pool.
    /// Default: 3.
    /// </summary>
    public int MaxServers { get; set; } = 3;

    /// <summary>
    /// Time after which an idle server is removed from the pool.
    /// Default: 10 minutes.
    /// </summary>
    public TimeSpan IdleTimeout { get; set; } = TimeSpan.FromMinutes(10);

    /// <summary>
    /// Interval for cleanup of idle servers.
    /// Default: 1 minute.
    /// </summary>
    public TimeSpan CleanupInterval { get; set; } = TimeSpan.FromMinutes(1);
}

/// <summary>
/// Counts the leases on a pooled server and retires it atomically: once <see cref="TryRetire"/> has
/// succeeded no lease can be acquired, and it succeeds only while no lease is held. Without that, a
/// server the idle sweep had picked could be leased in the gap before it was stopped.
/// </summary>
internal sealed class LeaseGate
{
    // Number of live leases; -1 once retired.
    private int _count;

    public bool IsHeld => Volatile.Read(ref _count) > 0;

    public bool IsRetired => Volatile.Read(ref _count) < 0;

    public bool TryAcquire()
    {
        while (true)
        {
            var count = Volatile.Read(ref _count);
            if (count < 0)
                return false;

            if (Interlocked.CompareExchange(ref _count, count + 1, count) == count)
                return true;
        }
    }

    public void Release()
    {
        if (Interlocked.Decrement(ref _count) < 0)
            throw new InvalidOperationException("A lease was released that was never acquired.");
    }

    public bool TryRetire() => Interlocked.CompareExchange(ref _count, -1, 0) == 0;
}

/// <summary>
/// A pooled llama-server instance.
/// </summary>
internal sealed class PooledServer : IAsyncDisposable
{
    private readonly LeaseGate _leases = new();
    private bool _disposed;

    public string Key { get; }
    public LlamaServerProcess Server { get; }
    public LlamaServerClient Client { get; }
    public string ModelPath { get; }
    public LlamaServerBackend Backend { get; }
    public DateTimeOffset LastUsed { get; private set; }

    public bool IsAlive => !_disposed && Server.IsRunning;
    public bool IsInUse => _leases.IsHeld;

    public PooledServer(
        string key,
        LlamaServerProcess server,
        LlamaServerClient client,
        string modelPath,
        LlamaServerBackend backend)
    {
        Key = key;
        Server = server;
        Client = client;
        ModelPath = modelPath;
        Backend = backend;
        LastUsed = DateTimeOffset.UtcNow;
    }

    public bool TryLease()
    {
        if (_disposed || !Server.IsRunning)
            return false;

        if (!_leases.TryAcquire())
            return false; // retired: the pool is stopping it

        LastUsed = DateTimeOffset.UtcNow;
        return true;
    }

    /// <summary>
    /// Marks the server retired when no lease holds it, so no lease can be taken afterwards;
    /// <see langword="false"/> when it is in use (or already retired).
    /// </summary>
    public bool TryRetire() => _leases.TryRetire();

    public void Release()
    {
        _leases.Release();
        LastUsed = DateTimeOffset.UtcNow;
    }

    public async ValueTask DisposeAsync()
    {
        if (_disposed)
            return;

        _disposed = true;
        Client.Dispose();
        await Server.DisposeAsync();
    }
}

/// <summary>
/// A lease for a pooled server.
/// Disposing the lease returns the server to the pool.
/// </summary>
public sealed class ServerLease : IAsyncDisposable
{
    private readonly PooledServer _server;
    private readonly LlamaServerPool _pool;
    private bool _disposed;

    internal ServerLease(PooledServer server, LlamaServerPool pool)
    {
        _server = server;
        _pool = pool;
    }

    /// <summary>
    /// Gets the server process.
    /// </summary>
    public LlamaServerProcess Server => _server.Server;

    /// <summary>
    /// Gets the HTTP client for the server.
    /// </summary>
    public LlamaServerClient Client => _server.Client;

    /// <summary>
    /// Gets the backend being used.
    /// </summary>
    public LlamaServerBackend Backend => _server.Backend;

    /// <summary>
    /// Returns the server to the pool.
    /// </summary>
    public ValueTask DisposeAsync()
    {
        if (_disposed)
            return ValueTask.CompletedTask;

        _disposed = true;
        LlamaServerPool.Release(_server);
        return ValueTask.CompletedTask;
    }
}

/// <summary>
/// Status of the server pool.
/// </summary>
public sealed record PoolStatus
{
    public int TotalServers { get; init; }
    public int ActiveServers { get; init; }
    public int IdleServers { get; init; }
    public IReadOnlyList<PoolEntry> Entries { get; init; } = [];
}

/// <summary>
/// Information about a pooled server.
/// </summary>
public sealed record PoolEntry
{
    public required string Key { get; init; }
    public required string ModelPath { get; init; }
    public required LlamaServerBackend Backend { get; init; }
    public bool IsInUse { get; init; }
    public DateTimeOffset LastUsed { get; init; }
    public int ProcessId { get; init; }
}
