using System.Diagnostics;

namespace LMSupply.Llama.Server;

/// <summary>
/// Holds a model's server lease and replaces it when the server has exited — killed, crashed, or out of memory because
/// another process took it — so the call that saw it die fails but the next one works. A model that kept its first
/// lease failed every later call on the dead port with "connection refused" and never said the server was gone.
/// </summary>
/// <typeparam name="TLease">The lease; generic so the replacement rule can be tested without a real server.</typeparam>
internal sealed class RestartingLease<TLease> : IAsyncDisposable
    where TLease : class, IAsyncDisposable
{
    private readonly Func<CancellationToken, Task<TLease>> _relet;
    private readonly Func<TLease, bool> _isAlive;
    private readonly Func<TLease, string> _describeDeath;
    private readonly string _label;
    private readonly SemaphoreSlim _gate = new(1, 1);
    private TLease _current;
    private bool _disposed;

    /// <param name="initial">The lease the model was loaded with.</param>
    /// <param name="relet">Leases a new server with the configuration the model was loaded with.</param>
    /// <param name="isAlive">Whether a lease's server is still running.</param>
    /// <param name="describeDeath">What to say about a dead lease's server (exit code, last output).</param>
    /// <param name="label">Who holds the lease, for the warning (e.g. <c>LlamaServerEmbeddingModel 'bge-m3'</c>).</param>
    public RestartingLease(
        TLease initial,
        Func<CancellationToken, Task<TLease>> relet,
        Func<TLease, bool> isAlive,
        Func<TLease, string> describeDeath,
        string label)
    {
        _current = initial;
        _relet = relet;
        _isAlive = isAlive;
        _describeDeath = describeDeath;
        _label = label;
    }

    /// <summary>The lease as it is now — for state that does not need a live server (backend, startup log).</summary>
    public TLease Current => Volatile.Read(ref _current);

    /// <summary>The current lease if its server is running, otherwise a new one (the dead one is released).</summary>
    public async ValueTask<TLease> GetAsync(CancellationToken cancellationToken)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var lease = Current;
        if (_isAlive(lease))
            return lease;

        await _gate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            lease = Current;
            if (_isAlive(lease))
                return lease;

            Trace.TraceWarning($"[{_label}] llama-server is no longer running ({_describeDeath(lease)}); starting a new one.");

            var replacement = await _relet(cancellationToken).ConfigureAwait(false);
            Volatile.Write(ref _current, replacement);
            await lease.DisposeAsync().ConfigureAwait(false);
            return replacement;
        }
        finally
        {
            _gate.Release();
        }
    }

    public async ValueTask DisposeAsync()
    {
        if (_disposed)
            return;
        _disposed = true;
        await Current.DisposeAsync().ConfigureAwait(false);
        _gate.Dispose();
    }
}

/// <summary>The <see cref="RestartingLease{TLease}"/> every GGUF model uses over <see cref="LlamaServerPool"/>.</summary>
internal static class RestartingServerLease
{
    public static RestartingLease<ServerLease> Create(
        ServerLease initial,
        Func<CancellationToken, Task<ServerLease>> relet,
        string label) =>
        new(initial, relet, static lease => lease.Server.IsRunning, DescribeDeath, label);

    internal static string DescribeDeath(ServerLease lease)
    {
        var exitCode = lease.Server.ExitCode;
        var lastLines = lease.Server.RecentLog
            .Split('\n', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .TakeLast(20);
        return $"exit code {exitCode?.ToString(System.Globalization.CultureInfo.InvariantCulture) ?? "unknown"}; its last output:" +
               $"{Environment.NewLine}{string.Join(Environment.NewLine, lastLines)}";
    }
}
