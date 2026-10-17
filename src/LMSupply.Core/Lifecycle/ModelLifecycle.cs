using System.Collections.Concurrent;
using System.Runtime.CompilerServices;
using System.Threading.Channels;
using LMSupply.Download;

namespace LMSupply.Lifecycle;

/// <summary>Where a model stands for a host: missing, downloading, ready to load offline, or its last download failed.</summary>
public enum ModelState
{
    /// <summary>Not everything a load needs is cached, and no download is running.</summary>
    Missing,

    /// <summary>A download started through <see cref="ModelLifecycle.DownloadAsync"/> is running.</summary>
    Downloading,

    /// <summary>Everything a load with downloads disabled needs is cached.</summary>
    Ready,

    /// <summary>The last download failed (see <see cref="ModelStatus.Error"/>); a new download clears it.</summary>
    Failed,
}

/// <summary>One model's status.</summary>
/// <param name="Kind">The kind of model (<see cref="IModelArtifacts.Kind"/>).</param>
/// <param name="ModelId">The model id as the host asked for it.</param>
/// <param name="State">Where the model stands.</param>
/// <param name="Progress">The latest progress of a running download; null otherwise.</param>
/// <param name="Error">Why the last download failed, when <paramref name="State"/> is <see cref="ModelState.Failed"/> — the exception
/// the download threw (for example a <c>ModelNotFoundException</c>, a <c>ModelDownloadException</c> or a llama-server acquisition
/// failure), so a host can tell «no build for this platform» from «network down».</param>
public sealed record ModelStatus(
    string Kind,
    string ModelId,
    ModelState State,
    DownloadProgress? Progress = null,
    Exception? Error = null);

/// <summary>
/// The models a host manages, across calls: each one's status, a stream of status changes, and downloads that any caller can
/// watch and cancel. Built over the <see cref="IModelArtifacts"/> of the model packages the host uses.
/// </summary>
/// <remarks>
/// <para>
/// It never downloads on its own: a status read only looks at the cache, and the network is used by <see cref="DownloadAsync"/>
/// alone — the host calls it when the user has agreed. Two calls for the same model share one download.
/// </para>
/// <para>
/// <see cref="DeleteAsync"/> removes what a download fetched; it is refused while that model downloads, and a download is refused
/// while it is being deleted.
/// </para>
/// <para>Thread-safe. Statuses are published to every <see cref="WatchAsync"/> stream open at the time.</para>
/// </remarks>
public sealed class ModelLifecycle
{
    private readonly Dictionary<string, IModelArtifacts> _artifacts;
    private readonly ConcurrentDictionary<(string Kind, string ModelId), Download> _downloads = new();
    private readonly ConcurrentDictionary<(string Kind, string ModelId), Exception> _failures = new();
    private readonly ConcurrentDictionary<Channel<ModelStatus>, byte> _watchers = new();

    // Guards the two transitions that must not interleave: a download starting and a delete starting for the same model.
    private readonly Lock _gate = new();
    private readonly HashSet<(string Kind, string ModelId)> _deleting = [];

    /// <param name="artifacts">One provider per kind of model, for example a <c>GeneratorArtifacts</c> and an <c>EmbedderArtifacts</c>.</param>
    /// <exception cref="ArgumentException">Two providers have the same <see cref="IModelArtifacts.Kind"/>.</exception>
    public ModelLifecycle(IEnumerable<IModelArtifacts> artifacts)
    {
        ArgumentNullException.ThrowIfNull(artifacts);
        _artifacts = new Dictionary<string, IModelArtifacts>(StringComparer.OrdinalIgnoreCase);
        foreach (var provider in artifacts)
        {
            if (!_artifacts.TryAdd(provider.Kind, provider))
                throw new ArgumentException($"Two model providers have the kind '{provider.Kind}'.", nameof(artifacts));
        }
    }

    /// <summary>The kinds of model this lifecycle manages.</summary>
    public IReadOnlyCollection<string> Kinds => _artifacts.Keys;

    /// <summary>The status of <paramref name="modelId"/>. Reads the cache only; never downloads.</summary>
    /// <exception cref="ArgumentException">No provider is registered for <paramref name="kind"/>.</exception>
    public async Task<ModelStatus> GetStatusAsync(string kind, string modelId, CancellationToken cancellationToken = default)
    {
        var provider = Provider(kind);
        ArgumentException.ThrowIfNullOrWhiteSpace(modelId);
        var key = Key(provider, modelId);

        if (_downloads.TryGetValue(key, out var running))
            return running.Status;

        if (await provider.IsAvailableOfflineAsync(modelId, cancellationToken))
        {
            _failures.TryRemove(key, out _);
            return new ModelStatus(provider.Kind, modelId, ModelState.Ready);
        }

        return _failures.TryGetValue(key, out var error)
            ? new ModelStatus(provider.Kind, modelId, ModelState.Failed, Error: error)
            : new ModelStatus(provider.Kind, modelId, ModelState.Missing);
    }

    /// <summary>
    /// Every status change from now on — a download starting, its progress, its end (ready, missing after a cancel, or failed) —
    /// until <paramref name="cancellationToken"/> is cancelled.
    /// </summary>
    public async IAsyncEnumerable<ModelStatus> WatchAsync([EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        var channel = Channel.CreateUnbounded<ModelStatus>(new UnboundedChannelOptions { SingleReader = true });
        _watchers.TryAdd(channel, 0);
        try
        {
            while (await channel.Reader.WaitToReadAsync(cancellationToken))
            {
                while (channel.Reader.TryRead(out var status))
                    yield return status;
            }
        }
        finally
        {
            _watchers.TryRemove(channel, out _);
        }
    }

    /// <summary>
    /// Downloads what a load of <paramref name="modelId"/> needs (the host's consent is calling this), or joins the download of
    /// it that is already running, and returns the status it ends in (<see cref="ModelState.Ready"/>).
    /// </summary>
    /// <param name="kind">The kind of model.</param>
    /// <param name="modelId">The model.</param>
    /// <param name="cancellationToken">Stops this caller waiting; the download goes on for others (stop it with <see cref="CancelDownload"/>).</param>
    /// <exception cref="OperationCanceledException">This caller's token was cancelled, or the download was cancelled with <see cref="CancelDownload"/>.</exception>
    /// <exception cref="ArgumentException">No provider is registered for <paramref name="kind"/>.</exception>
    /// <exception cref="InvalidOperationException">The model is being deleted (<see cref="DeleteAsync"/>).</exception>
    /// <remarks>A failed download is rethrown to every caller waiting on it, and the model's status becomes <see cref="ModelState.Failed"/>.</remarks>
    public async Task<ModelStatus> DownloadAsync(string kind, string modelId, CancellationToken cancellationToken = default)
    {
        var provider = Provider(kind);
        ArgumentException.ThrowIfNullOrWhiteSpace(modelId);
        var key = Key(provider, modelId);

        Download download;
        lock (_gate)
        {
            if (_deleting.Contains(key))
                throw new InvalidOperationException($"The {provider.Kind} '{modelId}' is being deleted; download it after the delete returns.");
            download = _downloads.GetOrAdd(key, _ => new Download(this, provider, modelId));
        }

        download.StartOnce();
        await download.Completion.WaitAsync(cancellationToken);
        return new ModelStatus(provider.Kind, modelId, ModelState.Ready);
    }

    /// <summary>Cancels the running download of <paramref name="modelId"/>. Its partial files stay, so a later download resumes them.</summary>
    /// <returns>Whether a download was running.</returns>
    public bool CancelDownload(string kind, string modelId)
    {
        var provider = Provider(kind);
        ArgumentException.ThrowIfNullOrWhiteSpace(modelId);
        if (!_downloads.TryGetValue(Key(provider, modelId), out var download))
            return false;

        download.Cancel();
        return true;
    }

    /// <summary>
    /// Deletes what LMSupply downloaded of <paramref name="modelId"/> (<see cref="IModelArtifacts.Delete"/>) and publishes the
    /// status it is left in — <see cref="ModelState.Missing"/>, or still <see cref="ModelState.Ready"/> when the files a load
    /// needs came from somewhere else (another tool's snapshot, a local path).
    /// </summary>
    /// <param name="kind">The kind of model.</param>
    /// <param name="modelId">The model, as the host downloads and loads it.</param>
    /// <param name="cancellationToken">Stops the status read after the delete; the delete itself is not interrupted.</param>
    /// <returns>Whether anything was deleted.</returns>
    /// <exception cref="ArgumentException">No provider is registered for <paramref name="kind"/>.</exception>
    /// <exception cref="InvalidOperationException">The model is downloading — cancel it (<see cref="CancelDownload"/>) first — or
    /// another caller is deleting it.</exception>
    /// <exception cref="IOException">A file is open, for example by a running server (Windows).</exception>
    public async Task<bool> DeleteAsync(string kind, string modelId, CancellationToken cancellationToken = default)
    {
        var provider = Provider(kind);
        ArgumentException.ThrowIfNullOrWhiteSpace(modelId);
        var key = Key(provider, modelId);

        lock (_gate)
        {
            if (_downloads.ContainsKey(key))
                throw new InvalidOperationException($"The {provider.Kind} '{modelId}' is downloading; cancel the download (CancelDownload) before deleting it.");
            if (!_deleting.Add(key))
                throw new InvalidOperationException($"The {provider.Kind} '{modelId}' is already being deleted.");
        }

        bool deleted;
        try
        {
            deleted = provider.Delete(modelId);
            _failures.TryRemove(key, out _);
        }
        finally
        {
            lock (_gate)
                _deleting.Remove(key);
        }

        if (deleted)
            Publish(await GetStatusAsync(provider.Kind, modelId, cancellationToken));
        return deleted;
    }

    private IModelArtifacts Provider(string kind)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(kind);
        return _artifacts.TryGetValue(kind, out var provider)
            ? provider
            : throw new ArgumentException($"No model provider is registered for the kind '{kind}'. Registered: {string.Join(", ", _artifacts.Keys)}.", nameof(kind));
    }

    private static (string Kind, string ModelId) Key(IModelArtifacts provider, string modelId) => (provider.Kind, modelId);

    private void Publish(ModelStatus status)
    {
        foreach (var watcher in _watchers.Keys)
            watcher.Writer.TryWrite(status);
    }

    /// <summary>One running download, shared by every caller of <see cref="DownloadAsync"/> for the model.</summary>
    private sealed class Download : IDisposable
    {
        private readonly ModelLifecycle _owner;
        private readonly IModelArtifacts _provider;
        private readonly string _modelId;
        private readonly CancellationTokenSource _cts = new();
        private readonly TaskCompletionSource _completion = new(TaskCreationOptions.RunContinuationsAsynchronously);
        private int _started;
        private volatile ModelStatus _status;

        public Download(ModelLifecycle owner, IModelArtifacts provider, string modelId)
        {
            _owner = owner;
            _provider = provider;
            _modelId = modelId;
            _status = new ModelStatus(provider.Kind, modelId, ModelState.Downloading);
        }

        public ModelStatus Status => _status;

        public Task Completion => _completion.Task;

        public void Cancel()
        {
            try
            {
                _cts.Cancel();
            }
            catch (ObjectDisposedException)
            {
                // The download already ended.
            }
        }

        public void Dispose() => _cts.Dispose();

        public void StartOnce()
        {
            if (Interlocked.Exchange(ref _started, 1) == 0)
                _ = RunAsync();
        }

        private async Task RunAsync()
        {
            var key = (_provider.Kind, _modelId);
            _owner._failures.TryRemove(key, out _);
            _owner.Publish(_status);
            try
            {
                var progress = new CoalescingProgress(new StatusProgress(this));
                await _provider.DownloadAsync(_modelId, progress, _cts.Token);
                _owner._downloads.TryRemove(key, out _);
                _owner.Publish(new ModelStatus(_provider.Kind, _modelId, ModelState.Ready));
                _completion.TrySetResult();
            }
            catch (OperationCanceledException) when (_cts.IsCancellationRequested)
            {
                _owner._downloads.TryRemove(key, out _);
                _owner.Publish(new ModelStatus(_provider.Kind, _modelId, ModelState.Missing));
                _completion.TrySetCanceled(_cts.Token);
            }
            catch (Exception ex)
            {
                _owner._failures[key] = ex;
                _owner._downloads.TryRemove(key, out _);
                _owner.Publish(new ModelStatus(_provider.Kind, _modelId, ModelState.Failed, Error: ex));
                _completion.TrySetException(ex);
            }
            finally
            {
                Dispose();
            }
        }

        private sealed class StatusProgress(Download download) : IProgress<DownloadProgress>
        {
            public void Report(DownloadProgress value)
            {
                var status = download._status with { Progress = value };
                download._status = status;
                download._owner.Publish(status);
            }
        }
    }
}
