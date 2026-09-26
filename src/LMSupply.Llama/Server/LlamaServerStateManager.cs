using System.Diagnostics;
using System.Text.Json;
using LMSupply.Json;

namespace LMSupply.Llama.Server;

/// <summary>
/// Manages persistent state for llama-server installations.
/// State is stored in a JSON file in the cache directory.
/// </summary>
public sealed class LlamaServerStateManager : IDisposable
{
    private const string StateFileName = "llama-server-state.json";

    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        WriteIndented = true,
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        TypeInfoResolver = LlamaJsonContext.Default
    };

    private readonly string _cacheDirectory;
    private readonly string _stateFilePath;
    private readonly SemaphoreSlim _lock = new(1, 1);
    private LlamaServerStateFile? _cachedState;
    private bool _disposed;

    /// <summary>
    /// Creates a new state manager.
    /// </summary>
    /// <param name="cacheDirectory">Directory to store state file.</param>
    public LlamaServerStateManager(string? cacheDirectory = null)
    {
        var dir = cacheDirectory ?? LMSupplyCachePaths.GetLlamaServerDirectory();

        Directory.CreateDirectory(dir);
        _cacheDirectory = dir;
        _stateFilePath = Path.Combine(dir, StateFileName);
    }

    /// <summary>
    /// Deletes versioned build directories (<c>b*</c>) under the cache root that no state entry
    /// references via InstalledPath, PendingPath, or PreviousVersions. This is the single retention
    /// mechanism: <see cref="ActivateUpdateAsync"/> trims state only, which makes superseded
    /// versions unreferenced and therefore reclaimable here. Deletion is best-effort per directory
    /// (a locked directory is retried on the next cleanup).
    /// </summary>
    /// <returns>The number of version directories deleted.</returns>
    public async Task<int> CleanupUnreferencedVersionsAsync(CancellationToken cancellationToken = default)
    {
        await _lock.WaitAsync(cancellationToken);
        try
        {
            // Destructive decision: re-read from disk. The cache directory is shared across
            // processes, so the process-local cache may miss references another process just
            // persisted (e.g. a pending download).
            _cachedState = null;
            var stateFile = await LoadStateFileAsync(cancellationToken);

            // Zero entries means "state unknown", not "everything is orphaned": pre-state caches
            // are adopted through GetCachedVersions discovery and must never be swept here.
            if (stateFile.Entries.Count == 0)
                return 0;

            var referencedPaths = stateFile.Entries.Values
                .SelectMany(state => state.PreviousVersions.Select(previous => previous.Path)
                    .Append(state.InstalledPath)
                    .Append(state.PendingPath))
                .Where(path => !string.IsNullOrEmpty(path))
                .Select(path => Path.GetFullPath(path!).TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar))
                .ToList();

            var comparison = OperatingSystem.IsWindows()
                ? StringComparison.OrdinalIgnoreCase
                : StringComparison.Ordinal;

            var deleted = 0;
            foreach (var candidate in Directory.GetDirectories(_cacheDirectory))
            {
                if (!IsVersionDirectoryName(Path.GetFileName(candidate)))
                    continue;

                var candidateFull = Path.GetFullPath(candidate)
                    .TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
                var candidatePrefix = candidateFull + Path.DirectorySeparatorChar;

                var isReferenced = referencedPaths.Any(path =>
                    path.Equals(candidateFull, comparison) ||
                    path.StartsWith(candidatePrefix, comparison));
                if (isReferenced)
                    continue;

                try
                {
                    Directory.Delete(candidate, recursive: true);
                    deleted++;
                }
                catch (Exception ex)
                {
                    Trace.TraceInformation(
                        $"[LlamaServerStateManager] Cleanup of superseded version dir failed (will retry next cleanup): {candidate} — {ex.Message}");
                }
            }

            return deleted;
        }
        finally
        {
            _lock.Release();
        }
    }

    private static bool IsVersionDirectoryName(string? name)
    {
        if (name is null || name.Length < 2 || name[0] != 'b')
            return false;

        for (var i = 1; i < name.Length; i++)
        {
            if (!char.IsAsciiDigit(name[i]))
                return false;
        }

        return true;
    }

    /// <summary>
    /// Gets the state for a specific backend and platform.
    /// </summary>
    /// <param name="backend">GPU backend (e.g., "vulkan").</param>
    /// <param name="platform">Platform identifier (e.g., "win-x64").</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>State for the backend/platform, or null if not found.</returns>
    public async Task<LlamaServerVersionState?> GetStateAsync(
        string backend,
        string platform,
        CancellationToken cancellationToken = default)
    {
        await _lock.WaitAsync(cancellationToken);
        try
        {
            var stateFile = await LoadStateFileAsync(cancellationToken);
            var key = MakeKey(backend, platform);
            return stateFile.Entries.GetValueOrDefault(key);
        }
        finally
        {
            _lock.Release();
        }
    }

    /// <summary>
    /// Updates the state for a specific backend and platform.
    /// </summary>
    public async Task UpdateStateAsync(
        string backend,
        string platform,
        LlamaServerVersionState state,
        CancellationToken cancellationToken = default)
    {
        await _lock.WaitAsync(cancellationToken);
        try
        {
            var stateFile = await LoadStateFileAsync(cancellationToken);
            var key = MakeKey(backend, platform);
            stateFile.Entries[key] = state;
            await SaveStateFileAsync(stateFile, cancellationToken);
        }
        finally
        {
            _lock.Release();
        }
    }

    /// <summary>
    /// Records the result of a version check.
    /// </summary>
    public async Task RecordVersionCheckAsync(
        string backend,
        string platform,
        string? latestVersion,
        CancellationToken cancellationToken = default)
    {
        await _lock.WaitAsync(cancellationToken);
        try
        {
            var stateFile = await LoadStateFileAsync(cancellationToken);
            var key = MakeKey(backend, platform);

            if (stateFile.Entries.TryGetValue(key, out var state))
            {
                state.LatestKnownVersion = latestVersion;
                state.LastVersionCheck = DateTimeOffset.UtcNow;
                await SaveStateFileAsync(stateFile, cancellationToken);
            }
        }
        finally
        {
            _lock.Release();
        }
    }

    /// <summary>
    /// Marks a new version as ready to be applied.
    /// </summary>
    public async Task MarkUpdateReadyAsync(
        string backend,
        string platform,
        string version,
        string path,
        CancellationToken cancellationToken = default)
    {
        await _lock.WaitAsync(cancellationToken);
        try
        {
            var stateFile = await LoadStateFileAsync(cancellationToken);
            var key = MakeKey(backend, platform);

            if (stateFile.Entries.TryGetValue(key, out var state))
            {
                state.PendingVersion = version;
                state.PendingPath = path;
                state.UpdateReady = true;
                await SaveStateFileAsync(stateFile, cancellationToken);
            }
        }
        finally
        {
            _lock.Release();
        }
    }

    /// <summary>
    /// Activates a pending update, moving the current version to previous versions.
    /// </summary>
    public async Task<LlamaServerVersionState?> ActivateUpdateAsync(
        string backend,
        string platform,
        int maxVersionsToKeep = 2,
        CancellationToken cancellationToken = default)
    {
        await _lock.WaitAsync(cancellationToken);
        try
        {
            var stateFile = await LoadStateFileAsync(cancellationToken);
            var key = MakeKey(backend, platform);

            if (!stateFile.Entries.TryGetValue(key, out var state))
                return null;

            if (!state.UpdateReady || state.PendingVersion == null || state.PendingPath == null)
                return null;

            // Move current to previous
            state.PreviousVersions.Insert(0, new VersionEntry
            {
                Version = state.InstalledVersion,
                Path = state.InstalledPath,
                InstalledAt = DateTimeOffset.UtcNow
            });

            // Trim old versions
            while (state.PreviousVersions.Count > maxVersionsToKeep)
            {
                var oldest = state.PreviousVersions[^1];
                state.PreviousVersions.RemoveAt(state.PreviousVersions.Count - 1);

                // Optionally clean up old files (leave for now, can be done manually)
            }

            // Activate pending
            state.InstalledVersion = state.PendingVersion;
            state.InstalledPath = state.PendingPath;
            state.PendingVersion = null;
            state.PendingPath = null;
            state.UpdateReady = false;

            await SaveStateFileAsync(stateFile, cancellationToken);
            return state;
        }
        finally
        {
            _lock.Release();
        }
    }

    /// <summary>
    /// Rolls back to a previous version.
    /// </summary>
    public async Task<LlamaServerVersionState?> RollbackAsync(
        string backend,
        string platform,
        CancellationToken cancellationToken = default)
    {
        await _lock.WaitAsync(cancellationToken);
        try
        {
            var stateFile = await LoadStateFileAsync(cancellationToken);
            var key = MakeKey(backend, platform);

            if (!stateFile.Entries.TryGetValue(key, out var state))
                return null;

            if (state.PreviousVersions.Count == 0)
                return null;

            // Mark current as failed
            state.FailedVersions.Add(state.InstalledVersion);

            // Restore previous
            var previous = state.PreviousVersions[0];
            state.PreviousVersions.RemoveAt(0);

            state.InstalledVersion = previous.Version;
            state.InstalledPath = previous.Path;

            // Clear pending update if any
            state.PendingVersion = null;
            state.PendingPath = null;
            state.UpdateReady = false;

            await SaveStateFileAsync(stateFile, cancellationToken);
            return state;
        }
        finally
        {
            _lock.Release();
        }
    }

    /// <summary>
    /// Records a version as failed (will be skipped in future checks).
    /// </summary>
    public async Task MarkVersionFailedAsync(
        string backend,
        string platform,
        string version,
        CancellationToken cancellationToken = default)
    {
        await _lock.WaitAsync(cancellationToken);
        try
        {
            var stateFile = await LoadStateFileAsync(cancellationToken);
            var key = MakeKey(backend, platform);

            if (stateFile.Entries.TryGetValue(key, out var state))
            {
                state.FailedVersions.Add(version);
                await SaveStateFileAsync(stateFile, cancellationToken);
            }
        }
        finally
        {
            _lock.Release();
        }
    }

    /// <summary>
    /// Creates initial state for a new installation.
    /// </summary>
    public async Task<LlamaServerVersionState> CreateInitialStateAsync(
        string backend,
        string platform,
        string version,
        string path,
        CancellationToken cancellationToken = default)
    {
        var state = new LlamaServerVersionState
        {
            InstalledVersion = version,
            InstalledPath = path,
            Backend = backend,
            LastVersionCheck = DateTimeOffset.UtcNow,
            LatestKnownVersion = version
        };

        await UpdateStateAsync(backend, platform, state, cancellationToken);
        return state;
    }

    private async Task<LlamaServerStateFile> LoadStateFileAsync(CancellationToken cancellationToken)
    {
        if (_cachedState != null)
            return _cachedState;

        if (!File.Exists(_stateFilePath))
        {
            _cachedState = new LlamaServerStateFile();
            return _cachedState;
        }

        try
        {
            var json = await File.ReadAllTextAsync(_stateFilePath, cancellationToken);
            _cachedState = JsonSerializer.Deserialize(json, JsonOptions.TypeInfo<LlamaServerStateFile>())
                ?? new LlamaServerStateFile();
        }
        catch (JsonException)
        {
            // Corrupted file, start fresh
            _cachedState = new LlamaServerStateFile();
        }

        return _cachedState;
    }

    private async Task SaveStateFileAsync(LlamaServerStateFile stateFile, CancellationToken cancellationToken)
    {
        _cachedState = stateFile;

        // Atomic write: write to temp file, then rename
        var tempPath = _stateFilePath + ".tmp";
        var json = JsonSerializer.Serialize(stateFile, JsonOptions.TypeInfoOf(stateFile));

        await File.WriteAllTextAsync(tempPath, json, cancellationToken);
        File.Move(tempPath, _stateFilePath, overwrite: true);
    }

    private static string MakeKey(string backend, string platform)
        => $"{backend.ToLowerInvariant()}|{platform.ToLowerInvariant()}";

    public void Dispose()
    {
        if (_disposed)
            return;

        _disposed = true;
        _lock.Dispose();
    }
}
