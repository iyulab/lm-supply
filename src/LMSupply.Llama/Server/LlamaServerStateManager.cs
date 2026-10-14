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
    /// The directory a build lives in: <c>&lt;cache root&gt;/&lt;version&gt;/&lt;build backend&gt;</c>, resolved against
    /// this manager's cache root. The state records only the version and the build backend, so a moved cache root
    /// (<c>LMSUPPLY_CACHE_DIR</c>, a moved profile, a renamed drive) keeps its builds.
    /// </summary>
    /// <param name="version">Build version (e.g. "b7898").</param>
    /// <param name="buildBackend">Backend directory the build was extracted under (e.g. "cuda12", "cpu").</param>
    public string GetBuildDirectory(string version, string buildBackend)
        => Path.Combine(_cacheDirectory, version, buildBackend.ToLowerInvariant());

    /// <summary>
    /// Deletes build directories (<c>b*/&lt;backend&gt;</c>) under the cache root that no state entry references as
    /// its installed, pending or previous build, and a version directory left empty by that. This is the single
    /// retention mechanism: <see cref="ActivateUpdateAsync"/> trims state only, which makes superseded builds
    /// unreferenced and therefore reclaimable here. A build is matched by version and backend directory, not by path,
    /// so a backend directory no entry references is reclaimed even when another backend of the same version is kept.
    /// Deletion is best-effort per directory (a locked directory is retried on the next cleanup).
    /// </summary>
    /// <returns>The number of build directories deleted.</returns>
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

            var comparer = OperatingSystem.IsWindows() ? StringComparer.OrdinalIgnoreCase : StringComparer.Ordinal;
            var referenced = new HashSet<string>(stateFile.Entries.Values.SelectMany(ReferencedBuilds), comparer);

            var deleted = 0;
            foreach (var versionDir in Directory.GetDirectories(_cacheDirectory))
            {
                var version = Path.GetFileName(versionDir);
                if (!IsVersionDirectoryName(version))
                    continue;

                foreach (var buildDir in Directory.GetDirectories(versionDir))
                {
                    if (referenced.Contains(BuildKey(version, Path.GetFileName(buildDir))))
                        continue;

                    if (TryDelete(buildDir))
                        deleted++;
                }

                if (Directory.Exists(versionDir) && Directory.GetFileSystemEntries(versionDir).Length == 0)
                    TryDelete(versionDir);
            }

            return deleted;
        }
        finally
        {
            _lock.Release();
        }
    }

    private static IEnumerable<string> ReferencedBuilds(LlamaServerVersionState state)
    {
        yield return BuildKey(state.InstalledVersion, state.InstalledBuildBackend ?? state.Backend);
        if (state.PendingVersion is not null)
            yield return BuildKey(state.PendingVersion, state.PendingBuildBackend ?? state.Backend);
        foreach (var previous in state.PreviousVersions)
            yield return BuildKey(previous.Version, previous.BuildBackend ?? state.Backend);
    }

    private static string BuildKey(string version, string buildBackend) => $"{version}/{buildBackend.ToLowerInvariant()}";

    private static bool TryDelete(string directory)
    {
        try
        {
            Directory.Delete(directory, recursive: true);
            return true;
        }
        catch (Exception ex)
        {
            Trace.TraceInformation(
                $"[LlamaServerStateManager] Cleanup of superseded build dir failed (will retry next cleanup): {directory} - {ex.Message}");
            return false;
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
    /// <param name="backend">GPU backend of the state entry (e.g. "cuda12").</param>
    /// <param name="platform">Platform identifier (e.g. "win-x64").</param>
    /// <param name="version">Downloaded build version.</param>
    /// <param name="buildBackend">Backend directory the build was extracted under (see <see cref="GetBuildDirectory"/>).</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    public async Task MarkUpdateReadyAsync(
        string backend,
        string platform,
        string version,
        string buildBackend,
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
                state.PendingBuildBackend = buildBackend.ToLowerInvariant();
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

            if (!state.UpdateReady || state.PendingVersion == null)
                return null;

            // Move current to previous
            state.PreviousVersions.Insert(0, new VersionEntry
            {
                Version = state.InstalledVersion,
                BuildBackend = state.InstalledBuildBackend,
                InstalledAt = DateTimeOffset.UtcNow
            });

            // Trim old versions (their builds become unreferenced; CleanupUnreferencedVersionsAsync reclaims them)
            while (state.PreviousVersions.Count > maxVersionsToKeep)
                state.PreviousVersions.RemoveAt(state.PreviousVersions.Count - 1);

            // Activate pending
            state.InstalledVersion = state.PendingVersion;
            state.InstalledBuildBackend = state.PendingBuildBackend;
            state.PendingVersion = null;
            state.PendingBuildBackend = null;
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
    /// Rolls back to the newest previous version whose build is still on disk under the cache root, marking the
    /// installed version as failed. Previous entries newer than it whose build is gone are dropped. Returns null,
    /// changing nothing, when no previous build exists.
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

            var index = state.PreviousVersions.FindIndex(previous =>
                Directory.Exists(GetBuildDirectory(previous.Version, previous.BuildBackend ?? state.Backend)));
            if (index < 0)
                return null;

            // Mark current as failed
            state.FailedVersions.Add(state.InstalledVersion);

            // Restore previous; the entries before it have no build left to roll back to
            var previous = state.PreviousVersions[index];
            state.PreviousVersions.RemoveRange(0, index + 1);

            state.InstalledVersion = previous.Version;
            state.InstalledBuildBackend = previous.BuildBackend;

            // Clear pending update if any
            state.PendingVersion = null;
            state.PendingBuildBackend = null;
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
    /// Records <paramref name="version"/> as the installed build of a backend: a first installation, or a build
    /// adopted from the cache when the recorded one is gone. An existing entry keeps its failed versions, its last
    /// version check, the previous versions whose builds are still on disk, and a pending update whose build is on disk.
    /// </summary>
    /// <param name="backend">GPU backend of the state entry (e.g. "cuda12").</param>
    /// <param name="platform">Platform identifier (e.g. "win-x64").</param>
    /// <param name="version">Installed build version.</param>
    /// <param name="buildBackend">Backend directory the build was extracted under (see <see cref="GetBuildDirectory"/>).</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    public async Task<LlamaServerVersionState> CreateInitialStateAsync(
        string backend,
        string platform,
        string version,
        string buildBackend,
        CancellationToken cancellationToken = default)
    {
        await _lock.WaitAsync(cancellationToken);
        try
        {
            var stateFile = await LoadStateFileAsync(cancellationToken);
            var key = MakeKey(backend, platform);
            var existing = stateFile.Entries.GetValueOrDefault(key);

            var state = new LlamaServerVersionState
            {
                InstalledVersion = version,
                InstalledBuildBackend = buildBackend.ToLowerInvariant(),
                Backend = backend,
                LastVersionCheck = DateTimeOffset.UtcNow,
                LatestKnownVersion = version
            };

            if (existing is not null)
            {
                state.FailedVersions = existing.FailedVersions;
                state.LatestKnownVersion = existing.LatestKnownVersion ?? version;
                state.LastVersionCheck = existing.LastVersionCheck;
                state.PreviousVersions = existing.PreviousVersions
                    .Where(previous => !string.Equals(previous.Version, version, StringComparison.OrdinalIgnoreCase)
                        && Directory.Exists(GetBuildDirectory(previous.Version, previous.BuildBackend ?? existing.Backend)))
                    .ToList();
                if (existing.PendingVersion is not null
                    && !string.Equals(existing.PendingVersion, version, StringComparison.OrdinalIgnoreCase)
                    && Directory.Exists(GetBuildDirectory(existing.PendingVersion, existing.PendingBuildBackend ?? existing.Backend)))
                {
                    state.PendingVersion = existing.PendingVersion;
                    state.PendingBuildBackend = existing.PendingBuildBackend;
                    state.UpdateReady = existing.UpdateReady;
                }
            }

            stateFile.Entries[key] = state;
            await SaveStateFileAsync(stateFile, cancellationToken);
            return state;
        }
        finally
        {
            _lock.Release();
        }
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
            foreach (var state in _cachedState.Entries.Values)
                MigrateLegacyPaths(state);
        }
        catch (JsonException)
        {
            // Corrupted file, start fresh
            _cachedState = new LlamaServerStateFile();
        }

        return _cachedState;
    }

    // A file written by a release before 0.122 records builds only as absolute paths. The build backend is the path's
    // last segment; the version is recorded on its own. The paths themselves are not used — they point at whatever
    // root was in use when they were written.
    private static void MigrateLegacyPaths(LlamaServerVersionState state)
    {
        state.InstalledBuildBackend ??= LegacyBuildBackend(state.LegacyInstalledPath, state.InstalledVersion);
        if (state.PendingVersion is not null)
            state.PendingBuildBackend ??= LegacyBuildBackend(state.LegacyPendingPath, state.PendingVersion);
        foreach (var previous in state.PreviousVersions)
            previous.BuildBackend ??= LegacyBuildBackend(previous.LegacyPath, previous.Version);
    }

    private static string? LegacyBuildBackend(string? legacyPath, string version)
    {
        if (string.IsNullOrEmpty(legacyPath))
            return null;

        // Split on both separators: the file may have been written on another OS than the one reading it.
        var segments = legacyPath.Split(['/', '\\'], StringSplitOptions.RemoveEmptyEntries);
        return segments.Length >= 2 && string.Equals(segments[^2], version, StringComparison.OrdinalIgnoreCase)
            ? segments[^1].ToLowerInvariant()
            : null;
    }

    private async Task SaveStateFileAsync(LlamaServerStateFile stateFile, CancellationToken cancellationToken)
    {
        _cachedState = stateFile;
        stateFile.SchemaVersion = 2;

        // Absolute directories for releases before 0.122 sharing this cache (see LegacyInstalledPath), resolved
        // against this manager's root; never read back except by MigrateLegacyPaths.
        foreach (var state in stateFile.Entries.Values)
        {
            state.LegacyInstalledPath = GetBuildDirectory(state.InstalledVersion, state.InstalledBuildBackend ?? state.Backend);
            state.LegacyPendingPath = state.PendingVersion is null
                ? null
                : GetBuildDirectory(state.PendingVersion, state.PendingBuildBackend ?? state.Backend);
            foreach (var previous in state.PreviousVersions)
                previous.LegacyPath = GetBuildDirectory(previous.Version, previous.BuildBackend ?? state.Backend);
        }

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
