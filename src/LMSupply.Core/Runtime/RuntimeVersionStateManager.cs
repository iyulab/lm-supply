using System.Text.Json;
using System.Text.Json.Serialization;
using LMSupply.Json;

namespace LMSupply.Runtime;

/// <summary>
/// Manages persistent runtime version state stored in runtime-versions.json.
/// Thread-safe with atomic file writes for reliability.
/// </summary>
public sealed class RuntimeVersionStateManager : IDisposable
{
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        TypeInfoResolver = CoreJsonContext.Default,
        WriteIndented = true,
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull
    };

    private readonly string _stateFilePath;
    private readonly SemaphoreSlim _lock = new(1, 1);
    private RuntimeVersionStateFile? _cachedState;
    private bool _disposed;

    /// <summary>
    /// Creates a new state manager using the specified cache directory.
    /// </summary>
    /// <param name="cacheDirectory">The cache directory. If null, uses default LMSupply cache.</param>
    public RuntimeVersionStateManager(string? cacheDirectory = null)
    {
        var dir = cacheDirectory ?? LMSupplyCachePaths.GetRuntimesDirectory();
        _stateFilePath = Path.Combine(dir, "runtime-versions.json");
    }

    /// <summary>
    /// Gets the state file path.
    /// </summary>
    public string StateFilePath => _stateFilePath;

    /// <summary>
    /// Generates a package key from components.
    /// Format: "{packageType}|{provider}|{rid}"
    /// </summary>
    public static string GetPackageKey(string packageType, string provider, string rid)
    {
        return $"{packageType.ToLowerInvariant()}|{provider.ToLowerInvariant()}|{rid.ToLowerInvariant()}";
    }

    /// <summary>
    /// Gets the state for a specific package, or null if not found.
    /// </summary>
    public async Task<RuntimeVersionState?> GetStateAsync(string packageKey, CancellationToken ct = default)
    {
        var state = await LoadStateFileAsync(ct);
        return state.Packages.GetValueOrDefault(packageKey);
    }

    /// <summary>
    /// Gets or creates state for a specific package.
    /// </summary>
    public async Task<RuntimeVersionState> GetOrCreateStateAsync(
        string packageKey,
        string initialVersion,
        CancellationToken ct = default)
    {
        await _lock.WaitAsync(ct);
        try
        {
            var state = await LoadStateFileAsync(ct);
            if (!state.Packages.TryGetValue(packageKey, out var packageState))
            {
                packageState = new RuntimeVersionState
                {
                    InstalledVersion = initialVersion,
                    LastVersionCheck = DateTimeOffset.MinValue
                };
                state.Packages[packageKey] = packageState;
                await SaveStateFileAsync(state, ct);
            }
            return packageState;
        }
        finally
        {
            _lock.Release();
        }
    }

    /// <summary>
    /// Updates the state for a specific package.
    /// </summary>
    public async Task UpdateStateAsync(
        string packageKey,
        RuntimeVersionState packageState,
        CancellationToken ct = default)
    {
        await _lock.WaitAsync(ct);
        try
        {
            var state = await LoadStateFileAsync(ct);
            state.Packages[packageKey] = packageState;
            await SaveStateFileAsync(state, ct);
        }
        finally
        {
            _lock.Release();
        }
    }

    /// <summary>
    /// Records a version check result.
    /// </summary>
    public async Task RecordVersionCheckAsync(
        string packageKey,
        string? latestVersion,
        CancellationToken ct = default)
    {
        await _lock.WaitAsync(ct);
        try
        {
            var state = await LoadStateFileAsync(ct);
            if (state.Packages.TryGetValue(packageKey, out var packageState))
            {
                packageState.LatestKnownVersion = latestVersion;
                packageState.LastVersionCheck = DateTimeOffset.UtcNow;
                await SaveStateFileAsync(state, ct);
            }
        }
        finally
        {
            _lock.Release();
        }
    }

    /// <summary>
    /// Marks an update as ready to be applied.
    /// </summary>
    public async Task MarkUpdateReadyAsync(
        string packageKey,
        string version,
        string path,
        CancellationToken ct = default)
    {
        await _lock.WaitAsync(ct);
        try
        {
            var state = await LoadStateFileAsync(ct);
            if (state.Packages.TryGetValue(packageKey, out var packageState))
            {
                packageState.PendingVersion = null;
                packageState.UpdateReady = true;
                packageState.UpdateReadyPath = path;
                packageState.LatestKnownVersion = version;
                await SaveStateFileAsync(state, ct);
            }
        }
        finally
        {
            _lock.Release();
        }
    }

    /// <summary>
    /// Activates a ready update, moving the current version to previous.
    /// </summary>
    public async Task<RuntimeVersionState?> ActivateUpdateAsync(
        string packageKey,
        int maxVersionsToKeep,
        CancellationToken ct = default)
    {
        await _lock.WaitAsync(ct);
        try
        {
            var state = await LoadStateFileAsync(ct);
            if (!state.Packages.TryGetValue(packageKey, out var packageState))
                return null;

            if (!packageState.UpdateReady || string.IsNullOrEmpty(packageState.LatestKnownVersion))
                return packageState;

            // Move current to previous (for rollback)
            packageState.PreviousVersions.Insert(0, packageState.InstalledVersion);

            // Trim old versions
            while (packageState.PreviousVersions.Count > maxVersionsToKeep)
            {
                packageState.PreviousVersions.RemoveAt(packageState.PreviousVersions.Count - 1);
            }

            // Activate update
            packageState.InstalledVersion = packageState.LatestKnownVersion;
            packageState.UpdateReady = false;
            packageState.UpdateReadyPath = null;

            await SaveStateFileAsync(state, ct);
            return packageState;
        }
        finally
        {
            _lock.Release();
        }
    }

    /// <summary>
    /// Performs a rollback to the previous version after a failed update.
    /// </summary>
    public async Task<(string? previousVersion, string? path)> RollbackAsync(
        string packageKey,
        string failedVersion,
        CancellationToken ct = default)
    {
        await _lock.WaitAsync(ct);
        try
        {
            var state = await LoadStateFileAsync(ct);
            if (!state.Packages.TryGetValue(packageKey, out var packageState))
                return (null, null);

            // Mark failed version
            packageState.FailedVersions.Add(failedVersion);

            // Get previous version
            if (packageState.PreviousVersions.Count == 0)
                return (null, null);

            var previousVersion = packageState.PreviousVersions[0];
            packageState.PreviousVersions.RemoveAt(0);
            packageState.InstalledVersion = previousVersion;
            packageState.UpdateReady = false;
            packageState.UpdateReadyPath = null;

            await SaveStateFileAsync(state, ct);
            return (previousVersion, null); // Path will be resolved by caller
        }
        finally
        {
            _lock.Release();
        }
    }

    /// <summary>
    /// Checks if a version check is due based on the interval.
    /// </summary>
    public async Task<bool> IsVersionCheckDueAsync(
        string packageKey,
        TimeSpan interval,
        CancellationToken ct = default)
    {
        var packageState = await GetStateAsync(packageKey, ct);
        if (packageState is null)
            return true;

        return DateTimeOffset.UtcNow - packageState.LastVersionCheck >= interval;
    }

    /// <summary>
    /// Marks a version download as pending.
    /// </summary>
    public async Task MarkPendingDownloadAsync(
        string packageKey,
        string version,
        CancellationToken ct = default)
    {
        await _lock.WaitAsync(ct);
        try
        {
            var state = await LoadStateFileAsync(ct);
            if (state.Packages.TryGetValue(packageKey, out var packageState))
            {
                packageState.PendingVersion = version;
                await SaveStateFileAsync(state, ct);
            }
        }
        finally
        {
            _lock.Release();
        }
    }

    /// <summary>
    /// Clears the pending download status.
    /// </summary>
    public async Task ClearPendingDownloadAsync(string packageKey, CancellationToken ct = default)
    {
        await _lock.WaitAsync(ct);
        try
        {
            var state = await LoadStateFileAsync(ct);
            if (state.Packages.TryGetValue(packageKey, out var packageState))
            {
                packageState.PendingVersion = null;
                await SaveStateFileAsync(state, ct);
            }
        }
        finally
        {
            _lock.Release();
        }
    }

    private async Task<RuntimeVersionStateFile> LoadStateFileAsync(CancellationToken ct)
    {
        if (_cachedState is not null)
            return _cachedState;

        if (!File.Exists(_stateFilePath))
        {
            _cachedState = new RuntimeVersionStateFile();
            return _cachedState;
        }

        try
        {
            var json = await File.ReadAllTextAsync(_stateFilePath, ct);
            _cachedState = JsonSerializer.Deserialize<RuntimeVersionStateFile>(json, JsonOptions)
                ?? new RuntimeVersionStateFile();
        }
        catch (JsonException)
        {
            // Corrupted file, start fresh
            _cachedState = new RuntimeVersionStateFile();
        }

        return _cachedState;
    }

    private async Task SaveStateFileAsync(RuntimeVersionStateFile state, CancellationToken ct)
    {
        _cachedState = state;

        // Ensure directory exists
        var dir = Path.GetDirectoryName(_stateFilePath);
        if (!string.IsNullOrEmpty(dir))
            Directory.CreateDirectory(dir);

        // Atomic write using temp file
        var tempPath = _stateFilePath + ".tmp";
        var json = JsonSerializer.Serialize(state, JsonOptions);

        await File.WriteAllTextAsync(tempPath, json, ct);

        // Atomic rename (works on Windows and Unix)
        File.Move(tempPath, _stateFilePath, overwrite: true);
    }

    public void Dispose()
    {
        if (_disposed)
            return;

        _disposed = true;
        _lock.Dispose();
    }
}
