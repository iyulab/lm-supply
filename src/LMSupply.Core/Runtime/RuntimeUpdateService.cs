using System.Collections.Concurrent;
using System.Diagnostics;
using LMSupply.Download;

namespace LMSupply.Runtime;

/// <summary>
/// Core service for runtime auto-update functionality.
/// Manages version checking, background downloads, and rollback.
/// </summary>
public sealed class RuntimeUpdateService : IAsyncDisposable
{
    private static readonly ConcurrentDictionary<string, RuntimeUpdateService> _instances = new();

    private readonly string _packageType;
    private readonly RuntimeUpdateOptions _options;
    private readonly RuntimeVersionStateManager _stateManager;
    private readonly NuGetPackageResolver _packageResolver;
    private readonly SemaphoreSlim _downloadLock = new(1, 1);
    private readonly ConcurrentDictionary<string, Task> _backgroundTasks = new();

    private bool _disposed;

    /// <summary>
    /// Gets or creates a singleton instance for the specified package type.
    /// </summary>
    /// <param name="packageType">The package type (e.g., "llama-server", "onnxruntime").</param>
    /// <param name="options">Optional update options. Only used on first creation.</param>
    public static RuntimeUpdateService GetInstance(
        string packageType,
        RuntimeUpdateOptions? options = null)
    {
        return _instances.GetOrAdd(packageType, pt =>
            new RuntimeUpdateService(pt, options ?? RuntimeUpdateOptions.Default));
    }

    private RuntimeUpdateService(string packageType, RuntimeUpdateOptions options)
    {
        _packageType = packageType;
        _options = options;
        _stateManager = new RuntimeVersionStateManager(options.CacheDirectory);
        _packageResolver = new NuGetPackageResolver();
    }

    /// <summary>
    /// Gets runtime path for LoadAsync. Returns cached version immediately,
    /// triggers background check/download if needed.
    /// </summary>
    /// <param name="packageId">The NuGet package ID.</param>
    /// <param name="provider">The execution provider (e.g., "vulkan", "cuda12").</param>
    /// <param name="platform">Platform info.</param>
    /// <param name="currentVersion">Current version from assembly or cache.</param>
    /// <param name="downloadFunc">Function to download runtime (called if not cached).</param>
    /// <param name="progress">Optional progress reporter.</param>
    /// <param name="ct">Cancellation token.</param>
    /// <returns>Path to the runtime binaries.</returns>
    public async Task<string> GetRuntimePathAsync(
        string packageId,
        string provider,
        PlatformInfo platform,
        string currentVersion,
        Func<string, IProgress<DownloadProgress>?, CancellationToken, Task<string>> downloadFunc,
        IProgress<DownloadProgress>? progress = null,
        CancellationToken ct = default)
    {
        ThrowIfDisposed();

        var packageKey = RuntimeVersionStateManager.GetPackageKey(_packageType, provider, platform.RuntimeIdentifier);

        // Initialize or get state
        var state = await _stateManager.GetOrCreateStateAsync(packageKey, currentVersion, ct);

        // If an update is ready, try to apply it
        if (state.UpdateReady && !string.IsNullOrEmpty(state.UpdateReadyPath))
        {
            if (Directory.Exists(state.UpdateReadyPath))
            {
                var readyPath = state.UpdateReadyPath;
                Trace.TraceInformation($"[RuntimeUpdateService] Applying ready update: {state.LatestKnownVersion}");
                await _stateManager.ActivateUpdateAsync(packageKey, _options.MaxVersionsToKeep, ct);
                return readyPath;
            }

            // Update path no longer exists, clear the ready flag
            state.UpdateReady = false;
            state.UpdateReadyPath = null;
            await _stateManager.UpdateStateAsync(packageKey, state, ct);
        }

        // Download current version if needed
        string runtimePath;
        try
        {
            runtimePath = await downloadFunc(currentVersion, progress, ct);
        }
        catch (Exception ex)
        {
            Trace.TraceWarning($"[RuntimeUpdateService] Download failed: {ex.Message}");
            throw;
        }

        // Trigger background update check if auto-update enabled and interval elapsed
        if (_options.AutoDownloadUpdates)
        {
            _ = TriggerBackgroundCheckAsync(packageId, provider, platform, downloadFunc);
        }

        return runtimePath;
    }

    /// <summary>
    /// Checks for updates and applies if available. Blocks until complete.
    /// Used by WarmupAsync for synchronous update application.
    /// </summary>
    public async Task<RuntimeUpdateResult> CheckAndApplyUpdateAsync(
        string packageId,
        string provider,
        PlatformInfo platform,
        string currentVersion,
        Func<string, IProgress<DownloadProgress>?, CancellationToken, Task<string>> downloadFunc,
        IProgress<DownloadProgress>? progress = null,
        CancellationToken ct = default)
    {
        ThrowIfDisposed();

        if (!_options.UpdateOnWarmup)
        {
            return RuntimeUpdateResult.NoUpdateNeeded(currentVersion, string.Empty);
        }

        var packageKey = RuntimeVersionStateManager.GetPackageKey(_packageType, provider, platform.RuntimeIdentifier);
        var state = await _stateManager.GetOrCreateStateAsync(packageKey, currentVersion, ct);

        // Check for latest version
        string? latestVersion;
        try
        {
            using var cts = CancellationTokenSource.CreateLinkedTokenSource(ct);
            cts.CancelAfter(_options.VersionCheckTimeout);

            latestVersion = await _packageResolver.GetLatestVersionAsync(
                packageId,
                _options.IncludePrerelease,
                cts.Token);

            await _stateManager.RecordVersionCheckAsync(packageKey, latestVersion, ct);
        }
        catch (OperationCanceledException) when (!ct.IsCancellationRequested)
        {
            Trace.TraceInformation("[RuntimeUpdateService] Version check timed out");
            return RuntimeUpdateResult.NoUpdateNeeded(currentVersion, string.Empty);
        }
        catch (Exception ex)
        {
            Trace.TraceWarning($"[RuntimeUpdateService] Version check failed: {ex.Message}");
            return RuntimeUpdateResult.NoUpdateNeeded(currentVersion, string.Empty);
        }

        if (string.IsNullOrEmpty(latestVersion))
        {
            return RuntimeUpdateResult.NoUpdateNeeded(currentVersion, string.Empty);
        }

        // Check if update needed
        if (string.Equals(currentVersion, latestVersion, StringComparison.OrdinalIgnoreCase))
        {
            return RuntimeUpdateResult.NoUpdateNeeded(currentVersion, string.Empty);
        }

        // Check if version failed before
        if (state.FailedVersions.Contains(latestVersion))
        {
            Trace.TraceWarning($"[RuntimeUpdateService] Skipping failed version: {latestVersion}");
            return RuntimeUpdateResult.NoUpdateNeeded(currentVersion, string.Empty);
        }

        Trace.TraceInformation($"[RuntimeUpdateService] Updating {currentVersion} -> {latestVersion}");

        // Download new version
        string newPath;
        try
        {
            await _downloadLock.WaitAsync(ct);
            try
            {
                newPath = await downloadFunc(latestVersion, progress, ct);
            }
            finally
            {
                _downloadLock.Release();
            }
        }
        catch (Exception ex) when (ex is not OperationCanceledException || !ct.IsCancellationRequested)
        {
            Trace.TraceWarning($"[RuntimeUpdateService] Update download failed: {ex.Message}");
            return RuntimeUpdateResult.Failed($"Download failed: {ex.Message}");
        }

        // Activate update
        await _stateManager.MarkUpdateReadyAsync(packageKey, latestVersion, newPath, ct);
        await _stateManager.ActivateUpdateAsync(packageKey, _options.MaxVersionsToKeep, ct);

        return RuntimeUpdateResult.UpdateApplied(currentVersion, latestVersion, newPath);
    }

    /// <summary>
    /// Triggers a background version check and download.
    /// Non-blocking, runs asynchronously.
    /// </summary>
    public Task TriggerBackgroundCheckAsync(
        string packageId,
        string provider,
        PlatformInfo platform,
        Func<string, IProgress<DownloadProgress>?, CancellationToken, Task<string>> downloadFunc)
    {
        ThrowIfDisposed();

        var taskKey = $"{packageId}|{provider}|{platform.RuntimeIdentifier}";

        // Only one background task per package
        if (_backgroundTasks.ContainsKey(taskKey))
            return Task.CompletedTask;

        var task = Task.Run(async () =>
        {
            try
            {
                await BackgroundCheckAndDownloadAsync(packageId, provider, platform, downloadFunc);
            }
            finally
            {
                _backgroundTasks.TryRemove(taskKey, out _);
            }
        });

        _backgroundTasks[taskKey] = task;
        return Task.CompletedTask;
    }

    private async Task BackgroundCheckAndDownloadAsync(
        string packageId,
        string provider,
        PlatformInfo platform,
        Func<string, IProgress<DownloadProgress>?, CancellationToken, Task<string>> downloadFunc)
    {
        var packageKey = RuntimeVersionStateManager.GetPackageKey(_packageType, provider, platform.RuntimeIdentifier);

        // Check if interval elapsed
        var isDue = await _stateManager.IsVersionCheckDueAsync(packageKey, _options.VersionCheckInterval);
        if (!isDue)
        {
            Trace.TraceInformation("[RuntimeUpdateService] Background check skipped (not due)");
            return;
        }

        // Get current state
        var state = await _stateManager.GetStateAsync(packageKey);
        if (state is null)
            return;

        // Check for latest version
        string? latestVersion;
        try
        {
            using var cts = new CancellationTokenSource(_options.VersionCheckTimeout);
            latestVersion = await _packageResolver.GetLatestVersionAsync(
                packageId, _options.IncludePrerelease, cts.Token);

            await _stateManager.RecordVersionCheckAsync(packageKey, latestVersion);
        }
        catch (Exception ex)
        {
            Trace.TraceWarning($"[RuntimeUpdateService] Background version check failed: {ex.Message}");
            return;
        }

        if (string.IsNullOrEmpty(latestVersion))
            return;

        // Check if update needed
        if (string.Equals(state.InstalledVersion, latestVersion, StringComparison.OrdinalIgnoreCase))
            return;

        // Check if already ready or pending
        if (state.UpdateReady || state.PendingVersion == latestVersion)
            return;

        // Check if version failed before
        if (state.FailedVersions.Contains(latestVersion))
            return;

        Trace.TraceInformation($"[RuntimeUpdateService] Background downloading: {latestVersion}");

        // Mark as pending
        await _stateManager.MarkPendingDownloadAsync(packageKey, latestVersion);

        // Download
        try
        {
            await _downloadLock.WaitAsync();
            try
            {
                var newPath = await downloadFunc(latestVersion, null, CancellationToken.None);
                await _stateManager.MarkUpdateReadyAsync(packageKey, latestVersion, newPath);
                Trace.TraceInformation($"[RuntimeUpdateService] Background download complete: {latestVersion}");
            }
            finally
            {
                _downloadLock.Release();
            }
        }
        catch (Exception ex)
        {
            Trace.TraceWarning($"[RuntimeUpdateService] Background download failed: {ex.Message}");
            await _stateManager.ClearPendingDownloadAsync(packageKey);
        }
    }

    /// <summary>
    /// Gets update information for diagnostics.
    /// </summary>
    public async Task<RuntimeUpdateInfo> GetUpdateInfoAsync(
        string provider,
        PlatformInfo platform,
        CancellationToken ct = default)
    {
        ThrowIfDisposed();

        var packageKey = RuntimeVersionStateManager.GetPackageKey(_packageType, provider, platform.RuntimeIdentifier);
        var state = await _stateManager.GetStateAsync(packageKey, ct);

        if (state is null)
        {
            return new RuntimeUpdateInfo
            {
                InstalledVersion = "unknown",
                Provider = provider
            };
        }

        return new RuntimeUpdateInfo
        {
            InstalledVersion = state.InstalledVersion,
            LatestVersion = state.LatestKnownVersion,
            UpdateReady = state.UpdateReady,
            UpdateReadyPath = state.UpdateReadyPath,
            LastChecked = state.LastVersionCheck,
            Provider = provider
        };
    }

    /// <summary>
    /// Handles a failed version load by rolling back to previous version.
    /// </summary>
    public async Task<RuntimeUpdateResult> HandleLoadFailureAsync(
        string provider,
        PlatformInfo platform,
        string failedVersion,
        Func<string, IProgress<DownloadProgress>?, CancellationToken, Task<string>> downloadFunc,
        CancellationToken ct = default)
    {
        ThrowIfDisposed();

        var packageKey = RuntimeVersionStateManager.GetPackageKey(_packageType, provider, platform.RuntimeIdentifier);

        var (previousVersion, _) = await _stateManager.RollbackAsync(packageKey, failedVersion, ct);

        if (string.IsNullOrEmpty(previousVersion))
        {
            return RuntimeUpdateResult.Failed($"No previous version available for rollback from {failedVersion}");
        }

        Trace.TraceWarning($"[RuntimeUpdateService] Rolling back from {failedVersion} to {previousVersion}");

        // Download previous version
        try
        {
            var path = await downloadFunc(previousVersion, null, ct);
            return RuntimeUpdateResult.Rollback(failedVersion, previousVersion, path);
        }
        catch (Exception ex) when (ex is not OperationCanceledException || !ct.IsCancellationRequested)
        {
            return RuntimeUpdateResult.Failed($"Rollback failed: {ex.Message}");
        }
    }

    /// <summary>
    /// Cleans up old versions beyond the retention limit.
    /// </summary>
    public async Task CleanupOldVersionsAsync(
        string provider,
        PlatformInfo platform,
        string cacheBasePath,
        CancellationToken ct = default)
    {
        ThrowIfDisposed();

        var packageKey = RuntimeVersionStateManager.GetPackageKey(_packageType, provider, platform.RuntimeIdentifier);
        var state = await _stateManager.GetStateAsync(packageKey, ct);

        if (state is null)
            return;

        // Get all versions to keep
        var versionsToKeep = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
        {
            state.InstalledVersion
        };

        foreach (var prev in state.PreviousVersions.Take(_options.MaxVersionsToKeep))
        {
            versionsToKeep.Add(prev);
        }

        // Find and delete old versions
        var providerPath = Path.Combine(cacheBasePath, _packageType, provider.ToLowerInvariant());
        if (!Directory.Exists(providerPath))
            return;

        foreach (var versionDir in Directory.GetDirectories(providerPath))
        {
            var version = Path.GetFileName(versionDir);
            if (!versionsToKeep.Contains(version))
            {
                try
                {
                    Directory.Delete(versionDir, recursive: true);
                    Trace.TraceInformation($"[RuntimeUpdateService] Cleaned up old version: {version}");
                }
                catch (Exception ex)
                {
                    Trace.TraceWarning($"[RuntimeUpdateService] Failed to clean up {version}: {ex.Message}");
                }
            }
        }
    }

    private void ThrowIfDisposed()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
    }

    public async ValueTask DisposeAsync()
    {
        if (_disposed)
            return;

        _disposed = true;

        // Wait for background tasks to complete
        var tasks = _backgroundTasks.Values.ToArray();
        if (tasks.Length > 0)
        {
            try
            {
                await Task.WhenAll(tasks).WaitAsync(TimeSpan.FromSeconds(5));
            }
            catch (Exception ex)
            {
                Trace.TraceInformation($"[RuntimeUpdateService] Dispose wait failed: {ex.Message}");
            }
        }

        _stateManager.Dispose();
        _packageResolver.Dispose();
        _downloadLock.Dispose();

        _instances.TryRemove(_packageType, out _);
    }
}
