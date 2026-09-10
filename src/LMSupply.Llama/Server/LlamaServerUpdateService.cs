using System.Diagnostics;
using System.Runtime.InteropServices;
using LMSupply.Download;

namespace LMSupply.Llama.Server;

/// <summary>
/// Service for managing llama-server auto-updates.
/// Handles version checking, background downloads, and rollback.
/// </summary>
public sealed class LlamaServerUpdateService : IAsyncDisposable
{
    private static readonly Lazy<LlamaServerUpdateService> _instance = new(
        () => new LlamaServerUpdateService(),
        LazyThreadSafetyMode.ExecutionAndPublication);

    /// <summary>
    /// Gets the singleton instance.
    /// </summary>
    public static LlamaServerUpdateService Instance => _instance.Value;

    private static readonly System.Runtime.CompilerServices.ConditionalWeakTable<LlamaServerUpdateOptions, LlamaServerUpdateService> s_scopedInstances = new();

    /// <summary>
    /// Resolves the service instance to use for a load: the process-wide <see cref="Instance"/> when
    /// <paramref name="options"/> is null, or a dedicated instance scoped to that options object
    /// otherwise. A non-null options instance is memoized by reference identity, so a consumer that
    /// holds one <see cref="LlamaServerUpdateOptions"/> and passes it on every
    /// GeneratorOptions/EmbedderOptions/RerankerOptions load shares a single background-update timer
    /// and state file across loads instead of spawning a duplicate per load. Constructing a fresh
    /// options instance on every call still works correctly — each gets its own isolated service —
    /// it just forgoes that sharing.
    /// </summary>
    public static LlamaServerUpdateService Resolve(LlamaServerUpdateOptions? options)
        => options is null
            ? Instance
            : s_scopedInstances.GetValue(options, static o => new LlamaServerUpdateService(o));

    private readonly LlamaServerStateManager _stateManager;
    private readonly LlamaServerDownloader _downloader;
    private readonly LlamaServerUpdateOptions _options;
    private readonly SemaphoreSlim _updateLock = new(1, 1);
    private Task? _backgroundCheckTask;
    private CancellationTokenSource? _backgroundCts;
    private bool _disposed;

    /// <summary>
    /// Creates a new update service with default options.
    /// </summary>
    public LlamaServerUpdateService()
        : this(LlamaServerUpdateOptions.Default)
    {
    }

    /// <summary>
    /// Creates a new update service with custom options.
    /// </summary>
    /// <param name="options">Update/acquisition policy.</param>
    /// <param name="httpClient">
    /// Optional HTTP client for GitHub Releases calls. Creates a new one if null. Exposed mainly
    /// so tests can substitute a fake handler and assert on network call counts (e.g. that a
    /// <see cref="LlamaServerUpdateOptions.PinnedVersion"/> cache hit makes zero calls).
    /// </param>
    public LlamaServerUpdateService(LlamaServerUpdateOptions options, HttpClient? httpClient = null)
    {
        _options = options;
        _stateManager = new LlamaServerStateManager(options.CacheDirectory);
        _downloader = new LlamaServerDownloader(options.CacheDirectory, httpClient, options.IncludePrerelease);
    }

    /// <summary>
    /// Gets the current platform identifier.
    /// </summary>
    public static string GetCurrentPlatform()
    {
        var os = OperatingSystem.IsWindows() ? "win"
            : OperatingSystem.IsMacOS() ? "osx"
            : "linux";

        var arch = RuntimeInformation.OSArchitecture switch
        {
            Architecture.Arm64 => "arm64",
            _ => "x64"
        };

        return $"{os}-{arch}";
    }

    /// <summary>
    /// Gets the server path for the specified backend, downloading if necessary.
    /// Uses cached version immediately, triggers background update check.
    /// </summary>
    public async Task<LlamaServerUpdateResult> GetServerPathAsync(
        LlamaServerBackend backend,
        IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        if (!string.IsNullOrEmpty(_options.ServerBinaryPath))
        {
            // Consumer supplied the binary directly. Acquisition — including the cudart companion
            // fetch below — is entirely their responsibility: no network calls, no state tracking.
            return File.Exists(_options.ServerBinaryPath)
                ? LlamaServerUpdateResult.NoUpdate(_options.ServerBinaryPath, backend, _options.PinnedVersion ?? "external")
                : LlamaServerUpdateResult.Failed(
                    _options.ServerBinaryPath, backend,
                    $"LlamaServerUpdateOptions.ServerBinaryPath '{_options.ServerBinaryPath}' does not exist.");
        }

        var result = await GetServerPathCoreAsync(backend, progress, cancellationToken);

        // For a CUDA backend, ensure the cudart runtime is present next to the binary (it ships as a
        // separate llama.cpp asset). Self-gating (no-op for non-CUDA) and best-effort. Runs on both
        // cache-hit and fresh-download results, so binaries cached before this logic existed are
        // backfilled. See LlamaServerDownloader.EnsureCudaRuntimeAsync.
        if (result.Success && !string.IsNullOrEmpty(result.ServerPath))
        {
            var versionDir = Path.GetDirectoryName(result.ServerPath);
            var version = result.NewVersion ?? result.PreviousVersion;
            if (!string.IsNullOrEmpty(versionDir) && !string.IsNullOrEmpty(version))
            {
                await _downloader.EnsureCudaRuntimeAsync(
                    versionDir, result.Backend, version, progress, cancellationToken);
            }
        }

        return result;
    }

    private async Task<LlamaServerUpdateResult> GetServerPathCoreAsync(
        LlamaServerBackend backend,
        IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        if (!string.IsNullOrEmpty(_options.PinnedVersion))
        {
            // A pinned version bypasses the state/rollback machinery entirely: it never
            // participates in auto-update tracking, so there is nothing to reconcile against a
            // previously-recorded "latest" installation.
            return await GetPinnedVersionResultAsync(_options.PinnedVersion, backend, progress, cancellationToken);
        }

        var platform = GetCurrentPlatform();
        var backendStr = backend.ToString().ToLowerInvariant();

        // Check existing state
        var state = await _stateManager.GetStateAsync(backendStr, platform, cancellationToken);

        if (state != null)
        {
            // Check if pending update should be applied
            if (state.UpdateReady && state.PendingPath != null)
            {
                var serverExe = GetServerExecutablePath(state.PendingPath);
                if (File.Exists(serverExe))
                {
                    // Apply the update
                    var activatedState = await _stateManager.ActivateUpdateAsync(
                        backendStr, platform, _options.MaxVersionsToKeep, cancellationToken);

                    if (activatedState != null)
                    {
                        await CleanupSupersededBestEffortAsync(cancellationToken);
                        TriggerBackgroundCheck(backend);
                        return LlamaServerUpdateResult.WithUpdate(
                            serverExe, backend, state.InstalledVersion, activatedState.InstalledVersion);
                    }
                }
            }

            // Use existing installation
            var existingExe = GetServerExecutablePath(state.InstalledPath);
            if (File.Exists(existingExe))
            {
                TriggerBackgroundCheck(backend);
                return LlamaServerUpdateResult.NoUpdate(existingExe, backend, state.InstalledVersion);
            }
        }

        // Check for existing cached versions before downloading
        var cachedVersions = _downloader.GetCachedVersions();
        foreach (var cachedVersion in cachedVersions)
        {
            var cachedPath = _downloader.GetCachedServerPath(cachedVersion, backend);
            if (cachedPath != null)
            {
                // Found existing cache without state - create state for it
                var versionDir = Path.GetDirectoryName(cachedPath)!;
                await _stateManager.CreateInitialStateAsync(backendStr, platform, cachedVersion, versionDir, cancellationToken);

                TriggerBackgroundCheck(backend);
                return LlamaServerUpdateResult.NoUpdate(cachedPath, backend, cachedVersion);
            }
        }

        // No cached version found, download latest
        progress?.Report(new DownloadProgress
        {
            FileName = "llama-server",
            Phase = DownloadPhase.Downloading
        });

        // Resolve the asset once and take the version from it: the resolved build tag is what the
        // download lands under, so it is also what the state file must record. (Re-asking "latest"
        // afterwards was a second network call that could name a different build than the one
        // just downloaded, and fell back to "unknown" on any hiccup.)
        var resolution = await _downloader.ResolveAssetAsync(version: null, preferredBackend: backend, cancellationToken);
        var asset = resolution.Asset
            ?? throw new LlamaServerAcquisitionException(resolution);

        var serverPath = await _downloader.DownloadAsync(asset, progress, cancellationToken);
        var version = asset.Version;

        // Create initial state
        var versionDir2 = Path.GetDirectoryName(serverPath)!;
        await _stateManager.CreateInitialStateAsync(backendStr, platform, version, versionDir2, cancellationToken);

        TriggerBackgroundCheck(backend);
        return LlamaServerUpdateResult.NoUpdate(serverPath, backend, version);
    }

    /// <summary>
    /// Resolves a pinned version without ever calling <see cref="LlamaServerDownloader.GetLatestVersionAsync"/>.
    /// A cache hit makes zero network calls; a cache miss downloads exactly the pinned asset. Does
    /// not touch <see cref="_stateManager"/> or trigger a background check — a pinned installation
    /// stays outside the auto-update/rollback machinery entirely.
    /// </summary>
    private async Task<LlamaServerUpdateResult> GetPinnedVersionResultAsync(
        string pinnedVersion,
        LlamaServerBackend backend,
        IProgress<DownloadProgress>? progress,
        CancellationToken cancellationToken)
    {
        var cachedPath = _downloader.GetCachedServerPath(pinnedVersion, backend);
        if (cachedPath != null)
            return LlamaServerUpdateResult.NoUpdate(cachedPath, backend, pinnedVersion);

        progress?.Report(new DownloadProgress
        {
            FileName = "llama-server",
            Phase = DownloadPhase.Downloading
        });

        var serverPath = await _downloader.EnsureServerAsync(
            version: pinnedVersion,
            preferredBackend: backend,
            progress: progress,
            cancellationToken: cancellationToken);

        return LlamaServerUpdateResult.NoUpdate(serverPath, backend, pinnedVersion);
    }

    /// <summary>
    /// Checks for updates and applies them immediately if available.
    /// Used during WarmupAsync when UpdateOnWarmup is true.
    /// </summary>
    public async Task<LlamaServerUpdateResult> CheckAndApplyUpdateAsync(
        LlamaServerBackend backend,
        IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        // A pinned installation (exact version or an externally supplied binary) never re-checks
        // for or auto-applies a newer version — that is the whole point of pinning for an
        // offline/security-reviewed deployment (see LlamaServerUpdateOptions.PinnedVersion/
        // ServerBinaryPath). Route straight to the pinned/external resolution instead.
        if (!string.IsNullOrEmpty(_options.ServerBinaryPath) || !string.IsNullOrEmpty(_options.PinnedVersion))
        {
            return await GetServerPathAsync(backend, progress, cancellationToken);
        }

        await _updateLock.WaitAsync(cancellationToken);
        try
        {
            var platform = GetCurrentPlatform();
            var backendStr = backend.ToString().ToLowerInvariant();

            var state = await _stateManager.GetStateAsync(backendStr, platform, cancellationToken);

            if (state == null)
            {
                // No existing state, just get latest
                return await GetServerPathAsync(backend, progress, cancellationToken);
            }

            // Check for new version
            var latestVersion = await _downloader.GetLatestVersionAsync(cancellationToken);
            await _stateManager.RecordVersionCheckAsync(backendStr, platform, latestVersion, cancellationToken);

            if (latestVersion == null ||
                string.Equals(state.InstalledVersion, latestVersion, StringComparison.OrdinalIgnoreCase))
            {
                // Already up to date
                var existingPath = GetServerExecutablePath(state.InstalledPath);
                return LlamaServerUpdateResult.NoUpdate(existingPath, backend, state.InstalledVersion);
            }

            // Check if this version has failed before
            if (state.FailedVersions.Contains(latestVersion))
            {
                var existingPath = GetServerExecutablePath(state.InstalledPath);
                return LlamaServerUpdateResult.NoUpdate(existingPath, backend, state.InstalledVersion);
            }

            // Download new version
            progress?.Report(new DownloadProgress
            {
                FileName = $"llama-server {latestVersion}",
                Phase = DownloadPhase.Downloading
            });

            try
            {
                var newServerPath = await _downloader.EnsureServerAsync(
                    version: latestVersion,
                    preferredBackend: backend,
                    progress: progress,
                    cancellationToken: cancellationToken);

                var newVersionDir = Path.GetDirectoryName(newServerPath)!;

                // Mark update ready
                await _stateManager.MarkUpdateReadyAsync(backendStr, platform, latestVersion, newVersionDir, cancellationToken);

                // Activate immediately
                var activatedState = await _stateManager.ActivateUpdateAsync(
                    backendStr, platform, _options.MaxVersionsToKeep, cancellationToken);

                if (activatedState != null)
                {
                    await CleanupSupersededBestEffortAsync(cancellationToken);
                    return LlamaServerUpdateResult.WithUpdate(
                        newServerPath, backend, state.InstalledVersion, latestVersion);
                }

                // Activation failed, use existing
                var existingPath = GetServerExecutablePath(state.InstalledPath);
                return LlamaServerUpdateResult.NoUpdate(existingPath, backend, state.InstalledVersion);
            }
            catch (Exception ex)
            {
                // Download/update failed, use existing
                var existingPath = GetServerExecutablePath(state.InstalledPath);
                return LlamaServerUpdateResult.Failed(existingPath, backend, ex.Message);
            }
        }
        finally
        {
            _updateLock.Release();
        }
    }

    /// <summary>
    /// Gets update information for the specified backend.
    /// </summary>
    public async Task<LlamaServerUpdateInfo?> GetUpdateInfoAsync(
        LlamaServerBackend backend,
        CancellationToken cancellationToken = default)
    {
        var platform = GetCurrentPlatform();
        var backendStr = backend.ToString().ToLowerInvariant();

        var state = await _stateManager.GetStateAsync(backendStr, platform, cancellationToken);
        if (state == null)
            return null;

        return new LlamaServerUpdateInfo
        {
            InstalledVersion = state.InstalledVersion,
            LatestVersion = state.LatestKnownVersion,
            UpdateReady = state.UpdateReady,
            PendingVersion = state.PendingVersion,
            LastChecked = state.LastVersionCheck,
            Backend = backend,
            ServerPath = GetServerExecutablePath(state.InstalledPath)
        };
    }

    /// <summary>
    /// Rolls back to the previous version if available.
    /// </summary>
    public async Task<LlamaServerUpdateResult> RollbackAsync(
        LlamaServerBackend backend,
        CancellationToken cancellationToken = default)
    {
        var platform = GetCurrentPlatform();
        var backendStr = backend.ToString().ToLowerInvariant();

        var state = await _stateManager.RollbackAsync(backendStr, platform, cancellationToken);
        if (state == null)
        {
            return LlamaServerUpdateResult.Failed("", backend, "No previous version available for rollback");
        }

        var serverPath = GetServerExecutablePath(state.InstalledPath);
        return LlamaServerUpdateResult.NoUpdate(serverPath, backend, state.InstalledVersion);
    }

    /// <summary>
    /// Triggers a background check for updates.
    /// </summary>
    private void TriggerBackgroundCheck(LlamaServerBackend backend)
    {
        if (!_options.AutoDownloadUpdates)
            return;

        // Don't start if already running
        if (_backgroundCheckTask != null && !_backgroundCheckTask.IsCompleted)
            return;

        _backgroundCts?.Cancel();
        _backgroundCts = new CancellationTokenSource();
        _backgroundCheckTask = BackgroundCheckAsync(backend, _backgroundCts.Token);
    }

    private async Task BackgroundCheckAsync(LlamaServerBackend backend, CancellationToken cancellationToken)
    {
        try
        {
            // Small delay to avoid blocking the main path
            await Task.Delay(1000, cancellationToken);

            // Off the critical path: reclaim superseded builds accumulated by earlier LMSupply
            // versions (the retention gap left directories behind forever — 24 GB observed).
            await CleanupSupersededBestEffortAsync(cancellationToken);

            var platform = GetCurrentPlatform();
            var backendStr = backend.ToString().ToLowerInvariant();

            var state = await _stateManager.GetStateAsync(backendStr, platform, cancellationToken);
            if (state == null)
                return;

            // Check if enough time has passed since last check
            var timeSinceLastCheck = DateTimeOffset.UtcNow - state.LastVersionCheck;
            if (timeSinceLastCheck < _options.VersionCheckInterval)
                return;

            // Check for new version
            var latestVersion = await _downloader.GetLatestVersionAsync(cancellationToken);
            await _stateManager.RecordVersionCheckAsync(backendStr, platform, latestVersion, cancellationToken);

            if (latestVersion == null ||
                string.Equals(state.InstalledVersion, latestVersion, StringComparison.OrdinalIgnoreCase))
            {
                return; // Already up to date
            }

            // Skip failed versions
            if (state.FailedVersions.Contains(latestVersion))
                return;

            // Download in background
            await _updateLock.WaitAsync(cancellationToken);
            try
            {
                var serverPath = await _downloader.EnsureServerAsync(
                    version: latestVersion,
                    preferredBackend: backend,
                    progress: null,
                    cancellationToken: cancellationToken);

                var versionDir = Path.GetDirectoryName(serverPath)!;
                await _stateManager.MarkUpdateReadyAsync(backendStr, platform, latestVersion, versionDir, cancellationToken);
            }
            finally
            {
                _updateLock.Release();
            }
        }
        catch (OperationCanceledException)
        {
            // Expected when disposing
        }
        catch (Exception ex)
        {
            Trace.TraceInformation($"[LlamaServerUpdateService] Background update error: {ex.Message}");
        }
    }

    private async Task CleanupSupersededBestEffortAsync(CancellationToken cancellationToken)
    {
        try
        {
            await _stateManager.CleanupUnreferencedVersionsAsync(cancellationToken);
        }
        catch (OperationCanceledException)
        {
            throw;
        }
        catch (Exception ex)
        {
            Trace.TraceInformation($"[LlamaServerUpdateService] Superseded-build cleanup failed: {ex.Message}");
        }
    }

    private static string GetServerExecutablePath(string directory)
    {
        var executable = OperatingSystem.IsWindows() ? "llama-server.exe" : "llama-server";
        return Path.Combine(directory, executable);
    }

    public async ValueTask DisposeAsync()
    {
        if (_disposed)
            return;

        _disposed = true;

        _backgroundCts?.Cancel();
        if (_backgroundCheckTask != null)
        {
            try
            {
                await _backgroundCheckTask;
            }
            catch (OperationCanceledException)
            {
                // Expected
            }
        }

        _backgroundCts?.Dispose();
        _updateLock.Dispose();
        _stateManager.Dispose();
        _downloader.Dispose();
    }
}
