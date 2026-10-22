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
        _downloader = new LlamaServerDownloader(options.CacheDirectory, httpClient, options.IncludePrerelease, options.ApiTimeout);
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
    /// By default uses the cached version immediately and triggers a background update check; with
    /// <see cref="LlamaServerUpdateOptions.UpdateOnWarmup"/> it first checks for a newer build and
    /// applies it (<see cref="CheckAndApplyUpdateAsync"/>).
    /// </summary>
    public Task<LlamaServerUpdateResult> GetServerPathAsync(
        LlamaServerBackend backend,
        IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
        => GetServerPathAsync(backend, localFilesOnly: false, progress, cancellationToken);

    /// <summary>
    /// The llama-server a model load runs: acquired as <see cref="GetServerPathAsync(LlamaServerBackend, IProgress{DownloadProgress}?, CancellationToken)"/>
    /// does, or — with <paramref name="localFilesOnly"/> — from the external binary, the pinned version's cache or the cached
    /// installation only: no download, no update check (also not in the background), no CUDA runtime fetch, so a load
    /// with downloads disabled (<c>DisableAutoDownload</c>) makes no network request at all.
    /// </summary>
    /// <exception cref="LMSupply.Exceptions.ModelNotFoundException"><paramref name="localFilesOnly"/> and no server is cached (<c>ModelId</c> <c>"llama-server"</c>).</exception>
    /// <exception cref="InvalidOperationException">The server could not be acquired.</exception>
    public async Task<LlamaServerInstallation> AcquireServerAsync(
        LlamaServerBackend backend,
        bool localFilesOnly,
        IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        var result = await GetServerPathAsync(backend, localFilesOnly, progress, cancellationToken);
        if (result.Success)
            return new LlamaServerInstallation(result.ServerPath, result.Backend, result.NewVersion ?? result.PreviousVersion);
        if (result.NotCached)
            throw new LMSupply.Exceptions.ModelNotFoundException(result.Error!, "llama-server");
        throw new InvalidOperationException($"Failed to get llama-server: {result.Error}");
    }

    private async Task<LlamaServerUpdateResult> GetServerPathAsync(
        LlamaServerBackend backend,
        bool localFilesOnly,
        IProgress<DownloadProgress>? progress,
        CancellationToken cancellationToken)
    {
        progress = RuntimeDownloadProgress.Wrap(progress);
        LlamaServerUpdateResult result;
        if (_options.UpdateOnWarmup && !IsPinned && !localFilesOnly)
        {
            result = await CheckAndApplyUpdateAsync(backend, progress, cancellationToken);
            await EnsureCudaRuntimeAsync(result, progress, cancellationToken);
        }
        else
        {
            result = await AcquireAsync(backend, progress, localFilesOnly, cancellationToken);
        }

        return await HoldToSecurityFloorAsync(result, backend, progress, localFilesOnly, cancellationToken);
    }

    // A build below the security floor is replaced by the latest when it may be (nothing pinned, downloads allowed)
    // and refused otherwise, unless the options allow it — then it is run, with a warning.
    private async Task<LlamaServerUpdateResult> HoldToSecurityFloorAsync(
        LlamaServerUpdateResult result,
        LlamaServerBackend backend,
        IProgress<DownloadProgress>? progress,
        bool localFilesOnly,
        CancellationToken cancellationToken)
    {
        var version = result.NewVersion ?? result.PreviousVersion;
        if (!result.Success || !LlamaServerVersionRequirements.IsBelowSecurityFloor(version))
            return result;

        if (_options.AllowVulnerableBuild)
        {
            Trace.TraceWarning($"[LlamaServerUpdateService] {BelowFloor(version)} It runs because AllowVulnerableBuild is set.");
            return result;
        }

        if (!IsPinned && !localFilesOnly)
        {
            var updated = await CheckAndApplyUpdateAsync(backend, progress, cancellationToken);
            if (updated.Success && !LlamaServerVersionRequirements.IsBelowSecurityFloor(updated.NewVersion ?? updated.PreviousVersion))
            {
                await EnsureCudaRuntimeAsync(updated, progress, cancellationToken);
                return updated;
            }
        }

        // Unpinned with downloads disabled, the cache holds no server a load may run: the same answer as an empty cache,
        // so a check of what is available offline says no and a download brings the latest build.
        if (!IsPinned && localFilesOnly)
        {
            return LlamaServerUpdateResult.NotInCache(backend,
                $"{BelowFloor(version)} Downloads are disabled (DisableAutoDownload): load once while they are allowed, so the " +
                "latest build replaces it, or set LlamaServerUpdateOptions.AllowVulnerableBuild to run it anyway.");
        }

        var remedy = IsPinned ? "Pin or supply a newer build" : "The latest build could not be acquired to replace it; try again once GitHub is reachable";
        return LlamaServerUpdateResult.Failed(
            result.ServerPath, backend,
            $"{BelowFloor(version)} {remedy}, or set LlamaServerUpdateOptions.AllowVulnerableBuild to run it anyway.");
    }

    private static string BelowFloor(string? version) =>
        $"llama-server {version} is below b{LlamaServerVersionRequirements.MinimumSecureBuild}, the oldest build no published " +
        "llama.cpp security advisory affects.";

    private string NotCachedMessage(LlamaServerBackend backend, string? version) =>
        $"llama-server ({backend.ToString().ToLowerInvariant()}{(version is null ? "" : $", {version}")}) is not in the cache '{_downloader.CacheDirectory}', " +
        "and downloads are disabled (DisableAutoDownload). Download the model while downloads are allowed, or point " +
        "LlamaServerUpdateOptions.ServerBinaryPath at a server shipped with the application.";

    // A pinned or externally supplied installation never checks for, or applies, a newer version.
    private bool IsPinned => !string.IsNullOrEmpty(_options.ServerBinaryPath) || !string.IsNullOrEmpty(_options.PinnedVersion);

    /// <summary>
    /// Acquires the server without checking for a newer build first: the external binary, the pinned
    /// version, the cached installation (with a background check), or a first download.
    /// </summary>
    private async Task<LlamaServerUpdateResult> AcquireAsync(
        LlamaServerBackend backend,
        IProgress<DownloadProgress>? progress,
        bool localFilesOnly,
        CancellationToken cancellationToken)
    {
        if (!string.IsNullOrEmpty(_options.ServerBinaryPath))
        {
            // Consumer supplied the binary directly. Acquisition — including the cudart companion
            // fetch below — is entirely their responsibility: no network calls, no state tracking.
            if (!File.Exists(_options.ServerBinaryPath))
            {
                return LlamaServerUpdateResult.Failed(
                    _options.ServerBinaryPath, backend,
                    $"LlamaServerUpdateOptions.ServerBinaryPath '{_options.ServerBinaryPath}' does not exist.");
            }

            // Its build still decides which flag spellings it parses, so ask the binary itself
            // (a declared PinnedVersion wins). "external" means the build could not be read.
            var version = _options.PinnedVersion
                ?? await LlamaServerBinaryVersion.ProbeAsync(_options.ServerBinaryPath, cancellationToken)
                ?? "external";
            return LlamaServerUpdateResult.NoUpdate(_options.ServerBinaryPath, backend, version);
        }

        var result = await GetServerPathCoreAsync(backend, progress, localFilesOnly, cancellationToken);
        if (!localFilesOnly)
            await EnsureCudaRuntimeAsync(result, progress, cancellationToken);
        return result;
    }

    // For a CUDA backend, ensure the cudart runtime is present next to the binary (it ships as a
    // separate llama.cpp asset). Self-gating (no-op for non-CUDA) and best-effort. Runs on both
    // cache-hit and fresh-download results, so binaries cached before this logic existed are
    // backfilled. See LlamaServerDownloader.EnsureCudaRuntimeAsync.
    private async Task EnsureCudaRuntimeAsync(
        LlamaServerUpdateResult result,
        IProgress<DownloadProgress>? progress,
        CancellationToken cancellationToken)
    {
        if (!result.Success || string.IsNullOrEmpty(result.ServerPath))
            return;

        var versionDir = Path.GetDirectoryName(result.ServerPath);
        var version = result.NewVersion ?? result.PreviousVersion;
        if (!string.IsNullOrEmpty(versionDir) && !string.IsNullOrEmpty(version))
        {
            await _downloader.EnsureCudaRuntimeAsync(
                versionDir, result.Backend, version, progress, cancellationToken);
        }
    }

    private async Task<LlamaServerUpdateResult> GetServerPathCoreAsync(
        LlamaServerBackend backend,
        IProgress<DownloadProgress>? progress,
        bool localFilesOnly,
        CancellationToken cancellationToken)
    {
        if (!string.IsNullOrEmpty(_options.PinnedVersion))
        {
            // A pinned version bypasses the state/rollback machinery entirely: it never
            // participates in auto-update tracking, so there is nothing to reconcile against a
            // previously-recorded "latest" installation.
            return await GetPinnedVersionResultAsync(_options.PinnedVersion, backend, progress, localFilesOnly, cancellationToken);
        }

        var platform = GetCurrentPlatform();
        var backendStr = backend.ToString().ToLowerInvariant();

        // Check existing state
        var state = await _stateManager.GetStateAsync(backendStr, platform, cancellationToken);

        if (state != null)
        {
            // Check if pending update should be applied
            if (state.UpdateReady && state.PendingVersion != null)
            {
                var serverExe = ResolvePendingServer(state);
                if (serverExe != null)
                {
                    // Apply the update
                    var activatedState = await _stateManager.ActivateUpdateAsync(
                        backendStr, platform, _options.MaxVersionsToKeep, cancellationToken);

                    if (activatedState != null)
                    {
                        await CleanupSupersededBestEffortAsync(cancellationToken);
                        if (!localFilesOnly)
                            TriggerBackgroundCheck(backend);
                        return LlamaServerUpdateResult.WithUpdate(
                            serverExe, backend, state.InstalledVersion, activatedState.InstalledVersion);
                    }
                }
            }

            // Use existing installation
            var existingExe = ResolveInstalledServer(state);
            if (existingExe != null)
            {
                if (!localFilesOnly)
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
                // Found a cached build the state does not record (no state yet, or the recorded build is gone):
                // adopt it, keeping the entry's rollback history where those builds still exist
                await _stateManager.CreateInitialStateAsync(
                    backendStr, platform, cachedVersion, BuildBackendOf(cachedPath), cancellationToken);

                if (!localFilesOnly)
                    TriggerBackgroundCheck(backend);
                return LlamaServerUpdateResult.NoUpdate(cachedPath, backend, cachedVersion);
            }
        }

        if (localFilesOnly)
            return LlamaServerUpdateResult.NotInCache(backend, NotCachedMessage(backend, version: null));

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
        var asset = await _downloader.ResolveAssetOrThrowAsync(version: null, preferredBackend: backend, cancellationToken);

        var serverPath = await _downloader.DownloadAsync(asset, progress, cancellationToken);
        var version = asset.Version;

        // Create initial state
        await _stateManager.CreateInitialStateAsync(backendStr, platform, version, BuildBackendOf(serverPath), cancellationToken);

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
        bool localFilesOnly,
        CancellationToken cancellationToken)
    {
        var cachedPath = _downloader.GetCachedServerPath(pinnedVersion, backend);
        if (cachedPath != null)
            return LlamaServerUpdateResult.NoUpdate(cachedPath, backend, pinnedVersion);
        if (localFilesOnly)
            return LlamaServerUpdateResult.NotInCache(backend, NotCachedMessage(backend, pinnedVersion));

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
    /// Checks for updates and applies them immediately if available. <see cref="GetServerPathAsync(LlamaServerBackend, IProgress{DownloadProgress}?, CancellationToken)"/>
    /// calls it when <see cref="LlamaServerUpdateOptions.UpdateOnWarmup"/> is set; the generator calls it
    /// when the cached build is too old for a model.
    /// </summary>
    public async Task<LlamaServerUpdateResult> CheckAndApplyUpdateAsync(
        LlamaServerBackend backend,
        IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        progress = RuntimeDownloadProgress.Wrap(progress);
        // A pinned installation (exact version or an externally supplied binary) never re-checks
        // for or auto-applies a newer version — that is the whole point of pinning for an
        // offline/security-reviewed deployment (see LlamaServerUpdateOptions.PinnedVersion/
        // ServerBinaryPath). Route straight to the pinned/external resolution instead.
        if (IsPinned)
        {
            return await AcquireAsync(backend, progress, localFilesOnly: false, cancellationToken);
        }

        await _updateLock.WaitAsync(cancellationToken);
        try
        {
            var platform = GetCurrentPlatform();
            var backendStr = backend.ToString().ToLowerInvariant();

            var state = await _stateManager.GetStateAsync(backendStr, platform, cancellationToken);

            // No existing state, or the recorded build is gone from the cache root: acquire as on a first load
            // (adopts a cached build or downloads), then the next check updates it
            var installedExe = state is null ? null : ResolveInstalledServer(state);
            if (state is null || installedExe is null)
                return await AcquireAsync(backend, progress, localFilesOnly: false, cancellationToken);

            // Check for new version
            var latestVersion = await _downloader.GetLatestVersionAsync(cancellationToken);
            await _stateManager.RecordVersionCheckAsync(backendStr, platform, latestVersion, cancellationToken);

            if (latestVersion == null ||
                string.Equals(state.InstalledVersion, latestVersion, StringComparison.OrdinalIgnoreCase))
            {
                // Already up to date
                return LlamaServerUpdateResult.NoUpdate(installedExe, backend, state.InstalledVersion);
            }

            // Check if this version has failed before
            if (state.FailedVersions.Contains(latestVersion))
                return LlamaServerUpdateResult.NoUpdate(installedExe, backend, state.InstalledVersion);

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

                // Mark update ready
                await _stateManager.MarkUpdateReadyAsync(
                    backendStr, platform, latestVersion, BuildBackendOf(newServerPath), cancellationToken);

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
                return LlamaServerUpdateResult.NoUpdate(installedExe, backend, state.InstalledVersion);
            }
            catch (Exception ex) when (ex is not OperationCanceledException || !cancellationToken.IsCancellationRequested)
            {
                // Download/update failed, use existing
                return LlamaServerUpdateResult.Failed(installedExe, backend, ex.Message);
            }
        }
        finally
        {
            _updateLock.Release();
        }
    }

    /// <summary>
    /// Gets update information for the specified backend, or null when nothing is installed for it — no state, or
    /// the recorded build is not under the cache root in use.
    /// </summary>
    public async Task<LlamaServerUpdateInfo?> GetUpdateInfoAsync(
        LlamaServerBackend backend,
        CancellationToken cancellationToken = default)
    {
        var platform = GetCurrentPlatform();
        var backendStr = backend.ToString().ToLowerInvariant();

        var state = await _stateManager.GetStateAsync(backendStr, platform, cancellationToken);
        var serverPath = state is null ? null : ResolveInstalledServer(state);
        if (state == null || serverPath == null)
            return null;

        return new LlamaServerUpdateInfo
        {
            InstalledVersion = state.InstalledVersion,
            LatestVersion = state.LatestKnownVersion,
            UpdateReady = state.UpdateReady,
            PendingVersion = state.PendingVersion,
            LastChecked = state.LastVersionCheck,
            Backend = backend,
            ServerPath = serverPath
        };
    }

    /// <summary>
    /// Rolls back to the previous version if available — one below
    /// <see cref="LlamaServerVersionRequirements.MinimumSecureBuild"/> only when
    /// <see cref="LlamaServerUpdateOptions.AllowVulnerableBuild"/> is set.
    /// </summary>
    public async Task<LlamaServerUpdateResult> RollbackAsync(
        LlamaServerBackend backend,
        CancellationToken cancellationToken = default)
    {
        var platform = GetCurrentPlatform();
        var backendStr = backend.ToString().ToLowerInvariant();

        var state = await _stateManager.RollbackAsync(
            backendStr, platform,
            _options.AllowVulnerableBuild ? null : version => !LlamaServerVersionRequirements.IsBelowSecurityFloor(version),
            cancellationToken);
        if (state == null)
        {
            return LlamaServerUpdateResult.Failed("", backend, _options.AllowVulnerableBuild
                ? "No previous version available for rollback"
                : $"No previous version at or above b{LlamaServerVersionRequirements.MinimumSecureBuild} is available for rollback " +
                  "(set LlamaServerUpdateOptions.AllowVulnerableBuild to roll back to an older build)");
        }

        // The state manager rolls back only to a build whose directory exists; the executable is checked here
        var serverPath = ResolveInstalledServer(state);
        return serverPath is null
            ? LlamaServerUpdateResult.Failed("", backend, $"Rolled back to {state.InstalledVersion}, but its server executable is missing")
            : LlamaServerUpdateResult.NoUpdate(serverPath, backend, state.InstalledVersion);
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

                await _stateManager.MarkUpdateReadyAsync(
                    backendStr, platform, latestVersion, BuildBackendOf(serverPath), cancellationToken);
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

    // Every reader of a recorded build goes through these two: the build is resolved against the cache root in use,
    // and a build whose executable is not there counts as not installed.
    private string? ResolveInstalledServer(LlamaServerVersionState state)
        => ResolveServer(state.InstalledVersion, state.InstalledBuildBackend ?? state.Backend);

    private string? ResolvePendingServer(LlamaServerVersionState state)
        => state.PendingVersion is null ? null : ResolveServer(state.PendingVersion, state.PendingBuildBackend ?? state.Backend);

    private string? ResolveServer(string version, string buildBackend)
    {
        var serverExe = GetServerExecutablePath(_stateManager.GetBuildDirectory(version, buildBackend));
        return File.Exists(serverExe) ? serverExe : null;
    }

    // The backend directory a downloaded or cached server executable sits in (a GPU request can land on the CPU build).
    private static string BuildBackendOf(string serverPath) => Path.GetFileName(Path.GetDirectoryName(serverPath)!);

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
