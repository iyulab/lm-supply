using System.Diagnostics;
using LMSupply.Download;
using LMSupply.Exceptions;

namespace LMSupply.Runtime;

/// <summary>
/// Orchestrates runtime binary management including detection, download, caching, and loading.
/// Downloads native binaries on-demand from NuGet.org - no pre-built manifest required.
/// </summary>
public sealed class RuntimeManager : IAsyncDisposable
{
    private const string PackageType = "onnxruntime";

    private readonly OnnxNuGetDownloader _nugetDownloader;
    private readonly HttpMessageHandler? _handler;
    private readonly RuntimeManagerOptions _options;
    private readonly RuntimeUpdateOptions _updateOptions;
    private readonly SemaphoreSlim _initLock = new(1, 1);

    private bool _initialized;
    private bool _disposed;
    private PlatformInfo? _platform;
    private volatile GpuInfo? _gpu;
    private readonly object _gpuLock = new();
    private string? _currentVersion;
    private string? _activeProvider;
    private string? _primaryLibraryName;

    private static readonly object s_configureLock = new();
    private static RuntimeManagerOptions? s_configuredOptions;
    private static RuntimeManager? s_instance;

    /// <summary>
    /// Gets the process-wide runtime manager every model load goes through. It is created on first read with the
    /// options given to <see cref="Configure"/>, or with defaults when nothing was configured.
    /// </summary>
    public static RuntimeManager Instance
    {
        get
        {
            if (Volatile.Read(ref s_instance) is { } existing)
                return existing;

            lock (s_configureLock)
            {
                if (s_instance is null)
                    Volatile.Write(ref s_instance, new RuntimeManager(s_configuredOptions ?? new RuntimeManagerOptions()));
                return s_instance!;
            }
        }
    }

    /// <summary>
    /// Sets the options of the process-wide <see cref="Instance"/>: where the native ONNX Runtime comes from
    /// (<see cref="RuntimeManagerOptions.RuntimeDirectory"/>), whether it may be fetched from nuget.org
    /// (<see cref="RuntimeManagerOptions.DisableAutoDownload"/>) and which version it is
    /// (<see cref="RuntimeManagerOptions.PinnedVersion"/>). A process loads one native runtime, so this is
    /// process-wide rather than per model. Call it once at startup, before the first model load.
    /// </summary>
    /// <exception cref="InvalidOperationException">
    /// <see cref="Instance"/> already exists (a model has loaded, or code read it), so the options it runs with can
    /// no longer change.
    /// </exception>
    public static void Configure(RuntimeManagerOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);

        lock (s_configureLock)
        {
            if (s_instance is not null)
            {
                throw new InvalidOperationException(
                    "RuntimeManager.Configure must be called before the first model load: the process-wide runtime manager " +
                    "already exists and its options cannot change.");
            }

            s_configuredOptions = options;
        }
    }

    /// <summary>
    /// Creates a new runtime manager with default options.
    /// </summary>
    public RuntimeManager() : this(new RuntimeManagerOptions(), RuntimeUpdateOptions.Default)
    {
    }

    /// <summary>
    /// Creates a new runtime manager with custom options.
    /// </summary>
    public RuntimeManager(RuntimeManagerOptions options, RuntimeUpdateOptions? updateOptions = null)
        : this(options, updateOptions, handler: null)
    {
    }

    /// <summary>Test seam: the same manager over a caller-supplied transport for every feed request it makes.</summary>
    internal RuntimeManager(RuntimeManagerOptions options, RuntimeUpdateOptions? updateOptions, HttpMessageHandler? handler)
    {
        ArgumentNullException.ThrowIfNull(options);
        _options = options;
        _updateOptions = updateOptions ?? RuntimeUpdateOptions.Default;
        _handler = handler;
        _nugetDownloader = new OnnxNuGetDownloader(options.CacheDirectory, handler);
    }

    /// <summary>The options this manager runs with.</summary>
    public RuntimeManagerOptions Options => _options;

    /// <summary>
    /// Gets the detected platform information.
    /// </summary>
    public PlatformInfo Platform => _platform ?? throw new InvalidOperationException("Runtime manager not initialized");

    /// <summary>
    /// Gets the detected GPU information. The GPU is probed on first read, not at initialization: the probe loads the
    /// vendor driver libraries (NVML, and the CUDA driver with it), which a process that only runs CPU sessions never needs.
    /// </summary>
    public GpuInfo Gpu => _initialized ? EnsureGpu() : throw new InvalidOperationException("Runtime manager not initialized");

    /// <summary>
    /// Gets the recommended execution provider based on detected hardware. Reading it probes the GPU.
    /// </summary>
    public ExecutionProvider RecommendedProvider => _initialized ? EnsureGpu().RecommendedProvider : ExecutionProvider.Cpu;

    /// <summary>Whether the GPU has been probed in this process by this manager.</summary>
    internal bool IsGpuDetected => _gpu is not null;

    /// <summary>
    /// Gets the current runtime version.
    /// </summary>
    public string? CurrentVersion => _currentVersion;

    /// <summary>
    /// Gets the active provider string.
    /// </summary>
    public string? ActiveProvider => _activeProvider;

    /// <summary>
    /// Gets the filesystem path of the native runtime binary actually resident in this
    /// process, or null if none has loaded yet. This can disagree with what
    /// <see cref="ActiveProvider"/>/<see cref="CurrentVersion"/> last requested: those two
    /// track what this manager most recently resolved and asked <see cref="NativeLoader"/>
    /// to load, but a native library name only ever binds to the first binary that
    /// successfully loads under it in this process -- if something else already preloaded
    /// the same library name (e.g. an earlier provider/version), a later
    /// <see cref="EnsureRuntimeAsync"/> call can update <see cref="ActiveProvider"/> while the
    /// actual resident binary silently stays the old one. Compare this path's directory
    /// against the path <see cref="EnsureRuntimeAsync"/> returned to detect that mismatch.
    /// </summary>
    public string? ActuallyLoadedRuntimePath =>
        _primaryLibraryName is null ? null : NativeLoader.Instance.GetLoadedPath(_primaryLibraryName);

    /// <summary>
    /// Initializes the runtime manager: detects the platform. The GPU is probed later, by the first operation that
    /// needs it (an Auto provider choice, a GPU provider's runtime), so a CPU-only load never loads GPU driver libraries.
    /// </summary>
    public async Task InitializeAsync(CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();

        if (_initialized)
            return;

        await _initLock.WaitAsync(cancellationToken);
        try
        {
            if (_initialized)
                return;

            _platform = EnvironmentDetector.DetectPlatform();
            _initialized = true;
        }
        finally
        {
            _initLock.Release();
        }
    }

    /// <summary>
    /// Probes the GPU once and, with it, sets up the CUDA/cuDNN DLL search paths a GPU session needs.
    /// </summary>
    private GpuInfo EnsureGpu()
    {
        if (_gpu is not null)
            return _gpu;

        lock (_gpuLock)
        {
            if (_gpu is not null)
                return _gpu;

            var gpu = EnvironmentDetector.DetectGpu();
            _gpu = gpu;

            // Before any GPU session is created (CUDA provider native dependencies).
            SetupCudaDllSearchPaths();
            return gpu;
        }
    }

    /// <summary>
    /// Sets up CUDA and cuDNN DLL search paths for Windows.
    /// This enables ONNX Runtime's CUDA provider to find native dependencies.
    /// Uses both AddDllDirectory (for LoadLibraryEx) and PATH modification (for LoadLibrary).
    /// </summary>
    private void SetupCudaDllSearchPaths()
    {
        if (!OperatingSystem.IsWindows())
            return;

        // Initialize CUDA environment detection
        var cudaEnv = CudaEnvironment.Instance;
        cudaEnv.Initialize();

        // Determine target CUDA version from detected GPU
        var cudaMajorVersion = _gpu?.CudaDriverVersionMajor ?? 12;

        // Get all DLL search paths from CudaEnvironment
        var pathsToAdd = cudaEnv.GetDllSearchPaths(cudaMajorVersion).ToList();

        // Register paths with NativeLoader
        foreach (var path in pathsToAdd)
        {
            NativeLoader.Instance.AddToWindowsDllSearchPath(path);
            Trace.TraceInformation($"[RuntimeManager] Added to DLL search path: {path}");
        }

        // Also modify PATH environment variable for current process
        // This ensures ONNX Runtime can find DLLs even when using standard LoadLibrary
        if (pathsToAdd.Count > 0)
        {
            var currentPath = Environment.GetEnvironmentVariable("PATH") ?? "";
            var newPaths = pathsToAdd.Where(p => !currentPath.Contains(p, StringComparison.OrdinalIgnoreCase));
            if (newPaths.Any())
            {
                var pathToAdd = string.Join(Path.PathSeparator.ToString(), newPaths);
                Environment.SetEnvironmentVariable("PATH", pathToAdd + Path.PathSeparator + currentPath);
                Trace.TraceInformation($"[RuntimeManager] Added to PATH: {pathToAdd}");
            }
        }

        // Log diagnostics in debug mode
        Trace.TraceInformation(cudaEnv.GetDiagnostics());
    }

    /// <summary>
    /// Ensures a runtime binary is available, downloading from NuGet if necessary.
    /// When provider is null (Auto mode), uses the fallback chain: CUDA → CoreML → CPU.
    /// </summary>
    /// <param name="package">The package name (e.g., "onnxruntime").</param>
    /// <param name="version">Optional version. If null, auto-detects from assembly.</param>
    /// <param name="provider">Optional provider. If null, uses fallback chain for best available.</param>
    /// <param name="progress">Optional progress reporter.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Path to the binary directory.</returns>
    public async Task<string> EnsureRuntimeAsync(
        string package,
        string? version = null,
        string? provider = null,
        IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();
        await InitializeAsync(cancellationToken);

        // Normalize package type
        var packageType = NormalizePackageType(package);

        // If provider is explicitly specified, download for that provider
        if (!string.IsNullOrEmpty(provider))
        {
            // A GPU provider's session needs the GPU probe's DLL search paths; the CPU one needs no probe at all.
            if (!provider.Equals("cpu", StringComparison.OrdinalIgnoreCase))
                EnsureGpu();
            return await DownloadRuntimeForProviderAsync(provider, packageType, version, progress, cancellationToken);
        }

        // Auto mode: try providers in fallback chain order
        var chain = GetProviderFallbackChain(packageType);
        Exception? lastException = null;

        foreach (var providerToTry in chain)
        {
            try
            {
                return await DownloadRuntimeForProviderAsync(providerToTry, packageType, version, progress, cancellationToken);
            }
            catch (Exception ex) when (StopsProviderFallbackChain(ex))
            {
                throw; // Cancellation, or a conflict every remaining provider would hit identically
            }
            catch (Exception ex) when ((ex is not OperationCanceledException || !cancellationToken.IsCancellationRequested) && (providerToTry != "cpu"))
            {
                // Log and continue to next provider in chain
                Trace.TraceInformation(
                    $"[RuntimeManager] Provider '{providerToTry}' failed: {ex.Message}. Trying next provider...");
                lastException = ex;
            }
        }

        // Should not reach here since CPU is always in chain
        throw lastException ?? new InvalidOperationException($"No provider available for {packageType}");
    }

    /// <summary>
    /// Determines whether an exception from a single provider attempt in
    /// <see cref="EnsureRuntimeAsync"/>'s Auto-mode fallback chain should propagate immediately,
    /// stopping the chain, rather than being logged so the next provider can be tried.
    /// <see cref="OperationCanceledException"/> always stops it (cancellation is not a
    /// per-provider failure). A <see cref="NativeLibraryConflictException"/> also always stops
    /// it: every provider in the chain registers its native binary under the same normalized
    /// library name (typically "onnxruntime"), so once one request conflicts with the resident
    /// binary, every remaining provider would conflict identically -- catching it here would
    /// just re-trigger the same exception on the next iteration, and if a later provider's
    /// attempt happened not to conflict (or CPU's fallback-of-last-resort masked it), the
    /// original conflict a caller opted into <see cref="RuntimeManagerOptions.
    /// FailOnRuntimeConflict"/> to see would be silently lost instead of reported.
    /// Internal (not private) so it can be unit-tested directly -- exercising this through
    /// <see cref="EnsureRuntimeAsync"/> end-to-end requires network access this test suite does
    /// not use.
    /// </summary>
    internal static bool StopsProviderFallbackChain(Exception ex) =>
        ex is OperationCanceledException or NativeLibraryConflictException;

    /// <summary>
    /// Provisions the runtime for a specific provider and registers it with <see cref="NativeLoader"/>.
    /// </summary>
    private async Task<string> DownloadRuntimeForProviderAsync(
        string provider,
        string packageType,
        string? version,
        IProgress<DownloadProgress>? progress,
        CancellationToken cancellationToken)
    {
        var (binaryPath, resolvedVersion, config) =
            await ResolveRuntimeForProviderAsync(provider, packageType, version, progress, cancellationToken);

        // Track current state
        _currentVersion = resolvedVersion;
        _activeProvider = provider;

        // Register with NativeLoader for DLL resolution
        var primaryLibrary = config.NativeLibraryName ?? "onnxruntime";
        _primaryLibraryName = primaryLibrary;
        NativeLoader.Instance.RegisterDirectory(
            binaryPath, preload: true, primaryLibrary: primaryLibrary,
            throwOnConflict: _options.FailOnRuntimeConflict);

        return binaryPath;
    }

    /// <summary>
    /// Finds the directory the runtime for <paramref name="provider"/> loads from, fetching it only when the options
    /// allow: a <see cref="RuntimeManagerOptions.RuntimeDirectory"/> is used as-is, a local-only manager
    /// (<see cref="RuntimeManagerOptions.DisableAutoDownload"/>) reads the cache or throws, and otherwise the version
    /// is resolved and downloaded on a cache miss. Registers nothing -- internal so tests can assert where a runtime
    /// comes from without loading a native library into the test process.
    /// </summary>
    internal async Task<(string Path, string Version, RuntimePackageRegistry.PackageConfig Config)> ResolveRuntimeForProviderAsync(
        string provider,
        string packageType,
        string? version,
        IProgress<DownloadProgress>? progress,
        CancellationToken cancellationToken)
    {
        // A provider this build cannot serve fails here, loud, before any lookup -- the registry falls back
        // to the CPU package for an unknown name, which would otherwise turn "directml" into a silent CPU run.
        ExecutionProviderSupport.ThrowIfUnsupported(provider);
        await InitializeAsync(cancellationToken);

        // Get package configuration
        var config = RuntimePackageRegistry.GetPackageConfig(packageType, provider, _platform!.RuntimeIdentifier);
        if (config is null)
        {
            throw new InvalidOperationException($"No package configuration found for {packageType}/{provider}");
        }

        var wantedVersion = version ?? _options.PinnedVersion;

        if (!string.IsNullOrEmpty(_options.RuntimeDirectory))
        {
            var bundled = UseRuntimeDirectory(_options.RuntimeDirectory, provider, packageType, config);
            return (bundled, wantedVersion ?? TryGetOnnxRuntimeVersion() ?? "bundled", config);
        }

        if (_options.DisableAutoDownload)
        {
            // Never the network: no version lookup, no download, no update check. An exact version (pinned, or the
            // one the loaded managed assembly expects) is preferred; without a pin, the newest cached copy stands in,
            // the same policy the downloader applies when the feed is unreachable.
            var expected = wantedVersion ?? TryGetOnnxRuntimeVersion();
            var cached = _nugetDownloader.FindCached(
                provider, _platform!, expected, packageType, exactOnly: _options.PinnedVersion is not null);
            if (cached is null)
            {
                throw new ModelLoadException(
                    $"The {config.PackageId} runtime for provider '{provider}' ({_platform!.RuntimeIdentifier}" +
                    (expected is null ? "" : $", version {expected}") + ") is not in the runtime cache " +
                    $"'{CacheDirectory}', and RuntimeManagerOptions.DisableAutoDownload forbids downloading it. " +
                    "Ship the native runtime with the application and point RuntimeManagerOptions.RuntimeDirectory at it, " +
                    "or provision the cache once with downloads allowed.");
            }

            return (cached.Value.Path, cached.Value.Version, config);
        }

        // Resolve initial version if not specified
        var currentVersion = wantedVersion ?? await ResolveVersionAsync(config.PackageId, cancellationToken);

        if (_options.PinnedVersion is not null)
        {
            // Pinned: exactly this version -- no background update check and no applying a previously downloaded
            // newer one, both of which the update service would do.
            var pinnedPath = await _nugetDownloader.DownloadAsync(
                provider, _platform!, currentVersion, progress, packageType, cancellationToken);
            return (pinnedPath, currentVersion, config);
        }

        // Get update service
        var updateService = RuntimeUpdateService.GetInstance(packageType, _updateOptions);

        // Download with update service
        var binaryPath = await updateService.GetRuntimePathAsync(
            config.PackageId,
            provider,
            _platform!,
            currentVersion,
            (ver, prog, ct) => _nugetDownloader.DownloadAsync(provider, _platform!, ver, prog, packageType, ct),
            progress,
            cancellationToken);

        // Validate runtime path before registration
        if (string.IsNullOrEmpty(binaryPath))
        {
            throw new ModelLoadException(
                $"Runtime binary path resolved to null for provider '{provider}'. " +
                "This may indicate a corrupted runtime state file. " +
                "Try deleting the runtime cache directory and retrying.");
        }

        return (binaryPath, currentVersion, config);
    }

    /// <summary>
    /// Checks that an application-supplied runtime directory holds every native library the provider needs and returns
    /// it. A missing library throws: in the Auto chain that moves on to the next provider (a CPU-only bundle serves CPU),
    /// and an explicit GPU request fails loud instead of running on a runtime without its provider.
    /// </summary>
    private string UseRuntimeDirectory(
        string directory, string provider, string packageType, RuntimePackageRegistry.PackageConfig config)
    {
        var fullPath = Path.GetFullPath(directory);
        var missing = new[] { config.NativeLibraryName }
            .Concat(config.AdditionalLibraries)
            .Select(lib => RuntimePackageRegistry.GetNativeLibraryFileName(lib, _platform!))
            .Where(file => !Directory.Exists(fullPath) || !Directory.EnumerateFiles(fullPath, file + "*").Any())
            .ToList();

        if (missing.Count > 0)
        {
            throw new ModelLoadException(
                $"RuntimeManagerOptions.RuntimeDirectory '{fullPath}' cannot serve {packageType} for provider '{provider}': " +
                $"missing {string.Join(", ", missing)}. Copy the native files of the {config.PackageId} package " +
                $"(runtimes/{_platform!.RuntimeIdentifier}/native) into it.");
        }

        return fullPath;
    }

    /// <summary>
    /// Resolves the version to use from loaded assembly.
    /// </summary>
    private async Task<string> ResolveVersionAsync(string packageId, CancellationToken ct)
    {
        // Try to get from loaded assembly
        var assemblyVersion = TryGetOnnxRuntimeVersion();
        if (!string.IsNullOrEmpty(assemblyVersion))
        {
            return assemblyVersion;
        }

        // Get latest from NuGet
        using var resolver = new NuGetPackageResolver(
            _handler is null ? null : new HttpClient(_handler, disposeHandler: false));
        var latest = await resolver.GetLatestVersionAsync(packageId, includePrerelease: false, ct);
        return latest ?? throw new InvalidOperationException($"Could not determine version for {packageId}");
    }

    private static string? TryGetOnnxRuntimeVersion()
    {
        try
        {
            var assembly = AppDomain.CurrentDomain.GetAssemblies()
                .FirstOrDefault(a => a.GetName().Name?.Equals("Microsoft.ML.OnnxRuntime", StringComparison.OrdinalIgnoreCase) == true);

            if (assembly is null)
                return null;

            var infoAttr = assembly.GetCustomAttributes(typeof(System.Reflection.AssemblyInformationalVersionAttribute), false)
                .OfType<System.Reflection.AssemblyInformationalVersionAttribute>()
                .FirstOrDefault();

            if (infoAttr != null)
            {
                var ver = infoAttr.InformationalVersion;
                var plusIdx = ver.IndexOf('+');
                if (plusIdx > 0)
                    ver = ver[..plusIdx];
                if (!string.IsNullOrEmpty(ver) && ver != "0.0.0")
                    return ver;
            }

            var version = assembly.GetName().Version;
            if (version != null && version.Major > 0)
            {
                return $"{version.Major}.{version.Minor}.{version.Build}";
            }
        }
        catch (Exception ex)
        {
            Trace.TraceInformation($"[RuntimeManager] Assembly version lookup failed: {ex.Message}");
        }

        return null;
    }

    /// <summary>
    /// Normalizes the package type string to a standard format.
    /// </summary>
    private static string NormalizePackageType(string package)
    {
        if (string.IsNullOrEmpty(package))
            return RuntimePackageRegistry.PackageTypes.OnnxRuntime;

        // Handle common aliases
        return package.ToLowerInvariant() switch
        {
            "onnxruntime" or "onnx" or "runtime" => RuntimePackageRegistry.PackageTypes.OnnxRuntime,
            "onnxruntime-genai" or "genai" or "gen-ai" or "generator" => RuntimePackageRegistry.PackageTypes.OnnxRuntimeGenAI,
            _ => package.ToLowerInvariant()
        };
    }

    /// <summary>
    /// Gets the best provider string for the current hardware.
    /// Returns the first provider in the fallback chain.
    /// </summary>
    public string GetDefaultProvider()
    {
        return GetProviderFallbackChain()[0];
    }

    /// <summary>
    /// Gets a prioritized list of providers to try based on detected hardware.
    /// The fallback chain ensures zero-configuration GPU acceleration:
    /// CUDA (cuda12/cuda11) → CoreML → CPU
    /// </summary>
    public IReadOnlyList<string> GetProviderFallbackChain()
    {
        return GetProviderFallbackChain(RuntimePackageRegistry.PackageTypes.OnnxRuntime);
    }

    /// <summary>
    /// Gets a prioritized list of providers to try based on detected hardware and package type.
    /// Different package types may support different provider sets.
    /// </summary>
    public IReadOnlyList<string> GetProviderFallbackChain(string packageType)
    {
        var chain = new List<string>();
        var supportedProviders = RuntimePackageRegistry.GetSupportedProviders(packageType).ToHashSet(StringComparer.OrdinalIgnoreCase);

        // The fallback chain is the Auto choice — it is what the GPU probe is for.
        var gpu = _initialized ? EnsureGpu() : _gpu;
        if (gpu is not null)
        {
            // CUDA first (if NVIDIA GPU with sufficient driver)
            if (gpu.Vendor == GpuVendor.Nvidia)
            {
                // For GenAI, use generic "cuda" which maps to CUDA package
                if (packageType.Equals(RuntimePackageRegistry.PackageTypes.OnnxRuntimeGenAI, StringComparison.OrdinalIgnoreCase))
                {
                    if (gpu.CudaDriverVersionMajor >= 11 && supportedProviders.Contains("cuda"))
                        chain.Add("cuda");
                }
                else
                {
                    // For standard ONNX Runtime, use specific CUDA versions
                    if (gpu.CudaDriverVersionMajor >= 12 && supportedProviders.Contains("cuda12"))
                        chain.Add("cuda12");
                    else if (gpu.CudaDriverVersionMajor >= 11 && supportedProviders.Contains("cuda11"))
                        chain.Add("cuda11");
                }
            }

            // CoreML (macOS/iOS)
            if (gpu.CoreMLSupported && supportedProviders.Contains("coreml"))
                chain.Add("coreml");
        }

        // CPU always as final fallback
        chain.Add("cpu");

        return chain;
    }

    /// <summary>
    /// Gets the cache directory path.
    /// </summary>
    public string CacheDirectory => _options.CacheDirectory ?? LMSupplyCachePaths.GetRuntimesDirectory();

    /// <summary>
    /// Gets environment information summary.
    /// </summary>
    public string GetEnvironmentSummary()
    {
        if (!_initialized)
            return "Runtime manager not initialized";

        return $"""
            Platform: {_platform}
            GPU: {_gpu?.ToString() ?? "(not probed)"}
            Recommended Provider: {RecommendedProvider}
            Default Provider String: {GetDefaultProvider()}
            Active Provider: {_activeProvider ?? "none"}
            Current Version: {_currentVersion ?? "unknown"}
            Actually Loaded Runtime Path: {ActuallyLoadedRuntimePath ?? "none"}
            Cache Directory: {CacheDirectory}
            """;
    }

    /// <summary>
    /// Checks for runtime updates and applies them synchronously.
    /// Called during WarmupAsync to ensure latest runtime before inference.
    /// </summary>
    public async Task<RuntimeUpdateResult> CheckAndApplyUpdateAsync(
        string packageType,
        IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();

        if (!_initialized || _platform is null || _activeProvider is null)
        {
            return RuntimeUpdateResult.Failed("Runtime not initialized. Call EnsureRuntimeAsync first.");
        }

        if (!_options.AllowsRuntimeUpdates)
        {
            // A bundled directory, a local-only manager and a pinned version all mean "exactly the runtime in hand".
            return RuntimeUpdateResult.NoUpdateNeeded(_currentVersion ?? "unknown", ActuallyLoadedRuntimePath ?? string.Empty);
        }

        var normalizedPackageType = NormalizePackageType(packageType);
        var config = RuntimePackageRegistry.GetPackageConfig(normalizedPackageType, _activeProvider, _platform.RuntimeIdentifier);
        if (config is null)
        {
            return RuntimeUpdateResult.Failed($"No package configuration found for {normalizedPackageType}/{_activeProvider}");
        }

        var updateService = RuntimeUpdateService.GetInstance(normalizedPackageType, _updateOptions);
        var currentVersion = _currentVersion ?? await ResolveVersionAsync(config.PackageId, cancellationToken);

        var result = await updateService.CheckAndApplyUpdateAsync(
            config.PackageId,
            _activeProvider,
            _platform,
            currentVersion,
            (ver, prog, ct) => _nugetDownloader.DownloadAsync(_activeProvider, _platform, ver, prog, normalizedPackageType, ct),
            progress,
            cancellationToken);

        if (result.Updated && !string.IsNullOrEmpty(result.RuntimePath))
        {
            // Re-register with NativeLoader
            var primaryLibrary = config.NativeLibraryName ?? "onnxruntime";
            _primaryLibraryName = primaryLibrary;
            NativeLoader.Instance.RegisterDirectory(
                result.RuntimePath, preload: true, primaryLibrary: primaryLibrary,
                throwOnConflict: _options.FailOnRuntimeConflict);
            _currentVersion = result.NewVersion;
        }

        return result;
    }

    /// <summary>
    /// Gets runtime update information for diagnostics.
    /// </summary>
    public async Task<RuntimeUpdateInfo> GetRuntimeUpdateInfoAsync(
        string packageType,
        CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();

        if (!_initialized || _platform is null || _activeProvider is null)
        {
            return new RuntimeUpdateInfo
            {
                InstalledVersion = "unknown",
                Provider = "not initialized"
            };
        }

        var normalizedPackageType = NormalizePackageType(packageType);
        var updateService = RuntimeUpdateService.GetInstance(normalizedPackageType, _updateOptions);

        return await updateService.GetUpdateInfoAsync(
            _activeProvider,
            _platform,
            cancellationToken);
    }

    public async ValueTask DisposeAsync()
    {
        if (_disposed)
            return;

        _disposed = true;
        _initialized = false;
        _nugetDownloader.Dispose();
        _initLock.Dispose();
    }

    private void ThrowIfDisposed()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
    }
}

/// <summary>
/// Options for the runtime manager.
/// </summary>
public sealed class RuntimeManagerOptions
{
    /// <summary>
    /// Gets or sets the cache directory.
    /// </summary>
    public string? CacheDirectory { get; set; }

    /// <summary>
    /// Gets or sets whether <see cref="RuntimeManager.EnsureRuntimeAsync"/>/
    /// <see cref="RuntimeManager.CheckAndApplyUpdateAsync"/> throw
    /// <see cref="LMSupply.Exceptions.NativeLibraryConflictException"/> when the runtime
    /// they resolved would bind to a native library name that is already resident from a
    /// different binary (e.g. an earlier provider/version preloaded the same library name in this
    /// process). Default false preserves the historical behavior: the request silently keeps
    /// using the resident binary and <see cref="RuntimeManager.ActuallyLoadedRuntimePath"/> is the
    /// only way to notice. This never unloads or replaces the resident binary either way -- it
    /// only decides whether the conflicting request fails instead of silently no-op'ing.
    /// </summary>
    public bool FailOnRuntimeConflict { get; set; }

    /// <summary>
    /// A directory the application ships the native ONNX Runtime in: the files of the package's
    /// <c>runtimes/&lt;rid&gt;/native</c> folder (for CPU, <c>onnxruntime.dll</c> / <c>libonnxruntime.so</c> /
    /// <c>libonnxruntime.dylib</c> and their companions). When set, the runtime is loaded from here and nothing is
    /// looked up, downloaded or updated. It must hold every library the requested provider needs: a CPU-only bundle
    /// serves CPU, the Auto chain moves past GPU providers it cannot serve, and an explicit GPU request fails.
    /// Default: null (the runtime is provisioned from nuget.org into the cache).
    /// </summary>
    public string? RuntimeDirectory { get; set; }

    /// <summary>
    /// When true, the runtime is never fetched: it comes from <see cref="RuntimeDirectory"/> or the runtime cache, and
    /// a cache miss throws instead of downloading. No version lookup and no update check reach nuget.org either. This is
    /// the runtime counterpart of the model options' <c>DisableAutoDownload</c>, which cover model files only.
    /// Default: false.
    /// </summary>
    public bool DisableAutoDownload { get; set; }

    /// <summary>
    /// An exact native runtime version (e.g. "1.30.0"). When set, the version is never resolved from nuget.org, not
    /// even when the loaded managed assembly's version cannot be read (trimming, Native AOT), and the runtime is
    /// never replaced by a newer one. With <see cref="DisableAutoDownload"/>, only this version is taken from the cache.
    /// Default: null (the version the loaded Microsoft.ML.OnnxRuntime assembly expects).
    /// </summary>
    public string? PinnedVersion { get; set; }

    /// <summary>Whether the runtime may be replaced by a newer version after it is provisioned.</summary>
    internal bool AllowsRuntimeUpdates =>
        string.IsNullOrEmpty(RuntimeDirectory) && !DisableAutoDownload && PinnedVersion is null;
}
