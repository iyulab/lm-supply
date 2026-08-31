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
    private readonly RuntimeManagerOptions _options;
    private readonly RuntimeUpdateOptions _updateOptions;
    private readonly SemaphoreSlim _initLock = new(1, 1);

    private bool _initialized;
    private bool _disposed;
    private PlatformInfo? _platform;
    private GpuInfo? _gpu;
    private string? _currentVersion;
    private string? _activeProvider;
    private string? _primaryLibraryName;

    /// <summary>
    /// Gets the singleton instance of the runtime manager.
    /// </summary>
    public static RuntimeManager Instance { get; } = new();

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
    {
        _options = options;
        _updateOptions = updateOptions ?? RuntimeUpdateOptions.Default;
        _nugetDownloader = new OnnxNuGetDownloader(options.CacheDirectory);
    }

    /// <summary>
    /// Gets the detected platform information.
    /// </summary>
    public PlatformInfo Platform => _platform ?? throw new InvalidOperationException("Runtime manager not initialized");

    /// <summary>
    /// Gets the detected GPU information.
    /// </summary>
    public GpuInfo Gpu => _gpu ?? throw new InvalidOperationException("Runtime manager not initialized");

    /// <summary>
    /// Gets the recommended execution provider based on detected hardware.
    /// </summary>
    public ExecutionProvider RecommendedProvider => _gpu?.RecommendedProvider ?? ExecutionProvider.Cpu;

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
    /// See docket iyulab/lm-supply#151.
    /// </summary>
    public string? ActuallyLoadedRuntimePath =>
        _primaryLibraryName is null ? null : NativeLoader.Instance.GetLoadedPath(_primaryLibraryName);

    /// <summary>
    /// Initializes the runtime manager by detecting hardware.
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

            // Detect platform and GPU
            _platform = EnvironmentDetector.DetectPlatform();
            _gpu = EnvironmentDetector.DetectGpu();

            // Setup CUDA/cuDNN DLL search paths for Windows
            // This must be done before any ONNX session creation
            SetupCudaDllSearchPaths();

            _initialized = true;
        }
        finally
        {
            _initLock.Release();
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
    /// When provider is null (Auto mode), uses the fallback chain: CUDA → DirectML → CoreML → CPU.
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
            catch (Exception ex) when (providerToTry != "cpu")
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
    /// Downloads runtime for a specific provider from NuGet with auto-update support.
    /// </summary>
    private async Task<string> DownloadRuntimeForProviderAsync(
        string provider,
        string packageType,
        string? version,
        IProgress<DownloadProgress>? progress,
        CancellationToken cancellationToken)
    {
        // Get package configuration
        var config = RuntimePackageRegistry.GetPackageConfig(packageType, provider, _platform!.RuntimeIdentifier);
        if (config is null)
        {
            throw new InvalidOperationException($"No package configuration found for {packageType}/{provider}");
        }

        // Resolve initial version if not specified
        var currentVersion = version ?? await ResolveVersionAsync(config.PackageId, cancellationToken);

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

        // Track current state
        _currentVersion = currentVersion;
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
    /// Resolves the version to use from loaded assembly.
    /// </summary>
    private static async Task<string> ResolveVersionAsync(string packageId, CancellationToken ct)
    {
        // Try to get from loaded assembly
        var assemblyVersion = TryGetOnnxRuntimeVersion();
        if (!string.IsNullOrEmpty(assemblyVersion))
        {
            return assemblyVersion;
        }

        // Get latest from NuGet
        using var resolver = new NuGetPackageResolver();
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
    /// CUDA (cuda12/cuda11) → DirectML → CoreML → CPU
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

        if (_gpu is not null)
        {
            // CUDA first (if NVIDIA GPU with sufficient driver)
            if (_gpu.Vendor == GpuVendor.Nvidia)
            {
                // For GenAI, use generic "cuda" which maps to CUDA package
                if (packageType.Equals(RuntimePackageRegistry.PackageTypes.OnnxRuntimeGenAI, StringComparison.OrdinalIgnoreCase))
                {
                    if (_gpu.CudaDriverVersionMajor >= 11 && supportedProviders.Contains("cuda"))
                        chain.Add("cuda");
                }
                else
                {
                    // For standard ONNX Runtime, use specific CUDA versions
                    if (_gpu.CudaDriverVersionMajor >= 12 && supportedProviders.Contains("cuda12"))
                        chain.Add("cuda12");
                    else if (_gpu.CudaDriverVersionMajor >= 11 && supportedProviders.Contains("cuda11"))
                        chain.Add("cuda11");
                }
            }

            // DirectML (Windows with D3D12 support - works with AMD, Intel, NVIDIA)
            if (_gpu.DirectMLSupported && supportedProviders.Contains("directml"))
                chain.Add("directml");

            // CoreML (macOS/iOS)
            if (_gpu.CoreMLSupported && supportedProviders.Contains("coreml"))
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
            GPU: {_gpu}
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
    /// Gets or sets the proxy URL.
    /// </summary>
    public string? ProxyUrl { get; set; }

    /// <summary>
    /// Gets or sets the proxy username.
    /// </summary>
    public string? ProxyUsername { get; set; }

    /// <summary>
    /// Gets or sets the proxy password.
    /// </summary>
    public string? ProxyPassword { get; set; }

    /// <summary>
    /// Gets or sets the maximum retry attempts for downloads.
    /// </summary>
    public int MaxRetries { get; set; } = 3;

    /// <summary>
    /// Gets or sets whether <see cref="RuntimeManager.EnsureRuntimeAsync"/>/
    /// <see cref="RuntimeManager.CheckAndApplyUpdateAsync"/> throw
    /// <see cref="LMSupply.Exceptions.NativeLibraryConflictException"/> when the runtime
    /// they resolved would bind to a native library name that is already resident from a
    /// different binary (e.g. an earlier provider/version preloaded the same library name in this
    /// process). Default false preserves the historical behavior: the request silently keeps
    /// using the resident binary and <see cref="RuntimeManager.ActuallyLoadedRuntimePath"/> is the
    /// only way to notice. This never unloads or replaces the resident binary either way -- it
    /// only decides whether the conflicting request fails instead of silently no-op'ing. See
    /// docket iyulab/lm-supply#151.
    /// </summary>
    public bool FailOnRuntimeConflict { get; set; }
}
