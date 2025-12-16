using LMSupply.Download;

namespace LMSupply.Runtime;

/// <summary>
/// Orchestrates runtime binary management including detection, download, caching, and loading.
/// This is the main entry point for the lazy binary distribution system.
/// </summary>
public sealed class RuntimeManager : IAsyncDisposable
{
    private readonly ManifestProvider _manifestProvider;
    private readonly BinaryDownloader _downloader;
    private readonly RuntimeCache _cache;
    private readonly RuntimeManagerOptions _options;
    private readonly SemaphoreSlim _initLock = new(1, 1);

    private bool _initialized;
    private PlatformInfo? _platform;
    private GpuInfo? _gpu;

    /// <summary>
    /// Gets the singleton instance of the runtime manager.
    /// </summary>
    public static RuntimeManager Instance { get; } = new();

    /// <summary>
    /// Creates a new runtime manager with default options.
    /// </summary>
    public RuntimeManager() : this(new RuntimeManagerOptions())
    {
    }

    /// <summary>
    /// Creates a new runtime manager with custom options.
    /// </summary>
    public RuntimeManager(RuntimeManagerOptions options)
    {
        _options = options;
        _manifestProvider = new ManifestProvider();
        _downloader = new BinaryDownloader(new BinaryDownloaderOptions
        {
            ProxyUrl = options.ProxyUrl,
            ProxyUsername = options.ProxyUsername,
            ProxyPassword = options.ProxyPassword,
            MaxRetries = options.MaxRetries
        });
        _cache = new RuntimeCache(new RuntimeCacheOptions
        {
            CacheDirectory = options.CacheDirectory,
            MaxCacheSize = options.MaxCacheSize
        });
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
    /// Initializes the runtime manager by detecting hardware.
    /// </summary>
    public async Task InitializeAsync(CancellationToken cancellationToken = default)
    {
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

            _initialized = true;
        }
        finally
        {
            _initLock.Release();
        }
    }

    /// <summary>
    /// Ensures a runtime binary is available, downloading if necessary.
    /// </summary>
    /// <param name="package">The package name (e.g., "onnxruntime").</param>
    /// <param name="version">Optional version. If null, uses latest available.</param>
    /// <param name="provider">Optional provider. If null, uses best available for hardware.</param>
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
        await InitializeAsync(cancellationToken);

        var rid = _platform!.RuntimeIdentifier;
        var actualProvider = provider ?? GetDefaultProvider();

        // Check cache first
        var manifest = await _manifestProvider.GetManifestAsync(cancellationToken: cancellationToken);

        // Resolve version if not specified
        var actualVersion = version;
        if (string.IsNullOrEmpty(actualVersion))
        {
            var pkg = manifest.GetPackage(package);
            actualVersion = pkg?.Versions.Keys
                .OrderByDescending(v => Version.TryParse(v, out var ver) ? ver : new Version(0, 0))
                .FirstOrDefault();

            if (actualVersion is null)
                throw new InvalidOperationException($"No versions available for package: {package}");
        }

        // Check cache
        var cachedPath = await _cache.GetCachedPathAsync(package, actualVersion, rid, actualProvider, cancellationToken);
        if (cachedPath is not null)
        {
            // Register with native loader
            NativeLoader.Instance.RegisterDirectory(Path.GetDirectoryName(cachedPath)!);
            return Path.GetDirectoryName(cachedPath)!;
        }

        // Get binary entry from manifest
        var entry = await _manifestProvider.GetBinaryAsync(package, actualVersion, rid, actualProvider, cancellationToken);
        if (entry is null)
        {
            // Try fallback to CPU if GPU provider not available
            if (actualProvider != "cpu")
            {
                entry = await _manifestProvider.GetBinaryAsync(package, actualVersion, rid, "cpu", cancellationToken);
                if (entry is not null)
                    actualProvider = "cpu";
            }
        }

        if (entry is null)
            throw new InvalidOperationException(
                $"No binary available for {package} {actualVersion} on {rid} with provider {actualProvider}");

        // Download and cache
        var targetDirectory = _cache.GetCacheDirectory(package, actualVersion, rid, actualProvider);
        var binaryPath = await _downloader.DownloadAsync(entry, targetDirectory, progress, cancellationToken);

        // Register in cache
        await _cache.RegisterAsync(entry, package, actualVersion, binaryPath, cancellationToken);

        // Register with native loader
        NativeLoader.Instance.RegisterDirectory(targetDirectory);

        return targetDirectory;
    }

    /// <summary>
    /// Gets the best provider string for the current hardware.
    /// </summary>
    public string GetDefaultProvider()
    {
        if (_gpu is null)
            return "cpu";

        return _gpu.Vendor switch
        {
            GpuVendor.Nvidia when _gpu.CudaDriverVersionMajor >= 12 => "cuda12",
            GpuVendor.Nvidia when _gpu.CudaDriverVersionMajor >= 11 => "cuda11",
            _ when _gpu.DirectMLSupported => "directml",
            _ when _gpu.CoreMLSupported => "coreml",
            _ => "cpu"
        };
    }

    /// <summary>
    /// Gets the runtime cache.
    /// </summary>
    public RuntimeCache Cache => _cache;

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
            Cache Directory: {_cache.GetCacheDirectory("", "", "", "")}
            Cache Size: {_cache.GetTotalCacheSize() / (1024.0 * 1024.0):F2} MB
            """;
    }

    public async ValueTask DisposeAsync()
    {
        _manifestProvider.Dispose();
        _downloader.Dispose();
        _cache.Dispose();
        _initLock.Dispose();
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
    /// Gets or sets the maximum cache size in bytes. Default is 10 GB.
    /// </summary>
    public long MaxCacheSize { get; set; } = 10L * 1024 * 1024 * 1024;

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
}
