using System.Net;
using System.Net.Http.Headers;
using System.Reflection;
using System.Text.Json;

namespace LMSupply.Runtime;

/// <summary>
/// Provides access to the runtime manifest with caching and conditional request support.
/// Uses ETag-based conditional requests to minimize GitHub API rate limit consumption.
/// 304 Not Modified responses don't count against the 60 req/hour unauthenticated limit.
/// </summary>
public sealed class ManifestProvider : IDisposable
{
    private readonly HttpClient _httpClient;
    private readonly string _cacheDirectory;
    private readonly TimeSpan _cacheExpiration;
    private readonly SemaphoreSlim _lock = new(1, 1);

    private RuntimeManifest? _cachedManifest;
    private string? _cachedETag;
    private DateTimeOffset _lastFetched;

    private const string DefaultManifestUrl = "https://raw.githubusercontent.com/iyulab/lm-supply/main/runtime-manifest.json";
    private const string ManifestCacheFileName = "runtime-manifest.json";
    private const string ETagCacheFileName = "runtime-manifest.etag";

    /// <summary>
    /// Creates a new manifest provider.
    /// </summary>
    /// <param name="httpClient">HTTP client for remote requests.</param>
    /// <param name="cacheDirectory">Directory for caching the manifest.</param>
    /// <param name="cacheExpiration">How long to use cached manifest before checking for updates.</param>
    public ManifestProvider(
        HttpClient? httpClient = null,
        string? cacheDirectory = null,
        TimeSpan? cacheExpiration = null)
    {
        _httpClient = httpClient ?? CreateDefaultHttpClient();
        _cacheDirectory = cacheDirectory ?? GetDefaultCacheDirectory();
        _cacheExpiration = cacheExpiration ?? TimeSpan.FromHours(1);
    }

    /// <summary>
    /// Gets the runtime manifest, using cache when available and not expired.
    /// </summary>
    /// <param name="forceRefresh">Force fetching from remote even if cache is valid.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    public async Task<RuntimeManifest> GetManifestAsync(
        bool forceRefresh = false,
        CancellationToken cancellationToken = default)
    {
        await _lock.WaitAsync(cancellationToken);
        try
        {
            // Return cached if valid and not forcing refresh
            if (!forceRefresh && _cachedManifest is not null &&
                DateTimeOffset.UtcNow - _lastFetched < _cacheExpiration)
            {
                return _cachedManifest;
            }

            // Try to load from disk cache
            if (_cachedManifest is null)
            {
                var diskManifest = await TryLoadFromDiskCacheAsync(cancellationToken);
                if (diskManifest is not null)
                {
                    _cachedManifest = diskManifest;
                    _cachedETag = await TryLoadETagFromDiskAsync(cancellationToken);
                }
            }

            // Try to fetch from remote with conditional request
            var remoteManifest = await TryFetchFromRemoteAsync(cancellationToken);
            if (remoteManifest is not null)
            {
                _cachedManifest = remoteManifest;
                _lastFetched = DateTimeOffset.UtcNow;
                return _cachedManifest;
            }

            // If we have a cached manifest, use it even if remote failed
            if (_cachedManifest is not null)
            {
                _lastFetched = DateTimeOffset.UtcNow;
                return _cachedManifest;
            }

            // Last resort: use embedded manifest
            return await LoadEmbeddedManifestAsync(cancellationToken);
        }
        finally
        {
            _lock.Release();
        }
    }

    /// <summary>
    /// Gets a specific binary entry from the manifest.
    /// </summary>
    public async Task<RuntimeBinaryEntry?> GetBinaryAsync(
        string packageName,
        string version,
        string runtimeIdentifier,
        string provider,
        CancellationToken cancellationToken = default)
    {
        var manifest = await GetManifestAsync(cancellationToken: cancellationToken);
        var package = manifest.GetPackage(packageName);
        if (package is null)
            return null;

        if (!package.Versions.TryGetValue(version, out var packageVersion))
            return null;

        return packageVersion.Binaries.FirstOrDefault(b =>
            b.RuntimeIdentifier.Equals(runtimeIdentifier, StringComparison.OrdinalIgnoreCase) &&
            b.Provider.Equals(provider, StringComparison.OrdinalIgnoreCase));
    }

    /// <summary>
    /// Gets the latest binary entry for a package and RID.
    /// </summary>
    public async Task<RuntimeBinaryEntry?> GetLatestBinaryAsync(
        string packageName,
        string runtimeIdentifier,
        string provider,
        CancellationToken cancellationToken = default)
    {
        var manifest = await GetManifestAsync(cancellationToken: cancellationToken);
        var package = manifest.GetPackage(packageName);
        if (package is null)
            return null;

        // Get the latest version
        var latestVersion = package.Versions
            .OrderByDescending(v => Version.TryParse(v.Key, out var ver) ? ver : new Version(0, 0))
            .FirstOrDefault();

        return latestVersion.Value?.Binaries.FirstOrDefault(b =>
            b.RuntimeIdentifier.Equals(runtimeIdentifier, StringComparison.OrdinalIgnoreCase) &&
            b.Provider.Equals(provider, StringComparison.OrdinalIgnoreCase));
    }

    private async Task<RuntimeManifest?> TryFetchFromRemoteAsync(CancellationToken cancellationToken)
    {
        try
        {
            var request = new HttpRequestMessage(HttpMethod.Get, DefaultManifestUrl);

            // Add ETag for conditional request (304 responses don't count against rate limit)
            if (!string.IsNullOrEmpty(_cachedETag))
            {
                request.Headers.IfNoneMatch.Add(new EntityTagHeaderValue(_cachedETag));
            }

            var response = await _httpClient.SendAsync(request, cancellationToken);

            // 304 Not Modified - use cached version
            if (response.StatusCode == HttpStatusCode.NotModified)
            {
                return null; // Signal to use cached version
            }

            if (!response.IsSuccessStatusCode)
            {
                return null;
            }

            // Parse the response
            var content = await response.Content.ReadAsStringAsync(cancellationToken);
            var manifest = RuntimeManifest.Parse(content);

            // Save to disk cache
            await SaveToDiskCacheAsync(content, cancellationToken);

            // Save ETag
            if (response.Headers.ETag is not null)
            {
                _cachedETag = response.Headers.ETag.Tag;
                await SaveETagToDiskAsync(_cachedETag, cancellationToken);
            }

            return manifest;
        }
        catch
        {
            return null;
        }
    }

    private async Task<RuntimeManifest?> TryLoadFromDiskCacheAsync(CancellationToken cancellationToken)
    {
        try
        {
            var cachePath = Path.Combine(_cacheDirectory, ManifestCacheFileName);
            if (!File.Exists(cachePath))
                return null;

            var content = await File.ReadAllTextAsync(cachePath, cancellationToken);
            return RuntimeManifest.Parse(content);
        }
        catch
        {
            return null;
        }
    }

    private async Task<string?> TryLoadETagFromDiskAsync(CancellationToken cancellationToken)
    {
        try
        {
            var etagPath = Path.Combine(_cacheDirectory, ETagCacheFileName);
            if (!File.Exists(etagPath))
                return null;

            return await File.ReadAllTextAsync(etagPath, cancellationToken);
        }
        catch
        {
            return null;
        }
    }

    private async Task SaveToDiskCacheAsync(string content, CancellationToken cancellationToken)
    {
        try
        {
            Directory.CreateDirectory(_cacheDirectory);
            var cachePath = Path.Combine(_cacheDirectory, ManifestCacheFileName);

            // Write to temp file first, then move (atomic operation)
            var tempPath = cachePath + $".{Guid.NewGuid()}.tmp";
            await File.WriteAllTextAsync(tempPath, content, cancellationToken);
            File.Move(tempPath, cachePath, overwrite: true);
        }
        catch
        {
            // Ignore cache write failures
        }
    }

    private async Task SaveETagToDiskAsync(string etag, CancellationToken cancellationToken)
    {
        try
        {
            Directory.CreateDirectory(_cacheDirectory);
            var etagPath = Path.Combine(_cacheDirectory, ETagCacheFileName);
            await File.WriteAllTextAsync(etagPath, etag, cancellationToken);
        }
        catch
        {
            // Ignore cache write failures
        }
    }

    private static async Task<RuntimeManifest> LoadEmbeddedManifestAsync(CancellationToken cancellationToken)
    {
        var assembly = Assembly.GetExecutingAssembly();
        var resourceName = assembly.GetManifestResourceNames()
            .FirstOrDefault(n => n.EndsWith("runtime-manifest.json", StringComparison.OrdinalIgnoreCase));

        if (resourceName is null)
        {
            // Return a minimal default manifest if no embedded resource
            return CreateDefaultManifest();
        }

        using var stream = assembly.GetManifestResourceStream(resourceName)
            ?? throw new InvalidOperationException("Failed to load embedded manifest");

        return await RuntimeManifest.ParseAsync(stream, cancellationToken);
    }

    private static RuntimeManifest CreateDefaultManifest()
    {
        return new RuntimeManifest
        {
            Version = "1.0.0",
            Updated = DateTimeOffset.UtcNow,
            Packages = new Dictionary<string, RuntimePackage>
            {
                ["onnxruntime"] = new RuntimePackage
                {
                    Description = "ONNX Runtime for cross-platform machine learning inference",
                    Homepage = "https://onnxruntime.ai",
                    Versions = new Dictionary<string, RuntimePackageVersion>()
                }
            }
        };
    }

    private static HttpClient CreateDefaultHttpClient()
    {
        var handler = new HttpClientHandler
        {
            UseProxy = true,
            Proxy = WebRequest.GetSystemWebProxy()
        };

        var client = new HttpClient(handler)
        {
            Timeout = TimeSpan.FromSeconds(30)
        };

        client.DefaultRequestHeaders.UserAgent.ParseAdd("LMSupply/1.0");
        client.DefaultRequestHeaders.Accept.Add(new MediaTypeWithQualityHeaderValue("application/json"));

        return client;
    }

    private static string GetDefaultCacheDirectory()
    {
        // Follow HuggingFace Hub conventions
        var hfCache = Environment.GetEnvironmentVariable("HF_HUB_CACHE");
        if (!string.IsNullOrEmpty(hfCache))
            return Path.Combine(hfCache, "LMSupply");

        var hfHome = Environment.GetEnvironmentVariable("HF_HOME");
        if (!string.IsNullOrEmpty(hfHome))
            return Path.Combine(hfHome, "hub", "LMSupply");

        var xdgCache = Environment.GetEnvironmentVariable("XDG_CACHE_HOME");
        if (!string.IsNullOrEmpty(xdgCache))
            return Path.Combine(xdgCache, "huggingface", "hub", "LMSupply");

        // Platform-specific defaults
        if (OperatingSystem.IsWindows())
        {
            var localAppData = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
            return Path.Combine(localAppData, "huggingface", "hub", "LMSupply");
        }
        else if (OperatingSystem.IsMacOS())
        {
            var home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
            return Path.Combine(home, "Library", "Caches", "huggingface", "hub", "LMSupply");
        }
        else
        {
            var home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
            return Path.Combine(home, ".cache", "huggingface", "hub", "LMSupply");
        }
    }

    public void Dispose()
    {
        _lock.Dispose();
    }
}
