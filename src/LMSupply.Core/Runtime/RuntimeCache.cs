using System.Text.Json;
using System.Text.Json.Serialization;

namespace LMSupply.Runtime;

/// <summary>
/// Manages a cache of downloaded runtime binaries with version isolation and LRU eviction.
/// Cache structure: {baseDirectory}/binaries/{package}/{version}/{rid}/{provider}/
/// </summary>
public sealed class RuntimeCache : IDisposable
{
    private readonly string _baseDirectory;
    private readonly RuntimeCacheOptions _options;
    private readonly SemaphoreSlim _lock = new(1, 1);
    private readonly string _metadataPath;

    private CacheMetadata _metadata;

    private const string BinariesDirectory = "binaries";
    private const string MetadataFileName = "cache-metadata.json";

    /// <summary>
    /// Creates a new runtime cache with default options.
    /// </summary>
    public RuntimeCache() : this(new RuntimeCacheOptions())
    {
    }

    /// <summary>
    /// Creates a new runtime cache with custom options.
    /// </summary>
    public RuntimeCache(RuntimeCacheOptions options)
    {
        _options = options;
        _baseDirectory = options.CacheDirectory ?? GetDefaultCacheDirectory();
        _metadataPath = Path.Combine(_baseDirectory, MetadataFileName);
        _metadata = LoadMetadata();
    }

    /// <summary>
    /// Gets the path to a cached binary, or null if not cached.
    /// Updates the last access time for LRU tracking.
    /// </summary>
    public async Task<string?> GetCachedPathAsync(
        string package,
        string version,
        string runtimeIdentifier,
        string provider,
        CancellationToken cancellationToken = default)
    {
        var key = GetCacheKey(package, version, runtimeIdentifier, provider);
        var directory = GetCacheDirectory(package, version, runtimeIdentifier, provider);

        await _lock.WaitAsync(cancellationToken);
        try
        {
            if (!_metadata.Entries.TryGetValue(key, out var entry))
                return null;

            var binaryPath = Path.Combine(directory, entry.FileName);
            if (!File.Exists(binaryPath))
            {
                // Cache entry exists but file is missing - clean up
                _metadata.Entries.Remove(key);
                await SaveMetadataAsync(cancellationToken);
                return null;
            }

            // Update last access time for LRU
            entry.LastAccessTime = DateTimeOffset.UtcNow;
            await SaveMetadataAsync(cancellationToken);

            return binaryPath;
        }
        finally
        {
            _lock.Release();
        }
    }

    /// <summary>
    /// Registers a binary in the cache.
    /// </summary>
    public async Task<string> RegisterAsync(
        RuntimeBinaryEntry binaryEntry,
        string package,
        string version,
        string sourcePath,
        CancellationToken cancellationToken = default)
    {
        var key = GetCacheKey(package, version, binaryEntry.RuntimeIdentifier, binaryEntry.Provider);
        var directory = GetCacheDirectory(package, version, binaryEntry.RuntimeIdentifier, binaryEntry.Provider);
        var targetPath = Path.Combine(directory, binaryEntry.FileName);

        await _lock.WaitAsync(cancellationToken);
        try
        {
            // Ensure cache size limit before adding
            await EnsureCacheSizeLimitAsync(binaryEntry.Size, cancellationToken);

            // Create directory and copy file
            Directory.CreateDirectory(directory);

            if (sourcePath != targetPath)
            {
                File.Copy(sourcePath, targetPath, overwrite: true);
            }

            // Register in metadata
            var entry = new CacheEntry
            {
                Package = package,
                Version = version,
                RuntimeIdentifier = binaryEntry.RuntimeIdentifier,
                Provider = binaryEntry.Provider,
                FileName = binaryEntry.FileName,
                Size = binaryEntry.Size,
                Sha256 = binaryEntry.Sha256,
                CachedTime = DateTimeOffset.UtcNow,
                LastAccessTime = DateTimeOffset.UtcNow
            };

            _metadata.Entries[key] = entry;
            await SaveMetadataAsync(cancellationToken);

            return targetPath;
        }
        finally
        {
            _lock.Release();
        }
    }

    /// <summary>
    /// Gets the directory path for a cache entry (creates if doesn't exist).
    /// </summary>
    public string GetCacheDirectory(string package, string version, string runtimeIdentifier, string provider)
    {
        return Path.Combine(_baseDirectory, BinariesDirectory, package, version, runtimeIdentifier, provider);
    }

    /// <summary>
    /// Gets all cached entries.
    /// </summary>
    public IReadOnlyDictionary<string, CacheEntry> GetAllEntries()
    {
        return _metadata.Entries;
    }

    /// <summary>
    /// Gets the total cache size in bytes.
    /// </summary>
    public long GetTotalCacheSize()
    {
        return _metadata.Entries.Values.Sum(e => e.Size);
    }

    /// <summary>
    /// Removes a specific entry from the cache.
    /// </summary>
    public async Task RemoveAsync(
        string package,
        string version,
        string runtimeIdentifier,
        string provider,
        CancellationToken cancellationToken = default)
    {
        var key = GetCacheKey(package, version, runtimeIdentifier, provider);
        var directory = GetCacheDirectory(package, version, runtimeIdentifier, provider);

        await _lock.WaitAsync(cancellationToken);
        try
        {
            if (_metadata.Entries.Remove(key))
            {
                // Delete the directory and files
                if (Directory.Exists(directory))
                {
                    Directory.Delete(directory, recursive: true);
                }

                // Clean up empty parent directories
                CleanupEmptyDirectories(Path.GetDirectoryName(directory)!);

                await SaveMetadataAsync(cancellationToken);
            }
        }
        finally
        {
            _lock.Release();
        }
    }

    /// <summary>
    /// Clears all cached binaries.
    /// </summary>
    public async Task ClearAsync(CancellationToken cancellationToken = default)
    {
        await _lock.WaitAsync(cancellationToken);
        try
        {
            var binariesPath = Path.Combine(_baseDirectory, BinariesDirectory);
            if (Directory.Exists(binariesPath))
            {
                Directory.Delete(binariesPath, recursive: true);
            }

            _metadata.Entries.Clear();
            await SaveMetadataAsync(cancellationToken);
        }
        finally
        {
            _lock.Release();
        }
    }

    /// <summary>
    /// Ensures the cache doesn't exceed the size limit by evicting LRU entries.
    /// </summary>
    private async Task EnsureCacheSizeLimitAsync(long newEntrySize, CancellationToken cancellationToken)
    {
        var currentSize = GetTotalCacheSize();
        var targetSize = _options.MaxCacheSize - newEntrySize;

        if (currentSize <= targetSize)
            return;

        // Get entries sorted by last access time (oldest first)
        var entriesToEvict = _metadata.Entries
            .OrderBy(e => e.Value.LastAccessTime)
            .ToList();

        foreach (var entry in entriesToEvict)
        {
            if (currentSize <= targetSize)
                break;

            // Delete the files
            var directory = GetCacheDirectory(
                entry.Value.Package,
                entry.Value.Version,
                entry.Value.RuntimeIdentifier,
                entry.Value.Provider);

            if (Directory.Exists(directory))
            {
                try
                {
                    Directory.Delete(directory, recursive: true);
                    CleanupEmptyDirectories(Path.GetDirectoryName(directory)!);
                }
                catch
                {
                    // Ignore deletion errors for now
                }
            }

            currentSize -= entry.Value.Size;
            _metadata.Entries.Remove(entry.Key);
        }
    }

    private static string GetCacheKey(string package, string version, string runtimeIdentifier, string provider)
    {
        return $"{package}|{version}|{runtimeIdentifier}|{provider}".ToLowerInvariant();
    }

    private void CleanupEmptyDirectories(string directory)
    {
        try
        {
            var binariesPath = Path.Combine(_baseDirectory, BinariesDirectory);
            while (!string.IsNullOrEmpty(directory) &&
                   directory.StartsWith(binariesPath, StringComparison.OrdinalIgnoreCase) &&
                   Directory.Exists(directory) &&
                   !Directory.EnumerateFileSystemEntries(directory).Any())
            {
                Directory.Delete(directory);
                directory = Path.GetDirectoryName(directory)!;
            }
        }
        catch
        {
            // Ignore cleanup errors
        }
    }

    private CacheMetadata LoadMetadata()
    {
        try
        {
            if (File.Exists(_metadataPath))
            {
                var json = File.ReadAllText(_metadataPath);
                return JsonSerializer.Deserialize<CacheMetadata>(json) ?? new CacheMetadata();
            }
        }
        catch
        {
            // Ignore load errors, start fresh
        }

        return new CacheMetadata();
    }

    private async Task SaveMetadataAsync(CancellationToken cancellationToken)
    {
        try
        {
            Directory.CreateDirectory(_baseDirectory);
            var json = JsonSerializer.Serialize(_metadata, new JsonSerializerOptions { WriteIndented = true });

            // Atomic write
            var tempPath = _metadataPath + $".{Guid.NewGuid()}.tmp";
            await File.WriteAllTextAsync(tempPath, json, cancellationToken);
            File.Move(tempPath, _metadataPath, overwrite: true);
        }
        catch
        {
            // Ignore save errors
        }
    }

    private static string GetDefaultCacheDirectory()
    {
        // Follow HuggingFace Hub conventions
        var hfCache = Environment.GetEnvironmentVariable("HF_HUB_CACHE");
        if (!string.IsNullOrEmpty(hfCache))
            return Path.Combine(hfCache, "LMSupply", "runtime");

        var hfHome = Environment.GetEnvironmentVariable("HF_HOME");
        if (!string.IsNullOrEmpty(hfHome))
            return Path.Combine(hfHome, "hub", "LMSupply", "runtime");

        var xdgCache = Environment.GetEnvironmentVariable("XDG_CACHE_HOME");
        if (!string.IsNullOrEmpty(xdgCache))
            return Path.Combine(xdgCache, "huggingface", "hub", "LMSupply", "runtime");

        if (OperatingSystem.IsWindows())
        {
            var localAppData = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
            return Path.Combine(localAppData, "huggingface", "hub", "LMSupply", "runtime");
        }
        else if (OperatingSystem.IsMacOS())
        {
            var home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
            return Path.Combine(home, "Library", "Caches", "huggingface", "hub", "LMSupply", "runtime");
        }
        else
        {
            var home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
            return Path.Combine(home, ".cache", "huggingface", "hub", "LMSupply", "runtime");
        }
    }

    public void Dispose()
    {
        _lock.Dispose();
    }
}

/// <summary>
/// Options for the runtime cache.
/// </summary>
public sealed class RuntimeCacheOptions
{
    /// <summary>
    /// Gets or sets the cache directory. Defaults to HuggingFace Hub cache conventions.
    /// </summary>
    public string? CacheDirectory { get; set; }

    /// <summary>
    /// Gets or sets the maximum cache size in bytes. Default is 10 GB.
    /// </summary>
    public long MaxCacheSize { get; set; } = 10L * 1024 * 1024 * 1024; // 10 GB

    /// <summary>
    /// Gets or sets whether to verify checksums when loading from cache.
    /// </summary>
    public bool VerifyChecksums { get; set; } = true;
}

/// <summary>
/// Metadata for the cache.
/// </summary>
internal sealed class CacheMetadata
{
    [JsonPropertyName("entries")]
    public Dictionary<string, CacheEntry> Entries { get; set; } = new();
}

/// <summary>
/// A single cache entry.
/// </summary>
public sealed class CacheEntry
{
    [JsonPropertyName("package")]
    public required string Package { get; set; }

    [JsonPropertyName("version")]
    public required string Version { get; set; }

    [JsonPropertyName("rid")]
    public required string RuntimeIdentifier { get; set; }

    [JsonPropertyName("provider")]
    public required string Provider { get; set; }

    [JsonPropertyName("fileName")]
    public required string FileName { get; set; }

    [JsonPropertyName("size")]
    public required long Size { get; set; }

    [JsonPropertyName("sha256")]
    public required string Sha256 { get; set; }

    [JsonPropertyName("cachedTime")]
    public DateTimeOffset CachedTime { get; set; }

    [JsonPropertyName("lastAccessTime")]
    public DateTimeOffset LastAccessTime { get; set; }
}
