using System.Diagnostics;
using System.IO.Compression;
using LMSupply.Download;

namespace LMSupply.Runtime;

/// <summary>
/// Downloads ONNX Runtime packages from NuGet.org and extracts native binaries.
/// Implements LMSupply's on-demand philosophy: binaries are downloaded only when first needed.
/// Supports both standard ONNX Runtime and ONNX Runtime GenAI packages.
/// </summary>
public sealed class OnnxNuGetDownloader : IDisposable
{
    private readonly HttpClient _httpClient;
    private readonly NuGetPackageResolver _packageResolver;
    private readonly string _cacheDirectory;
    private readonly bool _ownsHttpClient;

    public OnnxNuGetDownloader() : this(null)
    {
    }

    public OnnxNuGetDownloader(string? cacheDirectory)
        : this(cacheDirectory, handler: null)
    {
    }

    /// <summary>Test seam: the same downloader over a caller-supplied transport.</summary>
    internal OnnxNuGetDownloader(string? cacheDirectory, HttpMessageHandler? handler)
    {
        _cacheDirectory = cacheDirectory ?? LMSupplyCachePaths.GetRuntimesDirectory();

        // Create HttpClient first, then wrap in try-catch to ensure cleanup on failure
        var httpClient = handler is null ? new HttpClient() : new HttpClient(handler);
        try
        {
            httpClient.DefaultRequestHeaders.Add("User-Agent", "LMSupply/1.0");
            _packageResolver = new NuGetPackageResolver(httpClient);
            _httpClient = httpClient;
            _ownsHttpClient = true;
        }
        catch
        {
            httpClient.Dispose();
            throw;
        }
    }

    /// <summary>
    /// Downloads runtime binaries for the specified package type and provider.
    /// </summary>
    /// <param name="provider">The execution provider (cpu, cuda12, etc.).</param>
    /// <param name="platform">The target platform info.</param>
    /// <param name="version">Optional version. If null, uses the latest stable version.</param>
    /// <param name="progress">Optional progress reporter.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <param name="packageType">The package type: "onnxruntime" (default) or "onnxruntime-genai".</param>
    /// <returns>Path to the extracted native binaries directory.</returns>
    public async Task<string> DownloadAsync(
        string provider,
        PlatformInfo platform,
        string? version = null,
        IProgress<DownloadProgress>? progress = null,
        string packageType = RuntimePackageRegistry.PackageTypes.OnnxRuntime,
        CancellationToken cancellationToken = default)
    {
        // Get package configuration from registry
        var config = RuntimePackageRegistry.GetPackageConfig(
            packageType,
            provider,
            platform.RuntimeIdentifier);

        if (config is null)
        {
            throw new InvalidOperationException(
                $"No package configuration found for {packageType}/{provider}");
        }

        // The version to serve is the one the loaded managed assembly expects (verified against the feed,
        // falling back to the package line's latest when that exact number does not exist for it) or the
        // one the caller named. A cached binary of some OTHER version is only ever a stand-in for an
        // unreachable feed - never a substitute for a download that would succeed. Preferring "whatever
        // is cached" ran managed 1.30.0 on a native 1.24.4 for as long as the cache existed.
        string? requestedVersion = version;
        string? feedFailure = null;
        if (requestedVersion is null)
        {
            try
            {
                requestedVersion = await ResolveVersionAsync(config.PackageId, cancellationToken);
            }
            catch (Exception ex) when (IsFeedUnreachable(ex, cancellationToken))
            {
                feedFailure = ex.Message;
            }
        }

        if (requestedVersion is not null)
        {
            var cachePath = GetCachePath(packageType, provider, requestedVersion, platform);
            if (Directory.Exists(cachePath) && IsValidCache(cachePath, config, platform))
            {
                Trace.TraceInformation($"[OnnxNuGetDownloader] Using cached binaries: {cachePath}");
                ReportCacheHit(progress);
                return cachePath;
            }

            try
            {
                return await DownloadAndExtractAsync(
                    config,
                    requestedVersion,
                    platform,
                    cachePath,
                    progress,
                    cancellationToken);
            }
            catch (Exception ex) when (IsFeedUnreachable(ex, cancellationToken))
            {
                feedFailure = ex.Message;
            }
        }

        // Feed unreachable: the newest cached version of this package is the only thing that can run.
        // Say so - a consumer that pinned a newer managed runtime is now on a different native.
        var existingCache = FindExistingCache(packageType, provider, platform, config);
        if (existingCache is not null)
        {
            var wanted = requestedVersion is null ? "the version the loaded runtime expects" : $"the requested {requestedVersion}";
            Trace.TraceWarning(
                $"[OnnxNuGetDownloader] Package feed unreachable ({feedFailure}); using cached {existingCache} instead of {wanted}. " +
                $"The managed and native ONNX Runtime versions may differ until the feed is reachable again.");
            ReportCacheHit(progress);
            return existingCache;
        }

        throw new InvalidOperationException(
            $"Cannot provision {config.PackageId}: the package feed is unreachable ({feedFailure}) and no cached version exists.");
    }

    /// <summary>
    /// A lookup/download failure caused by the network or the feed, as opposed to a caller cancellation
    /// or a bug - the only failures a stale cached version is allowed to paper over.
    /// </summary>
    private static bool IsFeedUnreachable(Exception ex, CancellationToken cancellationToken)
        => !cancellationToken.IsCancellationRequested
           && ex is HttpRequestException or TaskCanceledException or System.Net.Sockets.SocketException or IOException;

    /// <summary>
    /// Resolves the version to use, either from assembly or NuGet API.
    /// Validates that the version exists for the specific package before using it.
    /// </summary>
    private async Task<string> ResolveVersionAsync(
        string packageId,
        CancellationToken cancellationToken)
    {
        // First try to detect from loaded assembly
        var assemblyVersion = TryGetAssemblyVersion(packageId);
        if (!string.IsNullOrEmpty(assemblyVersion))
        {
            // Verify this version exists for the specific package
            // (e.g., a provider package on a different version line than base ONNX Runtime)
            var availableVersions = await _packageResolver.GetVersionsAsync(packageId, cancellationToken);
            if (availableVersions.Contains(assemblyVersion, StringComparer.OrdinalIgnoreCase))
            {
                Trace.TraceInformation($"[OnnxNuGetDownloader] Using assembly version: {assemblyVersion}");
                return assemblyVersion;
            }

            Trace.TraceInformation($"[OnnxNuGetDownloader] Assembly version {assemblyVersion} not found for {packageId}, falling back to latest");
        }

        // Get latest stable version from NuGet
        var latestVersion = await _packageResolver.GetLatestVersionAsync(
            packageId,
            includePrerelease: false,
            cancellationToken);

        if (string.IsNullOrEmpty(latestVersion))
        {
            throw new InvalidOperationException(
                $"Could not determine version for package {packageId}. " +
                "Please specify a version explicitly.");
        }

        Trace.TraceInformation($"[OnnxNuGetDownloader] Using latest NuGet version: {latestVersion}");
        return latestVersion;
    }

    /// <summary>
    /// Downloads a package and extracts native binaries.
    /// </summary>
    private async Task<string> DownloadAndExtractAsync(
        RuntimePackageRegistry.PackageConfig config,
        string version,
        PlatformInfo platform,
        string cachePath,
        IProgress<DownloadProgress>? progress,
        CancellationToken cancellationToken)
    {
        var downloadUrl = NuGetPackageResolver.GetPackageDownloadUrl(config.PackageId, version);

        progress?.Report(new DownloadProgress
        {
            FileName = $"{config.PackageId}.{version}.nupkg",
            BytesDownloaded = 0,
            TotalBytes = 0
        });

        var tempDir = Path.Combine(Path.GetTempPath(), $"lmsupply-onnx-{Guid.NewGuid()}");

        try
        {
            Directory.CreateDirectory(tempDir);

            // Download .nupkg
            var nupkgPath = Path.Combine(tempDir, $"{config.PackageId}.{version}.nupkg");
            await DownloadFileAsync(downloadUrl, nupkgPath, config.PackageId, progress, cancellationToken);

            // Extract native binaries
            progress?.Report(new DownloadProgress
            {
                FileName = "Extracting native libraries...",
                BytesDownloaded = 0,
                TotalBytes = 0
            });

            var extractedPath = await ExtractNativeBinariesAsync(
                nupkgPath, platform, cancellationToken);

            if (extractedPath is null)
            {
                throw new InvalidOperationException(
                    $"No native binaries found for {platform.RuntimeIdentifier} in {config.PackageId}");
            }

            // Move to cache
            EnsureCacheDirectory(cachePath);
            Directory.Move(extractedPath, cachePath);

            progress?.Report(new DownloadProgress
            {
                FileName = $"{config.NativeLibraryName} ready",
                BytesDownloaded = 1,
                TotalBytes = 1
            });

            return cachePath;
        }
        finally
        {
            CleanupTempDirectory(tempDir);
        }
    }

    /// <summary>
    /// Extracts native binaries from a .nupkg file for the specified platform.
    /// </summary>
    private static async Task<string?> ExtractNativeBinariesAsync(
        string nupkgPath,
        PlatformInfo platform,
        CancellationToken cancellationToken)
    {
        var tempExtract = Path.Combine(Path.GetDirectoryName(nupkgPath)!, "extracted");
        Directory.CreateDirectory(tempExtract);

        using var archive = ZipFile.OpenRead(nupkgPath);

        // NuGet package structure: runtimes/{rid}/native/
        var runtimeIdentifiers = GetRuntimeIdentifiers(platform);

        foreach (var rid in runtimeIdentifiers)
        {
            var nativePrefix = $"runtimes/{rid}/native/";

            var nativeEntries = archive.Entries
                .Where(e => e.FullName.StartsWith(nativePrefix, StringComparison.OrdinalIgnoreCase)
                            && !string.IsNullOrEmpty(e.Name))
                .ToList();

            if (nativeEntries.Count > 0)
            {
                var outputDir = Path.Combine(tempExtract, rid);
                Directory.CreateDirectory(outputDir);

                foreach (var entry in nativeEntries)
                {
                    var destPath = Path.Combine(outputDir, entry.Name);
                    entry.ExtractToFile(destPath, overwrite: true);

                    // Set executable permission on Unix
                    if (!platform.IsWindows)
                    {
                        await SetExecutableAsync(destPath, cancellationToken);
                    }
                }

                return outputDir;
            }
        }

        return null;
    }

    /// <summary>
    /// Gets runtime identifiers to search for in order of preference.
    /// </summary>
    private static string[] GetRuntimeIdentifiers(PlatformInfo platform)
    {
        var arch = platform.Architecture == System.Runtime.InteropServices.Architecture.Arm64 ? "arm64" : "x64";

        if (platform.IsWindows)
            return [$"win-{arch}", "win"];
        if (platform.IsLinux)
            return [$"linux-{arch}", "linux"];
        if (platform.IsMacOS)
            return [$"osx-{arch}", "osx"];

        return ["any"];
    }

    private string GetCachePath(string packageType, string provider, string version, PlatformInfo platform)
    {
        return Path.Combine(
            _cacheDirectory,
            packageType,
            provider.ToLowerInvariant(),
            version,
            platform.RuntimeIdentifier);
    }

    /// <summary>
    /// Scans for any existing valid cached version of the package, returning the newest one found.
    /// Used as a fallback when the requested version is not cached.
    /// </summary>
    private string? FindExistingCache(
        string packageType,
        string provider,
        PlatformInfo platform,
        RuntimePackageRegistry.PackageConfig config)
    {
        var providerCacheDir = Path.Combine(_cacheDirectory, packageType, provider.ToLowerInvariant());
        if (!Directory.Exists(providerCacheDir))
            return null;

        // Prefer newest version - by version, not by string ("1.9.0" must not outrank "1.30.0").
        foreach (var versionDir in Directory.GetDirectories(providerCacheDir)
                     .OrderByDescending(dir => Version.TryParse(Path.GetFileName(dir), out var v) ? v : new Version(0, 0)))
        {
            var candidatePath = Path.Combine(versionDir, platform.RuntimeIdentifier);
            if (Directory.Exists(candidatePath) && IsValidCache(candidatePath, config, platform))
                return candidatePath;
        }

        return null;
    }

    private static bool IsValidCache(
        string path,
        RuntimePackageRegistry.PackageConfig config,
        PlatformInfo platform)
    {
        if (!Directory.Exists(path))
            return false;

        var expectedLib = RuntimePackageRegistry.GetNativeLibraryFileName(
            config.NativeLibraryName, platform);

        return Directory.EnumerateFiles(path, expectedLib + "*").Any();
    }

    // A package archive is only the archive at the length the feed announced; a body that ends early is
    // resumed from its ".part" rather than handed to the extractor as a corrupt zip.
    private Task DownloadFileAsync(
        string url,
        string destinationPath,
        string fileName,
        IProgress<DownloadProgress>? progress,
        CancellationToken cancellationToken)
        => LMSupply.Download.ResumableFileDownload.DownloadAsync(_httpClient, new LMSupply.Download.ResumableFileDownload.Request
        {
            Url = url,
            DestinationPath = destinationPath,
            FileName = fileName,
            Progress = progress,
        }, cancellationToken);

    private static async Task SetExecutableAsync(string path, CancellationToken cancellationToken)
    {
        try
        {
            var process = Process.Start(new ProcessStartInfo
            {
                FileName = "chmod",
                Arguments = $"+x \"{path}\"",
                UseShellExecute = false,
                CreateNoWindow = true
            });

            if (process != null)
            {
                await process.WaitForExitAsync(cancellationToken);
            }
        }
        catch (Exception ex)
        {
            Trace.TraceInformation($"[OnnxNuGetDownloader] dotnet command failed: {ex.Message}");
        }
    }

    private static string? TryGetAssemblyVersion(string packageId)
    {
        try
        {
            // Map package ID to assembly name
            var assemblyName = packageId.Replace(".ML.", ".ML.")
                .Replace("GenAI", "RuntimeGenAI");

            // Handle specific mappings
            if (packageId.Contains("OnnxRuntimeGenAI", StringComparison.OrdinalIgnoreCase))
            {
                assemblyName = "Microsoft.ML.OnnxRuntimeGenAI";
            }
            else if (packageId.Contains("OnnxRuntime", StringComparison.OrdinalIgnoreCase))
            {
                assemblyName = "Microsoft.ML.OnnxRuntime";
            }

            var assembly = AppDomain.CurrentDomain.GetAssemblies()
                .FirstOrDefault(a => a.GetName().Name?.Equals(assemblyName, StringComparison.OrdinalIgnoreCase) == true);

            if (assembly != null)
            {
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
        }
        catch (Exception ex)
        {
            Trace.TraceInformation($"[OnnxNuGetDownloader] Assembly version lookup failed: {ex.Message}");
        }

        return null;
    }

    private static void ReportCacheHit(IProgress<DownloadProgress>? progress)
    {
        progress?.Report(new DownloadProgress
        {
            FileName = "Using cached runtime (already downloaded)",
            BytesDownloaded = 1,
            TotalBytes = 1
        });
    }

    private static void EnsureCacheDirectory(string cachePath)
    {
        var parentDir = Path.GetDirectoryName(cachePath);
        if (!string.IsNullOrEmpty(parentDir))
        {
            Directory.CreateDirectory(parentDir);
        }

        if (Directory.Exists(cachePath))
        {
            Directory.Delete(cachePath, recursive: true);
        }
    }

    private static void CleanupTempDirectory(string tempDir)
    {
        try
        {
            if (Directory.Exists(tempDir))
            {
                Directory.Delete(tempDir, recursive: true);
            }
        }
        catch (Exception ex)
        {
            Trace.TraceInformation($"[OnnxNuGetDownloader] Temp directory cleanup failed: {ex.Message}");
        }
    }

    public void Dispose()
    {
        _packageResolver.Dispose();
        if (_ownsHttpClient)
        {
            _httpClient.Dispose();
        }
    }
}
