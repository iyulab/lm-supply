using System.IO.Compression;
using System.Net;
using System.Security.Cryptography;
using LMSupply.Download;
using LMSupply.Exceptions;

namespace LMSupply.Runtime;

/// <summary>
/// Downloads runtime binaries with retry logic, proxy support, and checksum verification.
/// Implements cross-process locking to prevent concurrent downloads of the same binary.
/// </summary>
public sealed class BinaryDownloader : IDisposable
{
    private readonly HttpClient _httpClient;
    private readonly BinaryDownloaderOptions _options;

    /// <summary>
    /// Creates a new binary downloader with default options.
    /// </summary>
    public BinaryDownloader() : this(new BinaryDownloaderOptions())
    {
    }

    /// <summary>
    /// Creates a new binary downloader with custom options.
    /// </summary>
    public BinaryDownloader(BinaryDownloaderOptions options)
    {
        _options = options;
        _httpClient = CreateHttpClient(options);
    }

    /// <summary>
    /// Downloads a binary from the manifest entry to the specified directory.
    /// </summary>
    /// <param name="entry">The binary entry from the manifest.</param>
    /// <param name="targetDirectory">Directory to extract the binary to.</param>
    /// <param name="progress">Optional progress reporter.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Path to the downloaded binary file.</returns>
    public async Task<string> DownloadAsync(
        RuntimeBinaryEntry entry,
        string targetDirectory,
        IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(entry);
        ArgumentException.ThrowIfNullOrEmpty(targetDirectory);

        Directory.CreateDirectory(targetDirectory);

        var targetPath = Path.Combine(targetDirectory, entry.FileName);

        // Check if already exists with correct checksum
        if (File.Exists(targetPath) && await VerifyChecksumAsync(targetPath, entry.Sha256, cancellationToken))
        {
            progress?.Report(new DownloadProgress
            {
                FileName = entry.FileName,
                BytesDownloaded = entry.Size,
                TotalBytes = entry.Size
            });
            return targetPath;
        }

        // Use cross-process lock to prevent concurrent downloads
        var mutexName = GetMutexName(entry);
        using var mutex = new Mutex(false, mutexName);

        try
        {
            if (!mutex.WaitOne(_options.MutexTimeout))
            {
                throw new TimeoutException($"Timeout waiting for download lock: {entry.FileName}");
            }

            // Double-check after acquiring lock
            if (File.Exists(targetPath) && await VerifyChecksumAsync(targetPath, entry.Sha256, cancellationToken))
            {
                return targetPath;
            }

            // Download with retry
            var downloadPath = await DownloadWithRetryAsync(entry, progress, cancellationToken);

            try
            {
                // Extract if archive
                if (IsArchive(entry.Url))
                {
                    await ExtractArchiveAsync(downloadPath, targetDirectory, entry.FileName, cancellationToken);
                }
                else
                {
                    // Direct file - move to target
                    var finalPath = Path.Combine(targetDirectory, entry.FileName);
                    File.Move(downloadPath, finalPath, overwrite: true);
                }

                // Verify checksum
                if (!entry.Sha256.StartsWith("placeholder", StringComparison.OrdinalIgnoreCase))
                {
                    if (!await VerifyChecksumAsync(targetPath, entry.Sha256, cancellationToken))
                    {
                        File.Delete(targetPath);
                        throw new LMSupplyException($"Checksum verification failed for {entry.FileName}");
                    }
                }

                return targetPath;
            }
            finally
            {
                // Clean up temp download file
                if (File.Exists(downloadPath))
                {
                    try { File.Delete(downloadPath); } catch { }
                }
            }
        }
        finally
        {
            try { mutex.ReleaseMutex(); } catch { }
        }
    }

    /// <summary>
    /// Downloads a binary with exponential backoff retry.
    /// </summary>
    private async Task<string> DownloadWithRetryAsync(
        RuntimeBinaryEntry entry,
        IProgress<DownloadProgress>? progress,
        CancellationToken cancellationToken)
    {
        var tempPath = Path.Combine(Path.GetTempPath(), $"LMSupply_{Guid.NewGuid()}{GetArchiveExtension(entry.Url)}");

        Exception? lastException = null;

        for (int attempt = 0; attempt <= _options.MaxRetries; attempt++)
        {
            if (attempt > 0)
            {
                // Exponential backoff: 1s, 2s, 4s, 8s...
                var delay = TimeSpan.FromSeconds(Math.Pow(2, attempt - 1));
                await Task.Delay(delay, cancellationToken);
            }

            try
            {
                await DownloadFileAsync(entry.Url, tempPath, entry.Size, entry.FileName, progress, cancellationToken);
                return tempPath;
            }
            catch (OperationCanceledException)
            {
                throw;
            }
            catch (Exception ex)
            {
                lastException = ex;

                // Clean up partial download
                if (File.Exists(tempPath))
                {
                    try { File.Delete(tempPath); } catch { }
                }
            }
        }

        throw lastException is not null
            ? new LMSupplyException($"Failed to download {entry.FileName} after {_options.MaxRetries + 1} attempts", lastException)
            : new LMSupplyException($"Failed to download {entry.FileName} after {_options.MaxRetries + 1} attempts");
    }

    /// <summary>
    /// Downloads a file with progress reporting.
    /// </summary>
    private async Task DownloadFileAsync(
        string url,
        string targetPath,
        long expectedSize,
        string fileName,
        IProgress<DownloadProgress>? progress,
        CancellationToken cancellationToken)
    {
        using var response = await _httpClient.GetAsync(url, HttpCompletionOption.ResponseHeadersRead, cancellationToken);
        response.EnsureSuccessStatusCode();

        var totalBytes = response.Content.Headers.ContentLength ?? expectedSize;

        await using var contentStream = await response.Content.ReadAsStreamAsync(cancellationToken);
        await using var fileStream = new FileStream(
            targetPath,
            FileMode.Create,
            FileAccess.Write,
            FileShare.None,
            bufferSize: 81920,
            useAsync: true);

        var buffer = new byte[81920];
        long totalBytesRead = 0;
        int bytesRead;

        while ((bytesRead = await contentStream.ReadAsync(buffer, cancellationToken)) > 0)
        {
            await fileStream.WriteAsync(buffer.AsMemory(0, bytesRead), cancellationToken);
            totalBytesRead += bytesRead;

            progress?.Report(new DownloadProgress
            {
                FileName = fileName,
                BytesDownloaded = totalBytesRead,
                TotalBytes = totalBytes
            });
        }
    }

    /// <summary>
    /// Extracts an archive to the target directory.
    /// </summary>
    private static async Task ExtractArchiveAsync(
        string archivePath,
        string targetDirectory,
        string targetFileName,
        CancellationToken cancellationToken)
    {
        var extension = Path.GetExtension(archivePath).ToLowerInvariant();

        if (extension == ".zip")
        {
            await Task.Run(() =>
            {
                using var archive = ZipFile.OpenRead(archivePath);
                foreach (var zipEntry in archive.Entries)
                {
                    cancellationToken.ThrowIfCancellationRequested();

                    // Extract only the target file or files in native directory
                    if (zipEntry.Name.Equals(targetFileName, StringComparison.OrdinalIgnoreCase) ||
                        zipEntry.FullName.Contains("/lib/", StringComparison.OrdinalIgnoreCase) ||
                        zipEntry.FullName.Contains("\\lib\\", StringComparison.OrdinalIgnoreCase))
                    {
                        var destPath = Path.Combine(targetDirectory, zipEntry.Name);
                        zipEntry.ExtractToFile(destPath, overwrite: true);
                    }
                }
            }, cancellationToken);
        }
        else if (extension is ".tgz" or ".gz")
        {
            // For .tar.gz files, use System.Formats.Tar if available (.NET 7+)
            await ExtractTarGzAsync(archivePath, targetDirectory, targetFileName, cancellationToken);
        }
    }

    /// <summary>
    /// Extracts a tar.gz archive.
    /// </summary>
    private static async Task ExtractTarGzAsync(
        string archivePath,
        string targetDirectory,
        string targetFileName,
        CancellationToken cancellationToken)
    {
        await using var fileStream = File.OpenRead(archivePath);
        await using var gzipStream = new GZipStream(fileStream, CompressionMode.Decompress);

        // Use System.Formats.Tar for .NET 7+
        await System.Formats.Tar.TarFile.ExtractToDirectoryAsync(
            gzipStream,
            targetDirectory,
            overwriteFiles: true,
            cancellationToken: cancellationToken);
    }

    /// <summary>
    /// Verifies the SHA256 checksum of a file.
    /// </summary>
    private static async Task<bool> VerifyChecksumAsync(
        string filePath,
        string expectedHash,
        CancellationToken cancellationToken)
    {
        if (string.IsNullOrEmpty(expectedHash) || expectedHash.StartsWith("placeholder", StringComparison.OrdinalIgnoreCase))
            return true;

        await using var stream = File.OpenRead(filePath);
        var hashBytes = await SHA256.HashDataAsync(stream, cancellationToken);
        var actualHash = Convert.ToHexString(hashBytes).ToLowerInvariant();

        return actualHash.Equals(expectedHash.ToLowerInvariant(), StringComparison.Ordinal);
    }

    private static string GetMutexName(RuntimeBinaryEntry entry)
    {
        // Global mutex for cross-process synchronization
        var hash = Convert.ToHexString(SHA256.HashData(
            System.Text.Encoding.UTF8.GetBytes($"{entry.Url}_{entry.Sha256}"))).Substring(0, 16);
        return $"Global\\LMSupply_Download_{hash}";
    }

    private static bool IsArchive(string url)
    {
        var extension = Path.GetExtension(url).ToLowerInvariant();
        return extension is ".zip" or ".tgz" or ".gz" or ".tar";
    }

    private static string GetArchiveExtension(string url)
    {
        if (url.EndsWith(".tar.gz", StringComparison.OrdinalIgnoreCase))
            return ".tar.gz";
        if (url.EndsWith(".tgz", StringComparison.OrdinalIgnoreCase))
            return ".tgz";
        return Path.GetExtension(url);
    }

    private static HttpClient CreateHttpClient(BinaryDownloaderOptions options)
    {
        var handler = new HttpClientHandler
        {
            UseProxy = true,
            Proxy = GetProxy(options)
        };

        if (options.AllowUntrustedCertificates)
        {
            handler.ServerCertificateCustomValidationCallback = (_, _, _, _) => true;
        }

        var client = new HttpClient(handler)
        {
            Timeout = options.Timeout
        };

        client.DefaultRequestHeaders.UserAgent.ParseAdd("LMSupply/1.0");

        return client;
    }

    private static IWebProxy? GetProxy(BinaryDownloaderOptions options)
    {
        // Check for explicit proxy configuration
        if (!string.IsNullOrEmpty(options.ProxyUrl))
        {
            var proxy = new WebProxy(options.ProxyUrl);

            if (!string.IsNullOrEmpty(options.ProxyUsername))
            {
                proxy.Credentials = new NetworkCredential(
                    options.ProxyUsername,
                    options.ProxyPassword,
                    options.ProxyDomain);
            }

            return proxy;
        }

        // Check environment variables
        var httpsProxy = Environment.GetEnvironmentVariable("HTTPS_PROXY")
            ?? Environment.GetEnvironmentVariable("https_proxy");
        var httpProxy = Environment.GetEnvironmentVariable("HTTP_PROXY")
            ?? Environment.GetEnvironmentVariable("http_proxy");

        if (!string.IsNullOrEmpty(httpsProxy))
            return new WebProxy(httpsProxy);
        if (!string.IsNullOrEmpty(httpProxy))
            return new WebProxy(httpProxy);

        // Use system proxy
        return WebRequest.GetSystemWebProxy();
    }

    public void Dispose()
    {
        _httpClient.Dispose();
    }
}

/// <summary>
/// Options for the binary downloader.
/// </summary>
public sealed class BinaryDownloaderOptions
{
    /// <summary>
    /// Maximum number of retry attempts for failed downloads.
    /// </summary>
    public int MaxRetries { get; set; } = 3;

    /// <summary>
    /// Timeout for HTTP requests.
    /// </summary>
    public TimeSpan Timeout { get; set; } = TimeSpan.FromMinutes(10);

    /// <summary>
    /// Timeout for acquiring the download mutex.
    /// </summary>
    public TimeSpan MutexTimeout { get; set; } = TimeSpan.FromMinutes(5);

    /// <summary>
    /// Proxy URL for HTTP requests.
    /// </summary>
    public string? ProxyUrl { get; set; }

    /// <summary>
    /// Proxy username for authentication.
    /// </summary>
    public string? ProxyUsername { get; set; }

    /// <summary>
    /// Proxy password for authentication.
    /// </summary>
    public string? ProxyPassword { get; set; }

    /// <summary>
    /// Proxy domain for NTLM authentication.
    /// </summary>
    public string? ProxyDomain { get; set; }

    /// <summary>
    /// Allow untrusted SSL certificates (for corporate proxies with custom CAs).
    /// </summary>
    public bool AllowUntrustedCertificates { get; set; }
}
