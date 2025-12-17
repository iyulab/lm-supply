using System.Net;
using System.Net.Http.Headers;
using LMSupply.Core.Download;
using LMSupply.Exceptions;

namespace LMSupply.Download;

/// <summary>
/// Downloads models from HuggingFace Hub with resume support and HuggingFace-compatible caching.
/// </summary>
public sealed class HuggingFaceDownloader : IDisposable
{
    private readonly HttpClient _httpClient;
    private readonly string _cacheDir;
    private bool _disposed;

    private const string HuggingFaceBaseUrl = "https://huggingface.co";
    private const int BufferSize = 81920; // 80KB
    private const int MaxRetries = 3;

    /// <summary>
    /// Gets the cache directory being used.
    /// </summary>
    public string CacheDirectory => _cacheDir;

    /// <summary>
    /// Initializes a new HuggingFace downloader.
    /// </summary>
    /// <param name="cacheDir">Custom cache directory, or null to use default HuggingFace cache location.</param>
    public HuggingFaceDownloader(string? cacheDir = null)
    {
        _cacheDir = cacheDir ?? CacheManager.GetDefaultCacheDirectory();

        var handler = new HttpClientHandler
        {
            AllowAutoRedirect = true,
            MaxAutomaticRedirections = 10,
            AutomaticDecompression = DecompressionMethods.All
        };

        _httpClient = new HttpClient(handler)
        {
            Timeout = TimeSpan.FromMinutes(30)
        };

        _httpClient.DefaultRequestHeaders.UserAgent.Add(
            new ProductInfoHeaderValue("LMSupply", "1.0"));
    }

    /// <summary>
    /// Downloads a model from HuggingFace using automatic file discovery.
    /// This eliminates the need to specify subfolder or file list manually.
    /// </summary>
    /// <param name="repoId">The HuggingFace repository ID (e.g., "microsoft/Phi-3-mini-4k-instruct-onnx").</param>
    /// <param name="preferences">Optional preferences for model selection (quantization, device, etc.).</param>
    /// <param name="revision">The revision/branch (default: "main").</param>
    /// <param name="progress">Optional progress reporter.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Discovery result with local directory path and file information.</returns>
    public async Task<(string LocalPath, ModelDiscoveryResult Discovery)> DownloadWithDiscoveryAsync(
        string repoId,
        ModelPreferences? preferences = null,
        string revision = "main",
        IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(repoId);

        using var discoveryService = new ModelDiscoveryService(_cacheDir);
        var discovery = await discoveryService.DiscoverModelAsync(repoId, preferences, revision, cancellationToken);

        var modelDir = CacheManager.GetModelDirectory(_cacheDir, repoId, revision);
        Directory.CreateDirectory(modelDir);

        // Download all discovered files
        foreach (var file in discovery.GetAllFiles())
        {
            var fileName = Path.GetFileName(file);
            var localPath = Path.Combine(modelDir, fileName);

            if (!File.Exists(localPath) || CacheManager.IsLfsPointerFile(localPath))
            {
                // Determine subfolder from file path
                var fileSubfolder = file.Contains('/') ? file[..file.LastIndexOf('/')] : null;

                await DownloadFileWithRetryAsync(
                    repoId, fileName, localPath, revision, fileSubfolder,
                    progress, cancellationToken);
            }
        }

        return (modelDir, discovery);
    }

    /// <summary>
    /// Downloads a model from HuggingFace and returns the local directory path.
    /// </summary>
    /// <param name="repoId">The HuggingFace repository ID (e.g., "sentence-transformers/all-MiniLM-L6-v2").</param>
    /// <param name="files">List of files to download. If null, downloads common model files.</param>
    /// <param name="revision">The revision/branch (default: "main").</param>
    /// <param name="subfolder">Optional subfolder within the repository (e.g., "onnx").</param>
    /// <param name="progress">Optional progress reporter.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>The local directory path containing the downloaded model files.</returns>
    public async Task<string> DownloadModelAsync(
        string repoId,
        IEnumerable<string>? files = null,
        string revision = "main",
        string? subfolder = null,
        IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(repoId);

        var modelDir = CacheManager.GetModelDirectory(_cacheDir, repoId, revision);
        Directory.CreateDirectory(modelDir);

        // Default files if not specified
        files ??= GetDefaultModelFiles();

        foreach (var file in files)
        {
            var localPath = Path.Combine(modelDir, file);
            if (!File.Exists(localPath) || CacheManager.IsLfsPointerFile(localPath))
            {
                var downloaded = await TryDownloadFileWithFallbackAsync(
                    repoId, file, localPath, revision, subfolder,
                    progress, cancellationToken);

                if (!downloaded && IsCriticalFile(file))
                {
                    var location = string.IsNullOrEmpty(subfolder) ? "root" : $"'{subfolder}/' and root";
                    throw new ModelDownloadException(
                        $"Required file '{file}' not found in repository '{repoId}' (searched in {location}).",
                        repoId);
                }
            }
        }

        return modelDir;
    }

    /// <summary>
    /// Attempts to download a file, with fallback to root directory for tokenizer files.
    /// </summary>
    /// <returns>True if the file was downloaded successfully, false if not found.</returns>
    private async Task<bool> TryDownloadFileWithFallbackAsync(
        string repoId,
        string filename,
        string localPath,
        string revision,
        string? subfolder,
        IProgress<DownloadProgress>? progress,
        CancellationToken cancellationToken)
    {
        // First, try downloading from the specified location (subfolder or root)
        try
        {
            await DownloadFileWithRetryAsync(repoId, filename, localPath, revision, subfolder, progress, cancellationToken);
            return true;
        }
        catch (HttpRequestException ex) when (ex.StatusCode == HttpStatusCode.NotFound)
        {
            // If subfolder is specified and this is a tokenizer/config file, try root
            if (!string.IsNullOrEmpty(subfolder) && IsTokenizerOrConfigFile(filename))
            {
                try
                {
                    await DownloadFileWithRetryAsync(repoId, filename, localPath, revision, subfolder: null, progress, cancellationToken);
                    return true;
                }
                catch (HttpRequestException rootEx) when (rootEx.StatusCode == HttpStatusCode.NotFound)
                {
                    // Not found in root either
                    return false;
                }
            }

            return false;
        }
    }

    /// <summary>
    /// Downloads a file with automatic retry on transient failures.
    /// </summary>
    private async Task DownloadFileWithRetryAsync(
        string repoId,
        string filename,
        string destinationPath,
        string revision,
        string? subfolder,
        IProgress<DownloadProgress>? progress,
        CancellationToken cancellationToken)
    {
        Exception? lastException = null;

        for (var attempt = 1; attempt <= MaxRetries; attempt++)
        {
            try
            {
                await DownloadFileAsync(repoId, filename, destinationPath, revision, subfolder, progress, cancellationToken);
                return;
            }
            catch (HttpRequestException ex) when (IsTransientError(ex) && attempt < MaxRetries)
            {
                lastException = ex;
                var delay = TimeSpan.FromSeconds(Math.Pow(2, attempt)); // Exponential backoff
                await Task.Delay(delay, cancellationToken);
            }
            catch (TaskCanceledException ex) when (!cancellationToken.IsCancellationRequested && attempt < MaxRetries)
            {
                // Timeout, not user cancellation
                lastException = ex;
                var delay = TimeSpan.FromSeconds(Math.Pow(2, attempt));
                await Task.Delay(delay, cancellationToken);
            }
        }

        // If we get here, all retries failed
        throw lastException ?? new InvalidOperationException("Download failed after retries");
    }

    /// <summary>
    /// Downloads a single file with resume support.
    /// </summary>
    public async Task DownloadFileAsync(
        string repoId,
        string filename,
        string destinationPath,
        string revision = "main",
        string? subfolder = null,
        IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(repoId);
        ArgumentException.ThrowIfNullOrWhiteSpace(filename);
        ArgumentException.ThrowIfNullOrWhiteSpace(destinationPath);

        // Ensure directory exists
        var dir = Path.GetDirectoryName(destinationPath);
        if (!string.IsNullOrEmpty(dir))
        {
            Directory.CreateDirectory(dir);
        }

        // Build URL using resolve endpoint (handles LFS automatically)
        var filePath = string.IsNullOrEmpty(subfolder) ? filename : $"{subfolder}/{filename}";
        var url = $"{HuggingFaceBaseUrl}/{repoId}/resolve/{revision}/{filePath}";

        var tempPath = destinationPath + ".part";
        long startPosition = 0;

        // Check for partial download
        if (File.Exists(tempPath))
        {
            startPosition = new FileInfo(tempPath).Length;
        }

        // Create request with optional range header for resume
        var request = new HttpRequestMessage(HttpMethod.Get, url);
        if (startPosition > 0)
        {
            request.Headers.Range = new RangeHeaderValue(startPosition, null);
        }

        using var response = await _httpClient.SendAsync(
            request,
            HttpCompletionOption.ResponseHeadersRead,
            cancellationToken);

        // Handle 416 (Range Not Satisfiable) - file already complete
        if (response.StatusCode == HttpStatusCode.RequestedRangeNotSatisfiable)
        {
            if (File.Exists(tempPath))
            {
                File.Move(tempPath, destinationPath, overwrite: true);
            }
            return;
        }

        if (!response.IsSuccessStatusCode)
        {
            throw new HttpRequestException(
                $"Failed to download '{filename}' from '{repoId}'. Status: {response.StatusCode}",
                inner: null,
                statusCode: response.StatusCode);
        }

        // Check if this is an LFS pointer (small file for large model)
        var contentLength = response.Content.Headers.ContentLength ?? 0;
        if (contentLength < 1024 && filename.EndsWith(".onnx", StringComparison.OrdinalIgnoreCase))
        {
            var content = await response.Content.ReadAsStringAsync(cancellationToken);
            if (content.StartsWith("version https://git-lfs.github.com/spec/v1"))
            {
                throw new ModelDownloadException(
                    $"Received LFS pointer for '{filename}'. This may indicate a network or redirect issue.",
                    repoId);
            }
        }

        // Determine total size
        long totalBytes = response.Content.Headers.ContentLength ?? 0;
        if (response.StatusCode == HttpStatusCode.PartialContent)
        {
            var contentRange = response.Content.Headers.ContentRange;
            if (contentRange?.Length.HasValue == true)
            {
                totalBytes = contentRange.Length.Value;
            }
            else
            {
                totalBytes = startPosition + (response.Content.Headers.ContentLength ?? 0);
            }
        }
        else
        {
            // Full download, reset position
            startPosition = 0;
        }

        // Download with progress
        await using var contentStream = await response.Content.ReadAsStreamAsync(cancellationToken);
        var fileMode = startPosition > 0 ? FileMode.Append : FileMode.Create;
        await using var fileStream = new FileStream(tempPath, fileMode, FileAccess.Write, FileShare.None, BufferSize, true);

        var buffer = new byte[BufferSize];
        long bytesDownloaded = startPosition;
        int bytesRead;

        while ((bytesRead = await contentStream.ReadAsync(buffer, cancellationToken)) > 0)
        {
            await fileStream.WriteAsync(buffer.AsMemory(0, bytesRead), cancellationToken);
            bytesDownloaded += bytesRead;

            progress?.Report(new DownloadProgress
            {
                FileName = filename,
                BytesDownloaded = bytesDownloaded,
                TotalBytes = totalBytes
            });
        }

        // Move to final location atomically
        fileStream.Close();
        File.Move(tempPath, destinationPath, overwrite: true);
    }

    private static IEnumerable<string> GetDefaultModelFiles()
    {
        return
        [
            "model.onnx",
            "config.json",
            "vocab.txt",
            "vocab.json",
            "merges.txt",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "sentencepiece.bpe.model"
        ];
    }

    private static bool IsCriticalFile(string filename)
    {
        // ONNX model files are critical
        return filename.EndsWith(".onnx", StringComparison.OrdinalIgnoreCase);
    }

    /// <summary>
    /// Checks if a file is a tokenizer or config file that may be located in the root
    /// even when the model files are in a subfolder.
    /// </summary>
    private static bool IsTokenizerOrConfigFile(string filename)
    {
        return filename.Equals("vocab.txt", StringComparison.OrdinalIgnoreCase) ||
               filename.Equals("vocab.json", StringComparison.OrdinalIgnoreCase) ||
               filename.Equals("merges.txt", StringComparison.OrdinalIgnoreCase) ||
               filename.Equals("tokenizer.json", StringComparison.OrdinalIgnoreCase) ||
               filename.Equals("tokenizer_config.json", StringComparison.OrdinalIgnoreCase) ||
               filename.Equals("special_tokens_map.json", StringComparison.OrdinalIgnoreCase) ||
               filename.Equals("config.json", StringComparison.OrdinalIgnoreCase) ||
               filename.Equals("sentencepiece.bpe.model", StringComparison.OrdinalIgnoreCase);
    }

    /// <summary>
    /// Determines if an HTTP error is transient and should be retried.
    /// </summary>
    private static bool IsTransientError(HttpRequestException ex)
    {
        return ex.StatusCode is
            HttpStatusCode.RequestTimeout or
            HttpStatusCode.TooManyRequests or
            HttpStatusCode.InternalServerError or
            HttpStatusCode.BadGateway or
            HttpStatusCode.ServiceUnavailable or
            HttpStatusCode.GatewayTimeout;
    }

    public void Dispose()
    {
        if (_disposed) return;
        _httpClient.Dispose();
        _disposed = true;
    }
}
