using System.Diagnostics;
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

        // Download all discovered files, preserving directory structure
        var allFiles = discovery.GetAllFiles().ToList();
        var totalFileCount = allFiles.Count;
        var fileIndex = 0;

        foreach (var file in allFiles)
        {
            fileIndex++;

            // Preserve the full relative path structure (e.g., "unet/model.onnx_data")
            var localPath = Path.GetFullPath(Path.Combine(modelDir, file.Replace('/', Path.DirectorySeparatorChar)));

            // Validate against path traversal (e.g., "../../../etc/passwd")
            if (!localPath.StartsWith(modelDir, StringComparison.OrdinalIgnoreCase))
                throw new InvalidOperationException($"Path traversal detected in file path: {file}");

            // Ensure parent directory exists
            var parentDir = Path.GetDirectoryName(localPath);
            if (!string.IsNullOrEmpty(parentDir))
            {
                Directory.CreateDirectory(parentDir);
            }

            if (!File.Exists(localPath) || CacheManager.IsLfsPointerFile(localPath))
            {
                // Wrap progress to include multi-file context
                var wrappedProgress = WrapProgress(progress, fileIndex, totalFileCount);

                // Download using the full file path (includes subfolder)
                await DownloadFileWithRetryAsync(
                    repoId, file, localPath, revision, subfolder: null,
                    wrappedProgress, cancellationToken);
            }
        }

        // After all files downloaded, write manifest
        var manifestFiles = allFiles
            .Select(file =>
            {
                var localPath = Path.GetFullPath(Path.Combine(modelDir, file.Replace('/', Path.DirectorySeparatorChar)));
                return new ManifestFileEntry
                {
                    Path = file,
                    Size = File.Exists(localPath) ? new FileInfo(localPath).Length : 0
                };
            })
            .Where(e => e.Size > 0)
            .ToList();

        var manifest = new DownloadManifest
        {
            RepoId = repoId,
            Revision = revision,
            Files = manifestFiles
        };
        await DownloadManifest.WriteAsync(modelDir, manifest);

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
        var fileList = (files ?? GetDefaultModelFiles()).ToList();
        var totalFileCount = fileList.Count;
        var fileIndex = 0;

        foreach (var file in fileList)
        {
            fileIndex++;
            var localPath = Path.Combine(modelDir, file);
            if (!File.Exists(localPath) || CacheManager.IsLfsPointerFile(localPath))
            {
                var wrappedProgress = WrapProgress(progress, fileIndex, totalFileCount);

                var downloaded = await TryDownloadFileWithFallbackAsync(
                    repoId, file, localPath, revision, subfolder,
                    wrappedProgress, cancellationToken);

                if (!downloaded)
                {
                    var location = string.IsNullOrEmpty(subfolder) ? "root" : $"'{subfolder}/' and root";
                    if (IsCriticalFile(file))
                    {
                        throw new ModelDownloadException(
                            $"Required file '{file}' not found in repository '{repoId}' (searched in {location}).",
                            repoId);
                    }

                    // Non-critical (e.g. tokenizer asset) — log a Trace warning so partial-cache
                    // problems can be diagnosed even when downstream tokenizer construction fails
                    // with a confusing error far away from the actual missing file.
                    Trace.TraceWarning(
                        $"[HuggingFaceDownloader] Optional file '{file}' not found for '{repoId}' " +
                        $"(searched in {location}). Downstream tokenizer/feature extraction may fail.");
                }
            }
        }

        // Write manifest from actually downloaded files (not directory scan)
        var downloadedFiles = fileList
            .Select(file =>
            {
                var filePath = Path.Combine(modelDir, file);
                return new ManifestFileEntry
                {
                    Path = file,
                    Size = File.Exists(filePath) ? new FileInfo(filePath).Length : 0
                };
            })
            .Where(e => e.Size > 0)
            .ToList();

        var downloadedManifest = new DownloadManifest
        {
            RepoId = repoId,
            Revision = revision,
            Files = downloadedFiles
        };
        await DownloadManifest.WriteAsync(modelDir, downloadedManifest);

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
                // Retry: the destination may be transiently held open by another process/AV
                // scanner right after a previous attempt renamed it into place.
                await FileIoRetry.ExecuteAsync(
                    () => File.Move(tempPath, destinationPath, overwrite: true),
                    cancellationToken);
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

        // Check if this is an LFS pointer (small file masquerading as a large binary asset).
        // Applies to ONNX model files AND SentencePiece protobufs (.spm / .bpe.model / .model),
        // both of which are stored in LFS on HuggingFace and would render the model unusable
        // if the resolve endpoint returns a pointer instead of the actual binary.
        var contentLength = response.Content.Headers.ContentLength ?? 0;
        if (contentLength < 1024 && IsLfsBinaryAsset(filename))
        {
            var content = await response.Content.ReadAsStringAsync(cancellationToken);
            if (content.StartsWith("version https://git-lfs.github.com/spec/v1", StringComparison.Ordinal))
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

        // Retry: two callers racing to acquire the same ".part" file (a caller bypassing the
        // model pool's lock, or a genuinely concurrent second process) hit this as IOException.
        await using var fileStream = await FileIoRetry.ExecuteAsync(
            () => new FileStream(tempPath, fileMode, FileAccess.Write, FileShare.None, BufferSize, true),
            cancellationToken);

        var buffer = new byte[BufferSize];
        long bytesDownloaded = startPosition;
        int bytesRead;

        // One report per read is one per 16 KB — 30,000 callbacks for a 470 MB model, each of which
        // a UI-bound Progress<T> posts to its thread. Coalesce here, once, to first/last/1%/250 ms.
        progress = CoalescingProgress.Wrap(progress);

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

        // Move to final location atomically. Retry: the destination path may be transiently
        // held open by another process/AV scanner immediately after this rename.
        fileStream.Close();
        await FileIoRetry.ExecuteAsync(
            () => File.Move(tempPath, destinationPath, overwrite: true),
            cancellationToken);
    }

    /// <summary>
    /// Wraps a progress reporter to include multi-file context (file index and total count).
    /// </summary>
    private static MultiFileProgress? WrapProgress(
        IProgress<DownloadProgress>? progress, int currentFileIndex, int totalFileCount)
    {
        if (progress is null)
            return null;

        return new MultiFileProgress(progress, currentFileIndex, totalFileCount);
    }

    private sealed class MultiFileProgress(
        IProgress<DownloadProgress> inner, int currentFileIndex, int totalFileCount)
        : IProgress<DownloadProgress>
    {
        public void Report(DownloadProgress value)
        {
            inner.Report(value with
            {
                CurrentFileIndex = currentFileIndex,
                TotalFileCount = totalFileCount
            });
        }
    }

    internal static IEnumerable<string> GetDefaultModelFiles()
    {
        return
        [
            "model.onnx",
            // External-weight companions. Models whose ONNX graph stores weights in an
            // external data file (e.g. BAAI/bge-m3 — the 'default'/'quality' embedder
            // alias) ship model.onnx as a small graph shell; without the companion the
            // session crashes at init ("file_size: ... model.onnx_data"). Both naming
            // conventions exist on HF. Repos without one simply skip it (non-critical
            // → Trace warning only). Chunked variants (model.onnx_data_0…) are not
            // covered here — the discovery path (DownloadWithDiscoveryAsync) owns those.
            "model.onnx_data",
            "model.onnx.data",
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
    /// Returns true for files that are stored as Git LFS binary assets on HuggingFace and
    /// must therefore be guarded against pointer-file responses (small text content).
    /// </summary>
    private static bool IsLfsBinaryAsset(string filename)
    {
        return filename.EndsWith(".onnx", StringComparison.OrdinalIgnoreCase) ||
               filename.EndsWith(".onnx_data", StringComparison.OrdinalIgnoreCase) ||
               filename.EndsWith(".spm", StringComparison.OrdinalIgnoreCase) ||
               filename.EndsWith(".bpe.model", StringComparison.OrdinalIgnoreCase) ||
               filename.Equals("sentencepiece.bpe.model", StringComparison.OrdinalIgnoreCase) ||
               filename.Equals("tokenizer.model", StringComparison.OrdinalIgnoreCase);
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
