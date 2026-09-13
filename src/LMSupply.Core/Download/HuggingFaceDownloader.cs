using System.Collections.Concurrent;
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
    private readonly bool _localFilesOnly;
    // Set only through the test seam, so that discovery goes through the same fake transport and a
    // test can count every request the downloader causes.
    private readonly HttpMessageHandler? _discoveryHandler;
    private bool _disposed;

    private const string HuggingFaceBaseUrl = "https://huggingface.co";
    private const int BufferSize = 81920; // 80KB
    private const int MaxRetries = 3;
    // A resume that keeps making progress may take many attempts on a link that drops every few tens of
    // megabytes; this caps the total so a server that always answers with one short body cannot loop.
    private const int MaxResumeAttempts = 20;
    // One gate per destination path: same-process callers loading the same model wait for one another
    // instead of racing for the ".part" (see DownloadFileWithRetryAsync).
    private static readonly ConcurrentDictionary<string, SemaphoreSlim> s_fileGates = new(StringComparer.OrdinalIgnoreCase);

    /// <summary>
    /// Gets the cache directory being used.
    /// </summary>
    public string CacheDirectory => _cacheDir;

    /// <summary>
    /// Gets whether this downloader only reads the local cache. When true it makes no network request:
    /// a model already in the cache is returned as usual, and one that is not throws
    /// <see cref="ModelNotFoundException"/>.
    /// </summary>
    public bool LocalFilesOnly => _localFilesOnly;

    /// <summary>
    /// Initializes a new HuggingFace downloader.
    /// </summary>
    /// <param name="cacheDir">Custom cache directory, or null to use default HuggingFace cache location.</param>
    public HuggingFaceDownloader(string? cacheDir = null)
        : this(cacheDir, localFilesOnly: false)
    {
    }

    /// <summary>
    /// Initializes a new HuggingFace downloader that may be restricted to the local cache.
    /// </summary>
    /// <param name="cacheDir">Custom cache directory, or null to use default HuggingFace cache location.</param>
    /// <param name="localFilesOnly">
    /// When true, never download — the <c>local_files_only</c> mode of <c>huggingface_hub</c>. Files and
    /// the repository's file list come from the cache alone; anything missing throws
    /// <see cref="ModelNotFoundException"/> without a network request.
    /// </param>
    public HuggingFaceDownloader(string? cacheDir, bool localFilesOnly)
        : this(cacheDir, new HttpClientHandler
        {
            AllowAutoRedirect = true,
            MaxAutomaticRedirections = 10,
            AutomaticDecompression = DecompressionMethods.All
        }, localFilesOnly, discoveryUsesHandler: false)
    {
    }

    /// <summary>Test seam: the same downloader over a caller-supplied transport, discovery included.</summary>
    internal HuggingFaceDownloader(string? cacheDir, HttpMessageHandler handler, bool localFilesOnly = false)
        : this(cacheDir, handler, localFilesOnly, discoveryUsesHandler: true)
    {
    }

    private HuggingFaceDownloader(string? cacheDir, HttpMessageHandler handler, bool localFilesOnly, bool discoveryUsesHandler)
    {
        _cacheDir = cacheDir ?? CacheManager.GetDefaultCacheDirectory();
        _localFilesOnly = localFilesOnly;
        _discoveryHandler = discoveryUsesHandler ? handler : null;

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

        using var discoveryService = CreateDiscoveryService();
        var discovery = await discoveryService.DiscoverModelAsync(repoId, preferences, revision, cancellationToken);

        // With local files only the cache is read, never written: an offline load must work from a
        // read-only cache, and a miss must not leave an empty snapshot directory behind.
        var modelDir = CacheManager.GetModelDirectory(_cacheDir, repoId, revision);
        if (!_localFilesOnly)
            Directory.CreateDirectory(modelDir);

        // Download all discovered files, preserving directory structure
        var allFiles = discovery.GetAllFiles().ToList();
        var totalFileCount = allFiles.Count;
        var fileIndex = 0;
        var manifestFiles = new List<ManifestFileEntry>();

        foreach (var file in allFiles)
        {
            fileIndex++;

            // Preserve the full relative path structure (e.g., "unet/model.onnx_data")
            var localPath = Path.GetFullPath(Path.Combine(modelDir, file.Replace('/', Path.DirectorySeparatorChar)));

            // Validate against path traversal (e.g., "../../../etc/passwd")
            if (!localPath.StartsWith(modelDir, StringComparison.OrdinalIgnoreCase))
                throw new InvalidOperationException($"Path traversal detected in file path: {file}");

            // The listing discovery just fetched says how long every file must be.
            var expectedSize = discovery.FileSizes.TryGetValue(file, out var listed) && listed > 0 ? listed : (long?)null;

            if (!IsUsableCachedFile(localPath, expectedSize, repoId))
            {
                // Every discovered file is part of the model (graph, external weights, config).
                if (_localFilesOnly)
                    throw NotCached(repoId, file, modelDir);

                // Ensure parent directory exists
                var parentDir = Path.GetDirectoryName(localPath);
                if (!string.IsNullOrEmpty(parentDir))
                {
                    Directory.CreateDirectory(parentDir);
                }

                // Wrap progress to include multi-file context
                var wrappedProgress = WrapProgress(progress, fileIndex, totalFileCount);

                // Download using the full file path (includes subfolder)
                await DownloadFileWithRetryAsync(
                    repoId, file, localPath, revision, subfolder: null, expectedSize,
                    wrappedProgress, cancellationToken);
            }

            if (File.Exists(localPath))
                manifestFiles.Add(new ManifestFileEntry { Path = file, Size = expectedSize ?? new FileInfo(localPath).Length });
        }

        if (_localFilesOnly)
            return (modelDir, discovery);

        // The manifest records what the repository listed, not what landed on disk — a length the
        // validator can hold the files to. Without a listing it records the disk and says so (version 1).
        var manifest = new DownloadManifest
        {
            Version = discovery.FileSizes.Count > 0 ? DownloadManifest.VerifiedVersion : 1,
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
    /// <param name="subfolder">
    /// Optional subfolder within the repository (e.g., "onnx"). Its files are stored under the same
    /// subfolder locally. Tokenizer and config files missing from the subfolder are fetched from the
    /// repository root into that same directory, so it holds everything requested.
    /// </param>
    /// <param name="progress">Optional progress reporter.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>
    /// The local directory containing the requested files — the subfolder's own directory when
    /// <paramref name="subfolder"/> is given, otherwise the snapshot root.
    /// </returns>
    public async Task<string> DownloadModelAsync(
        string repoId,
        IEnumerable<string>? files = null,
        string revision = "main",
        string? subfolder = null,
        IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(repoId);

        // A subfolder's files are stored under that subfolder. Repositories keep several models under
        // the same file names (one recognizer per script, one variant per execution provider); writing
        // them all to the snapshot root let the second find the first's file, skip its own download,
        // and run the wrong model.
        var snapshotDir = CacheManager.GetModelDirectory(_cacheDir, repoId, revision);
        var modelDir = CacheManager.GetSubfolderDirectory(snapshotDir, subfolder);

        // With local files only the cache is read, never written (see DownloadWithDiscoveryAsync).
        if (!_localFilesOnly)
            Directory.CreateDirectory(modelDir);

        // Default files if not specified
        var fileList = (files ?? GetDefaultModelFiles()).ToList();
        var totalFileCount = fileList.Count;
        var fileIndex = 0;

        // How long each file must be. A manifest written from the repository listing (version 2) answers
        // for cached files without a request; when a file is missing, or the manifest is absent or
        // predates verification, the listing is fetched once — it is cached by the discovery service, so
        // a warm load still makes no request. Unknown lengths only mean the byte-count check in
        // DownloadFileCoreAsync stands alone.
        var manifest = await DownloadManifest.ReadAsync(modelDir);
        var manifestSizes = manifest is { Version: >= DownloadManifest.VerifiedVersion }
            ? manifest.Files.Where(f => f.Size > 0).ToDictionary(f => f.Path, f => f.Size, StringComparer.Ordinal)
            : new Dictionary<string, long>(StringComparer.Ordinal);
        var needsListing = !_localFilesOnly && fileList.Any(f =>
            !manifestSizes.ContainsKey(f) || !CacheManager.IsCachedFile(Path.Combine(modelDir, f)));
        var listing = needsListing
            ? await TryListRepositoryFileSizesAsync(repoId, revision, cancellationToken)
            : null;

        long? ListedAt(string? location, string file) =>
            listing is not null && listing.TryGetValue(string.IsNullOrEmpty(location) ? file : $"{location}/{file}", out var size)
                ? size
                : null;

        // A cached file came from the subfolder when the repository has it there, else from the root.
        long? ExpectedOnDisk(string file) =>
            manifestSizes.TryGetValue(file, out var recorded) ? recorded : ListedAt(subfolder, file) ?? ListedAt(null, file);

        foreach (var file in fileList)
        {
            fileIndex++;
            var localPath = Path.Combine(modelDir, file);
            if (!IsUsableCachedFile(localPath, ExpectedOnDisk(file), repoId))
            {
                if (_localFilesOnly)
                {
                    if (IsCriticalFile(file))
                        throw NotCached(repoId, file, modelDir);

                    Trace.TraceWarning(
                        $"[HuggingFaceDownloader] Optional file '{file}' for '{repoId}' is not in the local cache " +
                        "and downloads are disabled; skipping it.");
                    continue;
                }

                var wrappedProgress = WrapProgress(progress, fileIndex, totalFileCount);

                var downloaded = await TryDownloadFileWithFallbackAsync(
                    repoId, file, localPath, revision, subfolder,
                    expectedInSubfolder: ListedAt(subfolder, file), expectedInRoot: ListedAt(null, file),
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

        if (_localFilesOnly)
            return modelDir;

        // Write manifest from the requested files that exist, at the length the repository listed
        // (or the manifest already held). When neither was available the entry records the disk and the
        // manifest stays version 1, so the next load verifies against the listing.
        var verified = listing is not null || manifestSizes.Count > 0;
        var downloadedFiles = fileList
            .Select(file =>
            {
                var filePath = Path.Combine(modelDir, file);
                return new ManifestFileEntry
                {
                    Path = file,
                    Size = File.Exists(filePath) ? ExpectedOnDisk(file) ?? new FileInfo(filePath).Length : 0
                };
            })
            .Where(e => e.Size > 0)
            .ToList();

        var downloadedManifest = new DownloadManifest
        {
            Version = verified ? DownloadManifest.VerifiedVersion : 1,
            RepoId = repoId,
            Revision = revision,
            Files = downloadedFiles
        };
        await DownloadManifest.WriteAsync(modelDir, downloadedManifest);

        return modelDir;
    }

    /// <summary>
    /// A cached file is usable when it holds real content and — when the repository listing or a verified
    /// manifest says how long it must be — is exactly that long. A final file of any other length has no
    /// known provenance (an interrupted copy, another revision, a rename that clobbered a complete file),
    /// so it is not a prefix to resume from: only a ".part" is. It is deleted so the download starts over.
    /// With local files only nothing is deleted; the file is simply not cached.
    /// </summary>
    private bool IsUsableCachedFile(string localPath, long? expectedSize, string repoId)
    {
        if (!CacheManager.IsCachedFile(localPath))
            return false;
        if (expectedSize is not { } expected)
            return true;

        var actual = new FileInfo(localPath).Length;
        if (actual == expected)
            return true;

        Trace.TraceWarning(
            $"[HuggingFaceDownloader] '{localPath}' of '{repoId}' is {actual} bytes but the repository lists {expected}; " +
            (_localFilesOnly ? "treating it as not cached." : "discarding it and downloading again."));
        if (!_localFilesOnly)
            File.Delete(localPath);
        return false;
    }

    /// <summary>
    /// The byte length of every file the repository lists, keyed by repository path. Empty when the
    /// listing cannot be fetched — the download then proceeds as before, checked against the server's
    /// own content length only.
    /// </summary>
    private async Task<IReadOnlyDictionary<string, long>> TryListRepositoryFileSizesAsync(
        string repoId, string revision, CancellationToken cancellationToken)
    {
        try
        {
            using var discoveryService = CreateDiscoveryService();
            var files = await discoveryService.ListRepositoryFilesAsync(repoId, revision, cancellationToken);
            return files
                .Where(f => f.IsFile && f.Size > 0)
                .GroupBy(f => f.Path, StringComparer.Ordinal)
                .ToDictionary(g => g.Key, g => g.First().Size, StringComparer.Ordinal);
        }
        catch (Exception ex) when (ex is HttpRequestException or ModelNotFoundException or IOException
                                      or InvalidOperationException or UnauthorizedAccessException)
        {
            Trace.TraceWarning(
                $"[HuggingFaceDownloader] Could not list '{repoId}' ({ex.GetType().Name}: {ex.Message}); " +
                "file lengths are unknown for this download.");
            return new Dictionary<string, long>(StringComparer.Ordinal);
        }
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
        long? expectedInSubfolder,
        long? expectedInRoot,
        IProgress<DownloadProgress>? progress,
        CancellationToken cancellationToken)
    {
        // First, try downloading from the specified location (subfolder or root)
        try
        {
            await DownloadFileWithRetryAsync(
                repoId, filename, localPath, revision, subfolder,
                string.IsNullOrEmpty(subfolder) ? expectedInRoot : expectedInSubfolder,
                progress, cancellationToken);
            return true;
        }
        catch (HttpRequestException ex) when (ex.StatusCode == HttpStatusCode.NotFound)
        {
            // If subfolder is specified and this is a tokenizer/config file, try root
            if (!string.IsNullOrEmpty(subfolder) && IsTokenizerOrConfigFile(filename))
            {
                try
                {
                    await DownloadFileWithRetryAsync(
                        repoId, filename, localPath, revision, subfolder: null, expectedInRoot,
                        progress, cancellationToken);
                    return true;
                }
                catch (HttpRequestException rootEx) when (rootEx.StatusCode == HttpStatusCode.NotFound)
                {
                    // Not found in root either
                    return false;
                }
            }

            // Not found and no fallback applicable
            return false;
        }
    }

    /// <summary>
    /// Downloads a file with automatic retry on transient failures, and resumes a body that ended early.
    /// Callers in the same process that want the same file wait here for one another; the one that
    /// arrives second finds the file complete and returns.
    /// </summary>
    private async Task DownloadFileWithRetryAsync(
        string repoId,
        string filename,
        string destinationPath,
        string revision,
        string? subfolder,
        long? expectedSize,
        IProgress<DownloadProgress>? progress,
        CancellationToken cancellationToken)
    {
        var gate = s_fileGates.GetOrAdd(Path.GetFullPath(destinationPath), _ => new SemaphoreSlim(1, 1));
        await gate.WaitAsync(cancellationToken);
        try
        {
            if (IsCompleteFile(destinationPath, expectedSize))
                return;

            var tempPath = destinationPath + ".part";
            var lastLength = PartLength(tempPath);
            var stalls = 0;

            for (var attempt = 1; ; attempt++)
            {
                try
                {
                    await DownloadFileCoreAsync(repoId, filename, destinationPath, revision, subfolder, expectedSize, progress, cancellationToken);
                    return;
                }
                catch (HttpRequestException ex) when (IsTransientError(ex) && attempt < MaxRetries)
                {
                    // Exponential backoff below.
                }
                catch (TaskCanceledException) when (!cancellationToken.IsCancellationRequested && attempt < MaxRetries)
                {
                    // Timeout, not user cancellation.
                }
                catch (Exception ex) when (ex is IOException or TruncatedDownloadException)
                {
                    // A body that ended early — EOF before the announced length, or a dropped connection —
                    // leaves a valid prefix in the ".part", and so does another process still writing it.
                    // Resuming that prefix is progress, so the budget counts stalls rather than attempts: an
                    // attempt after which the ".part" is no longer than before is a stall, and MaxRetries
                    // stalls in a row (or MaxResumeAttempts attempts overall) give up.
                    var length = PartLength(tempPath);
                    stalls = length > lastLength ? 0 : stalls + 1;
                    lastLength = length;
                    if (stalls >= MaxRetries || attempt >= MaxResumeAttempts)
                        throw;
                }

                var delay = TimeSpan.FromSeconds(Math.Pow(2, Math.Min(attempt, 4)));
                await Task.Delay(delay, cancellationToken);
            }
        }
        finally
        {
            gate.Release();
        }
    }

    private static bool IsCompleteFile(string path, long? expectedSize) =>
        expectedSize is { } expected
        && File.Exists(path)
        && new FileInfo(path).Length == expected
        && !CacheManager.IsLfsPointerFile(path);

    private static long PartLength(string tempPath) =>
        File.Exists(tempPath) ? new FileInfo(tempPath).Length : 0;

    /// <summary>
    /// Downloads a single file with resume support.
    /// </summary>
    public Task DownloadFileAsync(
        string repoId,
        string filename,
        string destinationPath,
        string revision = "main",
        string? subfolder = null,
        IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
        => DownloadFileCoreAsync(repoId, filename, destinationPath, revision, subfolder, expectedSize: null, progress, cancellationToken);

    /// <summary>
    /// One attempt at one file. Resumes from the ".part" it owns, and treats the result as the file only
    /// when it is as long as the server announced (and as the repository listed, when known): a body that
    /// ends early leaves the ".part" for the next attempt and throws <see cref="TruncatedDownloadException"/>.
    /// </summary>
    private async Task DownloadFileCoreAsync(
        string repoId,
        string filename,
        string destinationPath,
        string revision,
        string? subfolder,
        long? expectedSize,
        IProgress<DownloadProgress>? progress,
        CancellationToken cancellationToken)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(repoId);
        ArgumentException.ThrowIfNullOrWhiteSpace(filename);
        ArgumentException.ThrowIfNullOrWhiteSpace(destinationPath);

        // This method always goes to the network; with local files only there is nothing it may do.
        if (_localFilesOnly)
            throw NotCached(repoId, filename, Path.GetDirectoryName(destinationPath) ?? destinationPath);

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

        // Own the ".part" before measuring it: its length is the resume offset, and a caller that measured
        // it while another was still writing would ask the server for a range it does not hold. With
        // FileShare.None the second caller waits in FileIoRetry (two callers racing to acquire the same
        // ".part" — a caller bypassing the model pool's lock, or a genuinely concurrent second process —
        // hit this as IOException) and, once it gets in, finds the finished file or an honest offset.
        await using var fileStream = await FileIoRetry.ExecuteAsync(
            () => new FileStream(tempPath, FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.None, BufferSize, true),
            cancellationToken);

        try
        {
            await DownloadIntoPartAsync(fileStream, tempPath, repoId, filename, filePath, url, destinationPath, revision, expectedSize, progress, cancellationToken);
        }
        catch
        {
            // Nothing landed (a 404, a refused request, a body that never started): an empty ".part" is
            // not a resume point, and the directory validator reads any ".part" as an unfinished download.
            if (fileStream.Length == 0)
            {
                fileStream.Close();
                File.Delete(tempPath);
            }
            throw;
        }
    }

    private async Task DownloadIntoPartAsync(
        FileStream fileStream,
        string tempPath,
        string repoId,
        string filename,
        string filePath,
        string url,
        string destinationPath,
        string revision,
        long? expectedSize,
        IProgress<DownloadProgress>? progress,
        CancellationToken cancellationToken)
    {
        // Whoever held the ".part" before us may have finished the download we came for.
        if (IsCompleteFile(destinationPath, expectedSize))
        {
            fileStream.Close();
            File.Delete(tempPath);
            return;
        }

        var startPosition = fileStream.Length;
        if (expectedSize is { } listedLength && startPosition >= listedLength)
        {
            // A ".part" at least as long as the file cannot be a prefix of it.
            fileStream.SetLength(0);
            startPosition = 0;
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

        // 416 (Range Not Satisfiable): the server has nothing past our offset. That is completion only
        // when the listing says we hold the whole file; otherwise the ".part" is not a prefix of this
        // file, and the next attempt starts over.
        if (response.StatusCode == HttpStatusCode.RequestedRangeNotSatisfiable)
        {
            if (expectedSize is { } whole && startPosition == whole)
            {
                fileStream.Close();
                await MoveIntoPlaceAsync(tempPath, destinationPath, cancellationToken);
                return;
            }

            fileStream.SetLength(0);
            throw new TruncatedDownloadException(
                $"Server has no bytes past offset {startPosition} of '{filePath}' in '{repoId}' (HTTP 416), " +
                "but the file is not known to be complete; restarting the download.",
                repoId);
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
            if (contentRange?.From is { } from && from != startPosition)
            {
                // The server resumed somewhere else; appending its bytes after ours would interleave two
                // offsets into one file.
                fileStream.SetLength(0);
                throw new TruncatedDownloadException(
                    $"Server resumed '{filePath}' in '{repoId}' at offset {from} while {startPosition} bytes were held; restarting the download.",
                    repoId);
            }

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
            // Full body (the server ignored the range, or none was sent): write from the start.
            fileStream.SetLength(0);
            startPosition = 0;
        }

        // The length the file must reach: what the server announced, else what the repository listed.
        var expectedTotal = totalBytes > 0 ? totalBytes : expectedSize ?? 0;

        // Download with progress
        await using var contentStream = await response.Content.ReadAsStreamAsync(cancellationToken);
        fileStream.Seek(0, SeekOrigin.End);

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
                TotalBytes = expectedTotal
            });
        }

        await fileStream.FlushAsync(cancellationToken);
        var length = fileStream.Length;

        // The body ended (end of stream, not an exception) before the announced length. The ".part" is a
        // valid prefix — keep it for the resume — but it is not the file.
        if (expectedTotal > 0 && length != expectedTotal)
        {
            throw new TruncatedDownloadException(
                $"Download of '{filePath}' from '{repoId}' ended after {length} of {expectedTotal} bytes " +
                $"(HTTP {(int)response.StatusCode}, content-length {response.Content.Headers.ContentLength?.ToString(System.Globalization.CultureInfo.InvariantCulture) ?? "absent"}, " +
                $"content-range {response.Content.Headers.ContentRange?.ToString() ?? "absent"}, resumed from {startPosition}); " +
                "the partial file is kept for a resume.",
                repoId);
        }

        // The server delivered everything it announced, but that is not the file the repository listed —
        // a stale listing or a revision that moved between the two requests. Not a prefix of anything.
        if (expectedSize is { } expected && length != expected)
        {
            fileStream.Close();
            File.Delete(tempPath);
            throw new ModelDownloadException(
                $"'{filePath}' from '{repoId}' is {length} bytes but the repository listing says {expected} " +
                $"(revision '{revision}'); the listing may be stale.",
                repoId);
        }

        fileStream.Close();
        await MoveIntoPlaceAsync(tempPath, destinationPath, cancellationToken);
    }

    // Move to final location atomically. Retry: the destination path may be transiently held open by
    // another process/AV scanner immediately after this rename.
    private static Task MoveIntoPlaceAsync(string tempPath, string destinationPath, CancellationToken cancellationToken) =>
        FileIoRetry.ExecuteAsync(
            () => File.Move(tempPath, destinationPath, overwrite: true),
            cancellationToken);

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

    private ModelDiscoveryService CreateDiscoveryService() =>
        _discoveryHandler is null
            ? new ModelDiscoveryService(_cacheDir) { LocalFilesOnly = _localFilesOnly }
            : new ModelDiscoveryService(_cacheDir, hfToken: null, _discoveryHandler) { LocalFilesOnly = _localFilesOnly };

    private static ModelNotFoundException NotCached(string repoId, string file, string directory) =>
        new($"'{file}' of model '{repoId}' is not in the local cache ({directory}) and downloads are disabled.", repoId);

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
