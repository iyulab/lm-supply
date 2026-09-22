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
    private const int MaxRetries = 3;

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

            if (!IsUsableCachedFile(localPath, expectedSize, repoId)
                && !TryAdoptSiblingCopy(localPath, expectedSize, snapshotDir: modelDir, repoId))
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
            if (!IsUsableCachedFile(localPath, ExpectedOnDisk(file), repoId)
                && !TryAdoptSiblingCopy(localPath, ExpectedOnDisk(file), snapshotDir, repoId))
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

                    // Non-critical (a tokenizer asset another tokenizer family uses, an external data
                    // file a single-file model does not have): its absence is the normal case for most
                    // models, so it is recorded at Information for cache diagnostics. Whether a file is
                    // actually required is the tokenizer factory's call, and it throws when one is.
                    Trace.TraceInformation(
                        $"[HuggingFaceDownloader] Optional file '{file}' not present for '{repoId}' (searched in {location}).");
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
    /// <summary>
    /// Before a file is downloaded to <paramref name="localPath"/>, looks for the same file elsewhere in
    /// the same snapshot — at the root when the target is in a subfolder, in an immediate subfolder when
    /// the target is at the root — and moves it there when it is a real file of the listed length.
    /// </summary>
    /// <remarks>
    /// 0.63.0 moved a subfolder's files from the snapshot root into the subfolder, and the new path
    /// looked in one place only: every existing cache downloaded the model again (about 1 GB for the
    /// default embedder) and kept both copies. A repository loaded once by alias (files under its
    /// subfolder) and once by id (repository layout) produced the same pair the other way round. A
    /// move keeps one copy and costs no request. Nothing is moved in read-only mode, and a copy whose
    /// length differs from the listing is left where it is — it is not the file the repository lists.
    /// </remarks>
    private bool TryAdoptSiblingCopy(string localPath, long? expectedSize, string snapshotDir, string repoId)
    {
        if (_localFilesOnly || File.Exists(localPath))
            return false;

        var fileName = Path.GetFileName(localPath);
        var targetDir = Path.GetDirectoryName(localPath)!;
        var root = Path.TrimEndingDirectorySeparator(Path.GetFullPath(snapshotDir));
        var targetIsRoot = string.Equals(Path.TrimEndingDirectorySeparator(Path.GetFullPath(targetDir)), root, StringComparison.OrdinalIgnoreCase);

        IEnumerable<string> candidates = targetIsRoot
            ? Directory.Exists(root)
                ? Directory.EnumerateDirectories(root).Select(d => Path.Combine(d, fileName))
                : []
            : [Path.Combine(root, fileName)];

        foreach (var candidate in candidates)
        {
            if (!CacheManager.IsCachedFile(candidate))
                continue;
            if (expectedSize is { } expected && new FileInfo(candidate).Length != expected)
                continue;

            try
            {
                Directory.CreateDirectory(targetDir);
                File.Move(candidate, localPath);
                Trace.TraceInformation(
                    $"[HuggingFaceDownloader] Adopted '{candidate}' as '{localPath}' for '{repoId}' instead of downloading it again.");
                return true;
            }
            catch (IOException ex)
            {
                Trace.TraceWarning($"[HuggingFaceDownloader] Could not move '{candidate}' to '{localPath}': {ex.Message}");
                return false;
            }
        }

        return false;
    }

    private bool IsUsableCachedFile(string localPath, long? expectedSize, string repoId)
    {
        if (!CacheManager.IsCachedFile(localPath) || expectedSize is not { } expected)
            return ResumableFileDownload.IsUsableCachedFile(localPath, expectedSize, readOnly: _localFilesOnly);

        var actual = new FileInfo(localPath).Length;
        if (actual == expected)
            return true;

        Trace.TraceWarning(
            $"[HuggingFaceDownloader] '{localPath}' of '{repoId}' is {actual} bytes but the repository lists {expected}; " +
            (_localFilesOnly ? "treating it as not cached." : "discarding it and downloading again."));
        return ResumableFileDownload.IsUsableCachedFile(localPath, expectedSize, readOnly: _localFilesOnly);
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
    /// Downloads a file with retry on transient failures, resuming a body that ended early — the shared
    /// <see cref="ResumableFileDownload"/> rules, with this source's own checks: the resolve URL and the
    /// Git LFS pointer test.
    /// </summary>
    private Task DownloadFileWithRetryAsync(
        string repoId,
        string filename,
        string destinationPath,
        string revision,
        string? subfolder,
        long? expectedSize,
        IProgress<DownloadProgress>? progress,
        CancellationToken cancellationToken)
        => ResumableFileDownload.DownloadAsync(_httpClient, BuildRequest(repoId, filename, destinationPath, revision, subfolder, expectedSize, progress), cancellationToken);

    /// <summary>
    /// Downloads a single file with resume support — one attempt, no retry.
    /// </summary>
    public Task DownloadFileAsync(
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

        // This method always goes to the network; with local files only there is nothing it may do.
        if (_localFilesOnly)
            throw NotCached(repoId, filename, Path.GetDirectoryName(destinationPath) ?? destinationPath);

        return ResumableFileDownload.AttemptAsync(_httpClient, BuildRequest(repoId, filename, destinationPath, revision, subfolder, expectedSize: null, progress), cancellationToken);
    }

    private static ResumableFileDownload.Request BuildRequest(
        string repoId, string filename, string destinationPath, string revision, string? subfolder, long? expectedSize, IProgress<DownloadProgress>? progress)
    {
        // Build URL using resolve endpoint (handles LFS automatically)
        var filePath = string.IsNullOrEmpty(subfolder) ? filename : $"{subfolder}/{filename}";
        return new ResumableFileDownload.Request
        {
            Url = $"{HuggingFaceBaseUrl}/{repoId}/resolve/{revision}/{filePath}",
            DestinationPath = destinationPath,
            FileName = filePath,
            ModelId = repoId,
            ExpectedSize = expectedSize,
            MaxRetries = MaxRetries,
            Progress = progress,
            InspectResponse = async (response, ct) =>
            {
                // Check if this is an LFS pointer (small file masquerading as a large binary asset).
                // Applies to ONNX model files AND SentencePiece protobufs (.spm / .bpe.model / .model),
                // both of which are stored in LFS on HuggingFace and would render the model unusable
                // if the resolve endpoint returns a pointer instead of the actual binary.
                var contentLength = response.Content.Headers.ContentLength ?? 0;
                if (contentLength < 1024 && IsLfsBinaryAsset(filename))
                {
                    var content = await response.Content.ReadAsStringAsync(ct).ConfigureAwait(false);
                    if (content.StartsWith("version https://git-lfs.github.com/spec/v1", StringComparison.Ordinal))
                    {
                        throw new ModelDownloadException(
                            $"Received LFS pointer for '{filename}'. This may indicate a network or redirect issue.",
                            repoId);
                    }
                }
            },
        };
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
            "sentencepiece.bpe.model",
            // sentence-transformers' own settings (max_seq_length). Optional: most repos lack it.
            "sentence_bert_config.json"
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
               filename.Equals("sentence_bert_config.json", StringComparison.OrdinalIgnoreCase) ||
               filename.Equals("sentencepiece.bpe.model", StringComparison.OrdinalIgnoreCase);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _httpClient.Dispose();
        _disposed = true;
    }
}
