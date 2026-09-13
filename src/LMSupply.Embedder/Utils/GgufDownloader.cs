using LMSupply.Core.Download;
using LMSupply.Download;
using LMSupply.Exceptions;
using LMSupply.Hardware;

namespace LMSupply.Embedder.Utils;

/// <summary>
/// Downloads GGUF embedding model files from HuggingFace.
/// </summary>
internal sealed class GgufDownloader : IDisposable
{
    private readonly HttpClient _httpClient;
    private readonly ModelDiscoveryService _discoveryService;
    private readonly string _cacheDirectory;
    private readonly bool _localFilesOnly;
    private bool _disposed;

    private const string HuggingFaceFileBase = "https://huggingface.co";

    /// <param name="cacheDirectory">Where downloaded GGUF files are kept.</param>
    /// <param name="localFilesOnly">
    /// True: serve from the cache only — no repository listing, no download; a repository with no cached
    /// GGUF file fails with <see cref="ModelNotFoundException"/>. The <c>DisableAutoDownload</c> option maps here.
    /// </param>
    public GgufDownloader(string cacheDirectory, bool localFilesOnly = false)
        : this(cacheDirectory, localFilesOnly, handler: null)
    {
    }

    /// <summary>Test seam: the same downloader over a caller-supplied transport, listing included.</summary>
    internal GgufDownloader(string cacheDirectory, bool localFilesOnly, HttpMessageHandler? handler)
    {
        _cacheDirectory = cacheDirectory;
        _localFilesOnly = localFilesOnly;
        _httpClient = handler is null
            ? new HttpClient { Timeout = TimeSpan.FromMinutes(30) }
            : new HttpClient(handler) { Timeout = TimeSpan.FromMinutes(30) };
        _httpClient.DefaultRequestHeaders.Add("User-Agent", "LMSupply/1.0");
        _discoveryService = handler is null
            ? new ModelDiscoveryService(cacheDirectory)
            : new ModelDiscoveryService(cacheDirectory, hfToken: null, handler);
    }

    /// <summary>
    /// Downloads a GGUF embedding model file from HuggingFace.
    /// </summary>
    public async Task<string> DownloadAsync(
        string repoId,
        string? preferredQuantization = null,
        IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        // Offline: the cache is the only source. Read it, never list the repository, never write.
        if (_localFilesOnly)
        {
            var cached = TrySelectFromLocalCache(repoId, preferredQuantization)
                ?? throw new ModelNotFoundException(
                    $"No GGUF file of model '{repoId}' is in the local cache ({GetCacheDirectory(repoId)}) and downloads are disabled.",
                    repoId);

            progress?.Report(new DownloadProgress
            {
                FileName = Path.GetFileName(cached),
                BytesDownloaded = 1,
                TotalBytes = 1
            });
            return cached;
        }

        // List files in the repository
        var files = await ListRepoFilesAsync(repoId, cancellationToken);
        var ggufFiles = files.Where(f => f.IsFile && f.Path.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase)).ToList();

        if (ggufFiles.Count == 0)
        {
            throw new ModelNotFoundException(
                $"No GGUF files found in repository '{repoId}'.",
                repoId);
        }

        // Select best file based on quantization preference
        var selectedFile = SelectBestFile(ggufFiles, preferredQuantization);

        // Check cache: a cached file counts only at the length the repository lists; one of another
        // length is not this file and is fetched again.
        var cachePath = GetCachePath(repoId, selectedFile.Path);
        if (ResumableFileDownload.IsUsableCachedFile(cachePath, selectedFile.Size > 0 ? selectedFile.Size : null))
        {
            progress?.Report(new DownloadProgress
            {
                FileName = selectedFile.Path,
                BytesDownloaded = 1,
                TotalBytes = 1
            });
            return cachePath;
        }

        // Download the file
        var downloadUrl = $"{HuggingFaceFileBase}/{repoId}/resolve/main/{selectedFile.Path}";

        progress?.Report(new DownloadProgress
        {
            FileName = selectedFile.Path,
            BytesDownloaded = 0,
            TotalBytes = selectedFile.Size
        });

        Directory.CreateDirectory(Path.GetDirectoryName(cachePath)!);
        await DownloadFileAsync(downloadUrl, cachePath, selectedFile.Path, selectedFile.Size, progress, cancellationToken);

        return cachePath;
    }

    private Task<IReadOnlyList<RepoFile>> ListRepoFilesAsync(string repoId, CancellationToken cancellationToken)
    {
        return _discoveryService.ListRepositoryFilesAsync(repoId, "main", cancellationToken);
    }

    private static RepoFile SelectBestFile(IReadOnlyList<RepoFile> files, string? preferredQuantization)
    {
        var rawFiles = files.Select(f => new GgufRawFile(Path.GetFileName(f.Path), f.Size));
        var groups = GgufFileGroup.GroupFiles(rawFiles).ToList();

        // Embedder only supports single-file download — exclude split groups
        var nonSplitGroups = groups.Where(g => !g.IsSplit).ToList();

        if (nonSplitGroups.Count > 0)
        {
            var memory = GgufFileSelector.FromHardwareProfile(HardwareProfile.Current);

            try
            {
                var selected = GgufFileSelector.Select(nonSplitGroups, memory, preferredQuantization);
                return files.First(f =>
                    Path.GetFileName(f.Path).Equals(selected.PrimaryFileName,
                        StringComparison.OrdinalIgnoreCase));
            }
            catch (InvalidOperationException)
            {
                // Nothing fits memory: fall back to smallest non-split file
                var smallestGroup = nonSplitGroups.MinBy(g => g.TotalSizeBytes)!;
                return files.First(f =>
                    Path.GetFileName(f.Path).Equals(smallestGroup.PrimaryFileName,
                        StringComparison.OrdinalIgnoreCase));
            }
        }

        // Split-only repo: single-file embedder cannot use split GGUF files.
        throw new InvalidOperationException(
            "No single-file GGUF model found in repository. " +
            "The embedder does not support split GGUF files (e.g., -00001-of-00003.gguf). " +
            "Please use a repository that provides a single-file GGUF model.");
    }

    private string GetCacheDirectory(string repoId) => Path.GetDirectoryName(GetCachePath(repoId, "model.gguf"))!;

    /// <summary>
    /// The cached GGUF file an offline load opens: the one matching the preferred quantization when
    /// there is one, otherwise the first by name. Null when nothing of the repository is cached.
    /// </summary>
    private string? TrySelectFromLocalCache(string repoId, string? preferredQuantization)
    {
        var dir = GetCacheDirectory(repoId);
        if (!Directory.Exists(dir))
            return null;

        var files = Directory.EnumerateFiles(dir, "*.gguf", SearchOption.AllDirectories)
            .OrderBy(f => f, StringComparer.Ordinal)
            .ToList();
        if (files.Count == 0)
            return null;

        if (!string.IsNullOrEmpty(preferredQuantization))
        {
            var preferred = files.FirstOrDefault(f =>
                Path.GetFileName(f).Contains(preferredQuantization, StringComparison.OrdinalIgnoreCase));
            if (preferred != null)
                return preferred;
        }

        return files[0];
    }

    private string GetCachePath(string repoId, string filename)
    {
        var safeRepoId = repoId.Replace('/', '_').Replace('\\', '_');
        return Path.Combine(_cacheDirectory, "gguf-embeddings", safeRepoId, filename);
    }

    private Task DownloadFileAsync(
        string url,
        string destinationPath,
        string fileName,
        long totalBytes,
        IProgress<DownloadProgress>? progress,
        CancellationToken cancellationToken)
        => ResumableFileDownload.DownloadAsync(_httpClient, new ResumableFileDownload.Request
        {
            Url = url,
            DestinationPath = destinationPath,
            FileName = fileName,
            ExpectedSize = totalBytes > 0 ? totalBytes : null,
            Progress = progress,
        }, cancellationToken);

    public void Dispose()
    {
        if (!_disposed)
        {
            _httpClient.Dispose();
            _discoveryService.Dispose();
            _disposed = true;
        }
    }
}
