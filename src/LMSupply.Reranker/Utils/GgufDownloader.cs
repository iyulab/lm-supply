using LMSupply.Core.Download;
using LMSupply.Download;
using LMSupply.Exceptions;
using LMSupply.Hardware;

namespace LMSupply.Reranker.Utils;

/// <summary>
/// Downloads GGUF reranker model files from HuggingFace.
/// </summary>
internal sealed class GgufDownloader : IDisposable
{
    private readonly HttpClient _httpClient;
    private readonly ModelDiscoveryService _discoveryService;
    private readonly string _cacheDirectory;
    private readonly bool _localFilesOnly;
    private bool _disposed;

    // The revision GGUF files are read at; a loader names no other.
    private const string Revision = "main";

    // Where earlier versions kept downloaded files (still read, never written).
    private const string LegacyTreeName = "gguf-rerankers";

    /// <param name="cacheDirectory">Where downloaded GGUF files are kept.</param>
    /// <param name="localFilesOnly">
    /// True: serve from the cache only — no repository listing, no download; a repository with no cached
    /// GGUF file fails with <see cref="ModelNotFoundException"/>. The <c>DisableAutoDownload</c> option maps here.
    /// </param>
    public GgufDownloader(string cacheDirectory, bool localFilesOnly = false)
    {
        _cacheDirectory = cacheDirectory;
        _localFilesOnly = localFilesOnly;
        _httpClient = new HttpClient
        {
            Timeout = TimeSpan.FromMinutes(30)
        };
        _httpClient.DefaultRequestHeaders.Add("User-Agent", "LMSupply/1.0");
        _discoveryService = new ModelDiscoveryService(cacheDirectory);
    }

    /// <summary>
    /// Downloads a GGUF reranker model file from HuggingFace.
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
            var cached = TrySelectFromLocalCache(_cacheDirectory, repoId, preferredQuantization)
                ?? throw new ModelNotFoundException(
                    $"No GGUF file of model '{repoId}' is in the local cache ({CacheManager.GetRepositoryDirectory(_cacheDirectory, repoId)}) and downloads are disabled.",
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
        // length is not this file and is fetched again (a copy in another tool's snapshot is left alone).
        var expectedSize = selectedFile.Size > 0 ? selectedFile.Size : (long?)null;
        foreach (var (cachePath, readOnly) in GgufCacheLookup.CandidatePaths(_cacheDirectory, repoId, Revision, selectedFile.Path, LegacyTreeName))
        {
            if (!ResumableFileDownload.IsUsableCachedFile(cachePath, expectedSize, readOnly))
                continue;

            progress?.Report(new DownloadProgress
            {
                FileName = selectedFile.Path,
                BytesDownloaded = 1,
                TotalBytes = 1
            });
            return cachePath;
        }

        progress?.Report(new DownloadProgress
        {
            FileName = selectedFile.Path,
            BytesDownloaded = 0,
            TotalBytes = selectedFile.Size
        });

        // Into the hub cache layout (blobs, snapshots/{commit}, refs), where other Hugging Face tools find it. The
        // bytes land in a ".part" and take the final name only once the whole file is there, so a transfer that stops
        // half way never leaves a truncated model under a name the cache - and IsModelDownloaded - would accept.
        return await HubCache.DownloadFileAsync(
            _httpClient, _cacheDirectory, repoId, Revision, selectedFile.Path, expectedSize,
            HubCache.BlobIdsOf(files).GetValueOrDefault(selectedFile.Path),
            (url, destination) => new ResumableFileDownload.Request
            {
                Url = url,
                DestinationPath = destination,
                FileName = selectedFile.Path,
                ModelId = repoId,
                ExpectedSize = expectedSize,
                Progress = progress,
            },
            cancellationToken);
    }

    private Task<IReadOnlyList<RepoFile>> ListRepoFilesAsync(string repoId, CancellationToken cancellationToken)
    {
        return _discoveryService.ListRepositoryFilesAsync(repoId, Revision, cancellationToken);
    }

    private static RepoFile SelectBestFile(IReadOnlyList<RepoFile> files, string? preferredQuantization)
    {
        var rawFiles = files.Select(f => new GgufRawFile(Path.GetFileName(f.Path), f.Size));
        var groups = GgufFileGroup.GroupFiles(rawFiles).ToList();

        // Reranker only supports single-file download — exclude split groups
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

        // Split-only repo: single-file reranker cannot use split GGUF files.
        throw new InvalidOperationException(
            "No single-file GGUF model found in repository. " +
            "The reranker does not support split GGUF files (e.g., -00001-of-00003.gguf). " +
            "Please use a repository that provides a single-file GGUF model.");
    }

    /// <summary>
    /// The cached GGUF file an offline load opens: the one matching the preferred quantization when
    /// there is one, otherwise the first by name — looked up in the repository's snapshots, then in the
    /// private tree earlier versions wrote (see <see cref="GgufCacheLookup"/>). Null when nothing of the
    /// repository is cached.
    /// A file that holds no model — a Git LFS pointer — is not counted, and a download that stopped half
    /// way never reaches a <c>.gguf</c> name (see <see cref="ResumableFileDownload"/>).
    /// </summary>
    /// <remarks>
    /// Static and free of side effects so that <see cref="LocalReranker.IsModelDownloaded"/> answers from
    /// the same choice the loader makes, without opening an HTTP client to ask.
    /// </remarks>
    internal static string? TrySelectFromLocalCache(string cacheDirectory, string repoId, string? preferredQuantization)
    {
        var files = GgufCacheLookup.EnumerateCachedFiles(cacheDirectory, repoId, Revision, LegacyTreeName);
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
