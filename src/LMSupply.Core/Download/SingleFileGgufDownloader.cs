using LMSupply.Download;
using LMSupply.Exceptions;
using LMSupply.Hardware;

namespace LMSupply.Core.Download;

/// <summary>
/// Downloads the single GGUF file a llama-server-backed embedder or reranker loads, and answers what that download
/// would be without making it. Shared by the domains so the selection, the cache layout and the plan cannot drift
/// apart between them; each domain names only the private tree its earlier versions wrote.
/// </summary>
internal class SingleFileGgufDownloader : IDisposable
{
    private readonly HttpClient _httpClient;
    private readonly ModelDiscoveryService _discoveryService;
    private readonly string _cacheDirectory;
    private readonly string _legacyTreeName;
    private readonly bool _localFilesOnly;
    private bool _disposed;

    // The revision GGUF files are read at; a loader names no other.
    private const string Revision = "main";

    /// <param name="cacheDirectory">Where downloaded GGUF files are kept.</param>
    /// <param name="legacyTreeName">The private tree earlier versions of the domain wrote (still read, never written).</param>
    /// <param name="localFilesOnly">
    /// True: serve from the cache only — no repository listing, no download; a repository with no cached
    /// GGUF file fails with <see cref="ModelNotFoundException"/>. The <c>DisableAutoDownload</c> option maps here.
    /// </param>
    /// <param name="handler">Test seam: the transport for listing and download; null for the default.</param>
    protected SingleFileGgufDownloader(string cacheDirectory, string legacyTreeName, bool localFilesOnly, HttpMessageHandler? handler)
    {
        _cacheDirectory = cacheDirectory;
        _legacyTreeName = legacyTreeName;
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
    /// Downloads the GGUF file a load of <paramref name="repoId"/> uses and returns its path.
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
            var cached = TrySelectFromLocalCache(_cacheDirectory, repoId, preferredQuantization, _legacyTreeName)
                ?? throw NotCached(repoId);

            progress?.Report(new DownloadProgress
            {
                FileName = Path.GetFileName(cached),
                BytesDownloaded = 1,
                TotalBytes = 1
            });
            return cached;
        }

        var files = await ListRepoFilesAsync(repoId, cancellationToken);
        var selectedFile = SelectBestFile(GgufFilesOf(files, repoId), preferredQuantization);

        // Check cache: a cached file counts only at the length the repository lists; one of another
        // length is not this file and is fetched again (a copy in another tool's snapshot is left alone).
        var expectedSize = selectedFile.Size > 0 ? selectedFile.Size : (long?)null;
        foreach (var (cachePath, readOnly) in GgufCacheLookup.CandidatePaths(_cacheDirectory, repoId, Revision, selectedFile.Path, _legacyTreeName))
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
        // half way never leaves a truncated model under a name the cache would accept.
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

    /// <summary>
    /// The file <see cref="DownloadAsync"/> would fetch for the same arguments, at the length the repository lists:
    /// chosen by the same selection, so the plan and the download cannot disagree. Downloads nothing. Offline, the
    /// cached file the load would open is the plan, at its length on disk.
    /// </summary>
    public async Task<DownloadPlan> PlanAsync(
        string repoId,
        string? preferredQuantization = null,
        CancellationToken cancellationToken = default)
    {
        if (_localFilesOnly)
        {
            var cached = TrySelectFromLocalCache(_cacheDirectory, repoId, preferredQuantization, _legacyTreeName)
                ?? throw NotCached(repoId);
            return new DownloadPlan
            {
                RepoId = repoId,
                Revision = Revision,
                Files = [new PlannedFile(Path.GetFileName(cached), CacheManager.GetContentLength(cached))],
                AlsoCachedIn = [GgufCacheLookup.GetLegacyDirectory(_cacheDirectory, _legacyTreeName, repoId)],
            };
        }

        var files = await ListRepoFilesAsync(repoId, cancellationToken);
        var selected = SelectBestFile(GgufFilesOf(files, repoId), preferredQuantization);
        if (selected.Size <= 0)
            throw new ModelDownloadException($"The listing of '{repoId}' gives no length for '{selected.Path}'.", repoId);

        return new DownloadPlan
        {
            RepoId = repoId,
            Revision = Revision,
            Files = [new PlannedFile(selected.Path, selected.Size)],
            AlsoCachedIn = [GgufCacheLookup.GetLegacyDirectory(_cacheDirectory, _legacyTreeName, repoId)],
        };
    }

    /// <summary>
    /// The cached GGUF file an offline load opens: the one matching the preferred quantization when
    /// there is one, otherwise the first by name — looked up in the repository's snapshots, then in the
    /// private tree earlier versions wrote (see <see cref="GgufCacheLookup"/>). Null when nothing of the
    /// repository is cached.
    /// </summary>
    internal static string? TrySelectFromLocalCache(
        string cacheDirectory, string repoId, string? preferredQuantization, string legacyTreeName)
    {
        var files = GgufCacheLookup.EnumerateCachedFiles(cacheDirectory, repoId, Revision, legacyTreeName);
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

    private ModelNotFoundException NotCached(string repoId) => new(
        $"No GGUF file of model '{repoId}' is in the local cache ({CacheManager.GetRepositoryDirectory(_cacheDirectory, repoId)}) and downloads are disabled.",
        repoId);

    private static List<RepoFile> GgufFilesOf(IReadOnlyList<RepoFile> files, string repoId)
    {
        var ggufFiles = files.Where(f => f.IsFile && f.Path.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase)).ToList();
        return ggufFiles.Count > 0
            ? ggufFiles
            : throw new ModelNotFoundException($"No GGUF files found in repository '{repoId}'.", repoId);
    }

    private Task<IReadOnlyList<RepoFile>> ListRepoFilesAsync(string repoId, CancellationToken cancellationToken)
        => _discoveryService.ListRepositoryFilesAsync(repoId, Revision, cancellationToken);

    private static RepoFile SelectBestFile(IReadOnlyList<RepoFile> files, string? preferredQuantization)
    {
        var rawFiles = files.Select(f => new GgufRawFile(Path.GetFileName(f.Path), f.Size));
        var groups = GgufFileGroup.GroupFiles(rawFiles).ToList();

        // These loaders open one file — split groups are excluded.
        var nonSplitGroups = groups.Where(g => !g.IsSplit).ToList();

        if (nonSplitGroups.Count > 0)
        {
            var memory = GgufFileSelector.FromHardwareProfile(HardwareProfile.Current);

            try
            {
                var selected = GgufFileSelector.Select(nonSplitGroups, memory, preferredQuantization);
                return files.First(f =>
                    Path.GetFileName(f.Path).Equals(selected.PrimaryFileName, StringComparison.OrdinalIgnoreCase));
            }
            catch (InvalidOperationException)
            {
                // Nothing fits memory: fall back to smallest non-split file
                var smallestGroup = nonSplitGroups.MinBy(g => g.TotalSizeBytes)!;
                return files.First(f =>
                    Path.GetFileName(f.Path).Equals(smallestGroup.PrimaryFileName, StringComparison.OrdinalIgnoreCase));
            }
        }

        throw new InvalidOperationException(
            "No single-file GGUF model found in repository. " +
            "This loader does not support split GGUF files (e.g., -00001-of-00003.gguf). " +
            "Please use a repository that provides a single-file GGUF model.");
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
