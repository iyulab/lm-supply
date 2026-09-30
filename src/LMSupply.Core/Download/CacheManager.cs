using System.Diagnostics;
using LMSupply.Json;

namespace LMSupply.Download;

/// <summary>
/// Manages the HuggingFace-compatible cache directory structure.
/// </summary>
public static class CacheManager
{
    private static readonly System.Text.Json.JsonSerializerOptions s_jsonOptions = new()
    {
        TypeInfoResolver = CoreJsonContext.Default,
        PropertyNameCaseInsensitive = true
    };

    /// <summary>
    /// Gets the default cache directory following HuggingFace Hub standard.
    /// </summary>
    /// <remarks>
    /// Priority order:
    /// 1. HF_HUB_CACHE environment variable
    /// 2. HF_HOME environment variable + /hub
    /// 3. XDG_CACHE_HOME environment variable + /huggingface/hub
    /// 4. ~/.cache/huggingface/hub (default)
    /// </remarks>
    public static string GetDefaultCacheDirectory()
    {
        // 1. HF_HUB_CACHE (highest priority)
        var hfHubCache = Environment.GetEnvironmentVariable("HF_HUB_CACHE");
        if (!string.IsNullOrWhiteSpace(hfHubCache))
            return hfHubCache;

        // 2. HF_HOME + /hub
        var hfHome = Environment.GetEnvironmentVariable("HF_HOME");
        if (!string.IsNullOrWhiteSpace(hfHome))
            return Path.Combine(hfHome, "hub");

        // 3. XDG_CACHE_HOME + /huggingface/hub
        var xdgCache = Environment.GetEnvironmentVariable("XDG_CACHE_HOME");
        if (!string.IsNullOrWhiteSpace(xdgCache))
            return Path.Combine(xdgCache, "huggingface", "hub");

        // 4. Default: ~/.cache/huggingface/hub
        var userProfile = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        return Path.Combine(userProfile, ".cache", "huggingface", "hub");
    }

    /// <summary>
    /// Gets the snapshot directory a download of <paramref name="repoId"/> at <paramref name="revision"/> writes to:
    /// <c>models--{org}--{name}/snapshots/{revision}</c>. To find files already in the cache — including those
    /// other Hugging Face tools put there under the commit the revision points at — use
    /// <see cref="GetSnapshotDirectories"/>, <see cref="FindSnapshotDirectory"/> or <see cref="GetModelFilePath"/>.
    /// </summary>
    /// <param name="cacheDir">The base cache directory.</param>
    /// <param name="repoId">The HuggingFace repository ID (e.g., "sentence-transformers/all-MiniLM-L6-v2").</param>
    /// <param name="revision">The revision/branch (default: "main").</param>
    /// <returns>The full path to the model snapshot directory.</returns>
    public static string GetModelDirectory(string cacheDir, string repoId, string revision = "main")
        => Path.Combine(GetRepositoryDirectory(cacheDir, repoId), "snapshots", revision);

    /// <summary>
    /// The repository's directory in the cache: <c>models--{org}--{name}</c>.
    /// </summary>
    internal static string GetRepositoryDirectory(string cacheDir, string repoId)
        => Path.Combine(cacheDir, $"models--{repoId.Replace("/", "--")}");

    /// <summary>
    /// The snapshot directories in the cache that may hold <paramref name="repoId"/> at <paramref name="revision"/>,
    /// in lookup order. Only directories that exist are returned.
    /// </summary>
    /// <remarks>
    /// The Hugging Face hub cache records which commit a branch or tag points at in <c>refs/{revision}</c> and keeps
    /// that commit's files in <c>snapshots/{commit}</c> (each entry a link into <c>blobs/</c>, or a plain copy where
    /// links cannot be created). That snapshot comes first. The directory named after the revision itself,
    /// <c>snapshots/{revision}</c> — where downloads by this library are written — comes second. A revision that is
    /// itself a commit id names the same directory both ways and is returned once.
    /// </remarks>
    /// <param name="cacheDir">The base cache directory.</param>
    /// <param name="repoId">The HuggingFace repository ID.</param>
    /// <param name="revision">The revision: a branch, tag or commit id (default: "main").</param>
    public static IReadOnlyList<string> GetSnapshotDirectories(string cacheDir, string repoId, string revision = "main")
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(cacheDir);
        ArgumentException.ThrowIfNullOrWhiteSpace(repoId);
        ArgumentException.ThrowIfNullOrWhiteSpace(revision);

        var repoDir = GetRepositoryDirectory(cacheDir, repoId);
        var result = new List<string>(2);

        if (TryReadRef(repoDir, revision) is { } commit)
        {
            var committed = Path.Combine(repoDir, "snapshots", commit);
            if (Directory.Exists(committed))
                result.Add(committed);
        }

        var named = GetModelDirectory(cacheDir, repoId, revision);
        if (Directory.Exists(named) && !result.Any(d => SamePath(d, named)))
            result.Add(named);

        return result;
    }

    /// <summary>
    /// The first snapshot directory (see <see cref="GetSnapshotDirectories"/>) that holds every one of
    /// <paramref name="files"/> with real content, or <see langword="null"/> when none does. The files are
    /// looked up together because a loader opens them from one directory.
    /// </summary>
    /// <param name="cacheDir">The base cache directory.</param>
    /// <param name="repoId">The HuggingFace repository ID.</param>
    /// <param name="files">File paths (repository-style, <c>/</c>-separated), relative to <paramref name="subfolder"/> when one is given.</param>
    /// <param name="subfolder">Optional subfolder within the repository.</param>
    /// <param name="revision">The revision (default: "main").</param>
    /// <returns>The directory holding the files — the subfolder's own directory when one is given.</returns>
    public static string? FindSnapshotDirectory(
        string cacheDir,
        string repoId,
        IEnumerable<string> files,
        string? subfolder = null,
        string revision = "main")
    {
        ArgumentNullException.ThrowIfNull(files);
        var fileList = files as IReadOnlyCollection<string> ?? files.ToList();

        foreach (var snapshot in GetSnapshotDirectories(cacheDir, repoId, revision))
        {
            var directory = GetSubfolderDirectory(snapshot, subfolder);
            if (fileList.All(file => IsCachedFile(Path.Combine(directory, ToLocalPath(file)))))
                return directory;
        }

        return null;
    }

    /// <summary>
    /// Gets the full path to a file within a cached model: in the first snapshot directory
    /// (see <see cref="GetSnapshotDirectories"/>) that holds it, otherwise where a download writes it.
    /// </summary>
    public static string GetModelFilePath(string cacheDir, string repoId, string fileName, string revision = "main")
        => TryFindModelFile(cacheDir, repoId, fileName, revision)
           ?? Path.Combine(GetModelDirectory(cacheDir, repoId, revision), fileName);

    /// <summary>
    /// Checks if a model file is in the cache, in any snapshot directory <see cref="GetSnapshotDirectories"/> returns.
    /// </summary>
    public static bool ModelFileExists(string cacheDir, string repoId, string fileName, string revision = "main")
        => TryFindModelFile(cacheDir, repoId, fileName, revision) is not null;

    private static string? TryFindModelFile(string cacheDir, string repoId, string fileName, string revision)
        => GetSnapshotDirectories(cacheDir, repoId, revision)
            .Select(snapshot => Path.Combine(snapshot, ToLocalPath(fileName)))
            .FirstOrDefault(IsCachedFile);

    /// <summary>
    /// Whether <paramref name="directory"/> lies outside the snapshot this library writes <paramref name="repoId"/>
    /// at <paramref name="revision"/> to — a snapshot another tool owns, which is read and never modified.
    /// </summary>
    internal static bool IsForeignSnapshot(string cacheDir, string repoId, string revision, string directory)
    {
        var own = Path.TrimEndingDirectorySeparator(Path.GetFullPath(GetModelDirectory(cacheDir, repoId, revision)));
        var full = Path.TrimEndingDirectorySeparator(Path.GetFullPath(directory));
        return !string.Equals(full, own, StringComparison.OrdinalIgnoreCase)
               && !full.StartsWith(own + Path.DirectorySeparatorChar, StringComparison.OrdinalIgnoreCase);
    }

    /// <summary>
    /// The commit a revision points at, from <c>refs/{revision}</c> in the repository's cache directory, or
    /// <see langword="null"/> when there is no such ref or its content is not a commit id.
    /// </summary>
    private static string? TryReadRef(string repoDir, string revision)
    {
        var refsDir = Path.GetFullPath(Path.Combine(repoDir, "refs"));
        var refPath = Path.GetFullPath(Path.Combine(refsDir, ToLocalPath(revision)));
        if (!refPath.StartsWith(refsDir + Path.DirectorySeparatorChar, StringComparison.OrdinalIgnoreCase))
            return null;

        try
        {
            if (!File.Exists(refPath) || new FileInfo(refPath).Length > 256)
                return null;

            var commit = File.ReadAllText(refPath).Trim();
            return IsCommitId(commit) ? commit : null;
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            Trace.TraceInformation($"[CacheManager] Could not read ref '{refPath}': {ex.Message}");
            return null;
        }
    }

    /// <summary>Whether <paramref name="value"/> is a hexadecimal commit id (the hub writes full 40-character ids).</summary>
    internal static bool IsCommitId(string value) =>
        value.Length is >= 7 and <= 64 && value.All(char.IsAsciiHexDigit);

    private static string ToLocalPath(string repoPath) => repoPath.Replace('/', Path.DirectorySeparatorChar);

    private static bool SamePath(string a, string b) =>
        string.Equals(
            Path.TrimEndingDirectorySeparator(Path.GetFullPath(a)),
            Path.TrimEndingDirectorySeparator(Path.GetFullPath(b)),
            StringComparison.OrdinalIgnoreCase);

    /// <summary>
    /// The length of a file's content: for a symbolic link (as the hub cache's snapshot entries are), the length
    /// of the file it points at, not of the link itself.
    /// </summary>
    /// <param name="filePath">The file.</param>
    /// <param name="length">The content length; 0 when the method returns <see langword="false"/>.</param>
    /// <returns><see langword="false"/> when the file, or the file a link points at, does not exist.</returns>
    public static bool TryGetContentLength(string filePath, out long length)
    {
        length = ResolveContent(filePath)?.Length ?? -1;
        if (length >= 0)
            return true;

        length = 0;
        return false;
    }

    /// <summary>
    /// <see cref="TryGetContentLength"/> for a file known to exist.
    /// </summary>
    /// <exception cref="FileNotFoundException">The file, or the file a link points at, does not exist.</exception>
    internal static long GetContentLength(string filePath)
        => ResolveContent(filePath)?.Length
           ?? throw new FileNotFoundException($"'{filePath}' does not exist or is a link to a missing file.", filePath);

    /// <summary>Whether <paramref name="filePath"/> exists and, when it is a link, the file it points at exists.</summary>
    internal static bool ContentExists(string filePath) => ResolveContent(filePath) is not null;

    /// <summary>
    /// The file holding <paramref name="filePath"/>'s content — itself, or the final target when it is a link —
    /// or <see langword="null"/> when there is none.
    /// </summary>
    private static FileInfo? ResolveContent(string filePath)
    {
        try
        {
            var info = new FileInfo(filePath);
            if (!info.Exists)
                return null;
            if (info.LinkTarget is null)
                return info;

            return info.ResolveLinkTarget(returnFinalTarget: true) is FileInfo { Exists: true } target ? target : null;
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            Trace.TraceInformation($"[CacheManager] Could not resolve '{filePath}': {ex.Message}");
            return null;
        }
    }

    /// <summary>
    /// Lists which of <paramref name="files"/> a download of <paramref name="repoId"/> would still have to
    /// fetch: none when one snapshot directory (see <see cref="FindSnapshotDirectory"/>) holds them all,
    /// otherwise the files that are not in the directory downloads write to, or are cached there only as
    /// Git LFS pointers.
    /// </summary>
    /// <remarks>
    /// The answer comes from the directory layout and the presence test <see cref="HuggingFaceDownloader"/>
    /// applies before it fetches a file, so an empty result means a load of these files makes no request.
    /// It makes none itself.
    /// </remarks>
    /// <param name="cacheDir">The base cache directory.</param>
    /// <param name="repoId">The HuggingFace repository ID.</param>
    /// <param name="files">File names, relative to <paramref name="subfolder"/> when one is given.</param>
    /// <param name="subfolder">Optional subfolder within the repository (e.g., "languages/korean").</param>
    /// <param name="revision">The revision/branch (default: "main").</param>
    /// <returns>The file names that are not cached, in the order given.</returns>
    public static IReadOnlyList<string> GetMissingFiles(
        string cacheDir,
        string repoId,
        IEnumerable<string> files,
        string? subfolder = null,
        string revision = "main")
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(cacheDir);
        ArgumentException.ThrowIfNullOrWhiteSpace(repoId);
        ArgumentNullException.ThrowIfNull(files);

        var fileList = files.ToList();
        var directory = GetSubfolderDirectory(GetModelDirectory(cacheDir, repoId, revision), subfolder);
        if (FindSnapshotDirectory(cacheDir, repoId, fileList, subfolder, revision) is not null)
            return [];

        return fileList.Where(file => !IsCachedFile(Path.Combine(directory, file))).ToList();
    }

    /// <summary>
    /// Whether a file holds real content: present (for a link, its target present), and not a Git LFS
    /// pointer. The downloader fetches a file exactly when this is false.
    /// </summary>
    internal static bool IsCachedFile(string filePath) => ResolveContent(filePath) is not null && !IsLfsPointerFile(filePath);

    /// <summary>
    /// The local directory for a repository subfolder: the snapshot root when there is none, otherwise
    /// the subfolder's own directory beneath it. Refuses a subfolder that resolves outside the snapshot.
    /// </summary>
    internal static string GetSubfolderDirectory(string snapshotDir, string? subfolder)
    {
        if (string.IsNullOrEmpty(subfolder))
            return snapshotDir;

        var root = Path.GetFullPath(snapshotDir);
        var dir = Path.GetFullPath(Path.Combine(root, subfolder.Replace('/', Path.DirectorySeparatorChar)));
        var rootWithSeparator = Path.EndsInDirectorySeparator(root) ? root : root + Path.DirectorySeparatorChar;

        if (!dir.StartsWith(rootWithSeparator, StringComparison.OrdinalIgnoreCase))
            throw new InvalidOperationException($"Path traversal detected in subfolder: {subfolder}");

        return dir;
    }

    /// <summary>
    /// Checks if a file is a Git LFS pointer file instead of actual content.
    /// </summary>
    public static bool IsLfsPointerFile(string filePath)
    {
        // A link's own length says nothing about its content: measure the file it points at.
        var fileInfo = ResolveContent(filePath);
        if (fileInfo is null || fileInfo.Length > 1024)
            return false;

        try
        {
            var content = File.ReadAllText(filePath);
            return content.StartsWith("version https://git-lfs.github.com/spec/v1", StringComparison.Ordinal);
        }
        catch (Exception ex)
        {
            Trace.TraceInformation($"[CacheManager] LFS pointer check failed: {ex.Message}");
            return false;
        }
    }

    /// <summary>
    /// Deletes a cached model.
    /// </summary>
    /// <returns>True if the model was found and deleted, false if it didn't exist.</returns>
    public static bool DeleteModel(string cacheDir, string repoId)
    {
        var sanitizedRepoId = repoId.Replace("/", "--");
        var modelDir = Path.Combine(cacheDir, $"models--{sanitizedRepoId}");

        if (Directory.Exists(modelDir))
        {
            Directory.Delete(modelDir, recursive: true);
            return true;
        }

        return false;
    }

    /// <summary>
    /// Finds files in the cache that can be deleted without any model losing a file it reads. The one
    /// criterion, applied per snapshot: a file at the snapshot root whose same-name twin in an
    /// immediate subfolder has the same length and the same SHA-256, where that subfolder copy is
    /// the one a manifest lists — the copy the loader reads. Nothing else is reported: a root file
    /// with no twin, a twin of another length or content, or a twin no manifest knows, all stay.
    /// </summary>
    /// <remarks>
    /// 0.63.0 moved a subfolder's files from the snapshot root into the subfolder; a cache from
    /// before it kept the root copy and downloaded the file again (adopt-before-fetch now prevents
    /// new pairs, but the ones already on disk stay until something deletes them). "A file no loader
    /// reads" in general is not decidable here — any repository can be loaded by id — so the criterion
    /// is the measured case and nothing wider. Hashing reads every candidate pair once; on a cache of
    /// several gigabytes this takes seconds, which is why it is a separate call and not part of loading.
    /// </remarks>
    /// <param name="cacheDir">The cache directory.</param>
    /// <returns>The reclaimable files, largest first. Pass the list (or a subset) to <see cref="Reclaim"/>.</returns>
    public static IReadOnlyList<ReclaimableFile> FindReclaimable(string cacheDir)
    {
        var result = new List<ReclaimableFile>();
        if (!Directory.Exists(cacheDir))
            return result;

        foreach (var (repoId, revision) in GetCachedModels(cacheDir))
        {
            var snapshot = GetModelDirectory(cacheDir, repoId, revision);
            if (!Directory.Exists(snapshot))
                continue;

            var rootManifest = DownloadManifest.Read(snapshot);

            foreach (var rootFile in Directory.EnumerateFiles(snapshot))
            {
                var name = Path.GetFileName(rootFile);
                if (name.StartsWith('.'))
                    continue; // manifests, metadata, partial downloads

                long rootLength;
                try
                {
                    rootLength = new FileInfo(rootFile).Length;
                }
                catch (IOException)
                {
                    continue;
                }

                foreach (var subDir in Directory.EnumerateDirectories(snapshot))
                {
                    var twin = Path.Combine(subDir, name);
                    if (!File.Exists(twin) || new FileInfo(twin).Length != rootLength)
                        continue;

                    var subfolder = Path.GetFileName(subDir);
                    if (!IsListed(rootManifest, subfolder + "/" + name) && !IsListed(DownloadManifest.Read(subDir), name))
                        continue; // the subfolder copy is not what any manifest says the loader reads

                    if (!SameContent(rootFile, twin))
                        continue;

                    result.Add(new ReclaimableFile(
                        repoId,
                        rootFile,
                        rootLength,
                        twin,
                        $"'{name}' at the snapshot root is a byte-identical copy of '{subfolder}/{name}', which the manifest lists as the file the loader reads (a root copy from before 0.63.0, or the same repository loaded once by alias and once by id)."));
                    break;
                }
            }
        }

        return result.OrderByDescending(f => f.Size).ThenBy(f => f.Path, StringComparer.Ordinal).ToList();
    }

    /// <summary>
    /// Deletes the files <see cref="FindReclaimable"/> reported — the list as returned, or the subset
    /// the consumer chose. Each entry is re-checked before deletion: the file must still exist, live
    /// under <paramref name="cacheDir"/>, and its twin must still be present, so a stale list cannot
    /// delete a file that has since become the only copy.
    /// </summary>
    /// <param name="cacheDir">The cache directory the list was computed for.</param>
    /// <param name="files">Entries from <see cref="FindReclaimable"/>.</param>
    /// <returns>The number of bytes freed.</returns>
    public static long Reclaim(string cacheDir, IEnumerable<ReclaimableFile> files)
    {
        ArgumentNullException.ThrowIfNull(files);
        var root = Path.TrimEndingDirectorySeparator(Path.GetFullPath(cacheDir));
        long freed = 0;

        foreach (var file in files)
        {
            var path = Path.GetFullPath(file.Path);
            if (!path.StartsWith(root + Path.DirectorySeparatorChar, StringComparison.OrdinalIgnoreCase))
            {
                Trace.TraceWarning($"[CacheManager] Not reclaiming '{file.Path}': outside the cache directory.");
                continue;
            }

            if (!File.Exists(path) || !File.Exists(file.TwinPath))
                continue;

            try
            {
                var length = new FileInfo(path).Length;
                File.Delete(path);
                freed += length;
                Trace.TraceInformation($"[CacheManager] Reclaimed '{path}' ({length} bytes); '{file.TwinPath}' stays.");
            }
            catch (IOException ex)
            {
                Trace.TraceWarning($"[CacheManager] Could not delete '{path}': {ex.Message}");
            }
        }

        return freed;
    }

    private static bool IsListed(DownloadManifest? manifest, string path) =>
        manifest is not null && manifest.Files.Any(f =>
            string.Equals(f.Path.Replace('\\', '/'), path, StringComparison.OrdinalIgnoreCase));

    private static bool SameContent(string a, string b)
    {
        using var sa = File.OpenRead(a);
        using var sb = File.OpenRead(b);
        return System.Security.Cryptography.SHA256.HashData(sa).AsSpan()
            .SequenceEqual(System.Security.Cryptography.SHA256.HashData(sb));
    }

    /// <summary>
    /// Gets all cached models.
    /// </summary>
    /// <returns>Enumerable of (ModelId, Revision) tuples.</returns>
    public static IEnumerable<(string ModelId, string Revision)> GetCachedModels(string cacheDir)
    {
        if (!Directory.Exists(cacheDir))
            yield break;

        foreach (var modelDir in Directory.EnumerateDirectories(cacheDir, "models--*"))
        {
            var dirName = Path.GetFileName(modelDir);
            var modelId = dirName["models--".Length..].Replace("--", "/");

            var snapshotsDir = Path.Combine(modelDir, "snapshots");
            if (!Directory.Exists(snapshotsDir))
                continue;

            foreach (var revisionDir in Directory.EnumerateDirectories(snapshotsDir))
            {
                var revision = Path.GetFileName(revisionDir);
                yield return (modelId, revision);
            }
        }
    }

    /// <summary>
    /// Gets detailed information about all cached models.
    /// </summary>
    /// <param name="cacheDir">The cache directory path.</param>
    /// <returns>List of cached model information.</returns>
    public static IReadOnlyList<CachedModelInfo> GetCachedModelsWithInfo(string cacheDir)
    {
        var models = new List<CachedModelInfo>();

        if (!Directory.Exists(cacheDir))
            return models;

        foreach (var modelDir in Directory.EnumerateDirectories(cacheDir, "models--*"))
        {
            var dirName = Path.GetFileName(modelDir);
            var parts = dirName.Split("--");

            if (parts.Length >= 3)
            {
                var org = parts[1];
                var name = string.Join("/", parts.Skip(2));
                var repoId = $"{org}/{name}";

                var info = GetModelInfoInternal(modelDir, repoId);
                if (info != null)
                {
                    models.Add(info);
                }
            }
        }

        return models.OrderBy(m => m.RepoId).ToList();
    }

    /// <summary>
    /// Gets cached models filtered by type (excludes incomplete models).
    /// </summary>
    /// <param name="cacheDir">The cache directory path.</param>
    /// <param name="type">The model type to filter by.</param>
    /// <returns>List of complete cached models of the specified type.</returns>
    public static IReadOnlyList<CachedModelInfo> GetCachedModelsByType(string cacheDir, ModelType type)
    {
        return GetCachedModelsWithInfo(cacheDir)
            .Where(m => m.DetectedType == type && m.IsComplete)
            .ToList();
    }

    /// <summary>
    /// Gets the total size of all cached models.
    /// </summary>
    public static long GetTotalCacheSize(string cacheDir)
    {
        if (!Directory.Exists(cacheDir))
            return 0;

        return Directory.EnumerateFiles(cacheDir, "*", SearchOption.AllDirectories)
            .Sum(f => new FileInfo(f).Length);
    }

    private static CachedModelInfo? GetModelInfoInternal(string modelDir, string repoId)
    {
        try
        {
            var snapshotsDir = Path.Combine(modelDir, "snapshots");
            if (!Directory.Exists(snapshotsDir))
                return null;

            // The snapshot "main" resolves to (see GetSnapshotDirectories), else the most recent one
            var resolved = GetSnapshotDirectories(Path.GetDirectoryName(modelDir)!, repoId);
            var latestSnapshot = (resolved.Count > 0 ? resolved[0] : null)
                ?? Directory.GetDirectories(snapshotsDir)
                    .OrderByDescending(Directory.GetLastWriteTime)
                    .FirstOrDefault();

            if (latestSnapshot == null)
                return null;

            // Calculate file list and total size (a snapshot entry may be a link into blobs/)
            var files = Directory.GetFiles(latestSnapshot, "*", SearchOption.AllDirectories);
            var totalSize = files.Sum(f => ResolveContent(f)?.Length ?? 0);
            var fileNames = files.Select(Path.GetFileName).Where(n => n != null).ToList();

            // Detect model type
            var detectedType = DetectModelType(fileNames!, repoId);

            // Try to load cached metadata
            var metadata = TryLoadMetadata(modelDir);

            return new CachedModelInfo
            {
                RepoId = repoId,
                LocalPath = latestSnapshot,
                SizeBytes = totalSize,
                FileCount = files.Length,
                DetectedType = detectedType,
                LastModified = Directory.GetLastWriteTime(latestSnapshot),
                Files = fileNames!,
                Metadata = metadata
            };
        }
        catch (Exception ex)
        {
            Trace.TraceInformation($"[CacheManager] Model info build failed: {ex.Message}");
            return null;
        }
    }

    /// <summary>
    /// Tries to load cached metadata from .metadata.json file.
    /// </summary>
    private static ModelMetadata? TryLoadMetadata(string modelDir)
    {
        try
        {
            var metadataPath = Path.Combine(modelDir, ".metadata.json");
            if (!File.Exists(metadataPath))
                return null;

            var json = File.ReadAllText(metadataPath);
            return System.Text.Json.JsonSerializer.Deserialize(json, s_jsonOptions.TypeInfo<ModelMetadata>());
        }
        catch (Exception ex)
        {
            Trace.TraceInformation($"[CacheManager] Metadata load failed: {ex.Message}");
            return null;
        }
    }

    /// <summary>
    /// Detects the model type based on file patterns and repository ID.
    /// </summary>
    /// <param name="files">List of file names in the model directory.</param>
    /// <param name="repoId">The HuggingFace repository ID.</param>
    /// <returns>The detected model type.</returns>
    public static ModelType DetectModelType(IReadOnlyList<string> files, string repoId)
    {
        var fileSet = new HashSet<string>(files, StringComparer.OrdinalIgnoreCase);
        var repoLower = repoId.ToLowerInvariant();

        // 1. RepoId-based pattern matching (highest confidence)

        // Generator: phi, llama, mistral, qwen, etc. (LLMs)
        if (repoLower.Contains("phi") || repoLower.Contains("llama") ||
            repoLower.Contains("mistral") || repoLower.Contains("qwen") ||
            (repoLower.Contains("gpt") && !repoLower.Contains("gpt2-image")))
        {
            if (fileSet.Contains("genai_config.json"))
                return ModelType.Generator;
        }

        // Reranker: reranker, cross-encoder (must check BEFORE embedder — bge-reranker matches bge-)
        if (repoLower.Contains("rerank") || repoLower.Contains("cross-encoder"))
            return ModelType.Reranker;

        // Embedder: bge, e5, gte, minilm, mpnet, etc.
        if (repoLower.Contains("bge-") || repoLower.Contains("/e5-") ||
            repoLower.Contains("gte-") || repoLower.Contains("minilm") ||
            repoLower.Contains("mpnet") || repoLower.Contains("sentence-transformers") ||
            repoLower.Contains("embedding") || repoLower.Contains("embed"))
            return ModelType.Embedder;

        // ImageGenerator: stable-diffusion, lcm, dreamshaper, sdxl
        if (repoLower.Contains("stable-diffusion") || repoLower.Contains("lcm") ||
            repoLower.Contains("dreamshaper") || repoLower.Contains("sdxl") ||
            repoLower.Contains("txt2img") || repoLower.Contains("text-to-image"))
            return ModelType.ImageGenerator;

        // Transcriber: whisper
        if (repoLower.Contains("whisper"))
            return ModelType.Transcriber;

        // Synthesizer: piper, vits, tts
        if (repoLower.Contains("piper") || repoLower.Contains("vits") ||
            repoLower.Contains("tts") || repoLower.Contains("speech"))
            return ModelType.Synthesizer;

        // 2. File pattern-based detection (fallback)

        // Generator: genai_config.json
        if (fileSet.Contains("genai_config.json"))
            return ModelType.Generator;

        // Transcriber: encoder + decoder combination
        if (fileSet.Contains("encoder_model.onnx") && fileSet.Contains("decoder_model.onnx"))
            return ModelType.Transcriber;

        // Synthesizer: .onnx.json config file
        if (files.Any(f => f.EndsWith(".onnx.json", StringComparison.OrdinalIgnoreCase)))
            return ModelType.Synthesizer;

        // ImageGenerator: model_index.json or unet/text_encoder/vae structure
        if (fileSet.Contains("model_index.json") ||
            (files.Any(f => f.Contains("unet", StringComparison.OrdinalIgnoreCase)) &&
             files.Any(f => f.Contains("text_encoder", StringComparison.OrdinalIgnoreCase)) &&
             files.Any(f => f.Contains("vae", StringComparison.OrdinalIgnoreCase))))
            return ModelType.ImageGenerator;

        // Embedder: pooling layer or sentence-transformers structure
        if (fileSet.Contains("sentence_bert_config.json") ||
            fileSet.Contains("modules.json") ||
            files.Any(f => f.Contains("pooling", StringComparison.OrdinalIgnoreCase)))
            return ModelType.Embedder;

        // Single model.onnx + tokenizer (no decoder) → Embedder
        if (fileSet.Contains("model.onnx") &&
            (fileSet.Contains("tokenizer.json") || fileSet.Contains("vocab.txt")) &&
            !files.Any(f => f.Contains("decoder", StringComparison.OrdinalIgnoreCase)))
            return ModelType.Embedder;

        return ModelType.Unknown;
    }
}
