using System.Diagnostics;
using System.Text.Json;
using LMSupply.Core.Download;

namespace LMSupply.Download;

/// <summary>
/// Writes downloads in the Hugging Face hub cache layout, so other Hugging Face tools find them:
/// the content in <c>blobs/{blob id}</c>, the commit's view of the repository in <c>snapshots/{commit}/</c> (each
/// entry a relative link to its blob, or the blob itself moved there where links cannot be created), and the
/// commit a branch or tag points at in <c>refs/{revision}</c>.
/// </summary>
/// <remarks>
/// A download writes this layout only when the revision resolves to a commit. When it does not — offline, a failed
/// request, a server that does not answer — the download keeps the layout of earlier versions: plain files in
/// <c>snapshots/{revision}/</c>, which every lookup in this library still reads.
/// </remarks>
internal static class HubCache
{
    private const string HuggingFaceBaseUrl = "https://huggingface.co";

    // The revision lookup is a small JSON answer; the downloaders' clients allow minutes for multi-gigabyte bodies.
    private static readonly TimeSpan s_revisionLookupTimeout = TimeSpan.FromSeconds(30);

    /// <summary>The directory of <paramref name="blobId"/> in the repository's cache directory.</summary>
    public static string GetBlobPath(string repoDir, string blobId) => Path.Combine(repoDir, "blobs", blobId);

    /// <summary>Whether <paramref name="value"/> names a blob: a Git blob id (40 hex) or a SHA-256 (64 hex).</summary>
    public static bool IsBlobId(string? value) =>
        value is { Length: 40 or 64 } && value.All(char.IsAsciiHexDigit);

    /// <summary>Whether <paramref name="revision"/> is a full commit id, which needs no lookup.</summary>
    public static bool IsFullCommitId(string revision) => revision.Length == 40 && CacheManager.IsCommitId(revision);

    /// <summary>
    /// The blob id of every file of a repository listing, keyed by repository path. Empty when the listing carries no
    /// Git LFS entry at all: a listing cached before LFS entries were recorded cannot tell an LFS file (named by its
    /// SHA-256 in the hub cache) from a Git file (named by its Git blob id), so its files are written without a blob.
    /// </summary>
    public static IReadOnlyDictionary<string, string> BlobIdsOf(IEnumerable<RepoFile> listing)
    {
        var files = listing.Where(f => f.IsFile).ToList();
        var result = new Dictionary<string, string>(StringComparer.Ordinal);
        if (!files.Any(f => f.Lfs is not null))
            return result;

        foreach (var file in files)
        {
            if (IsBlobId(file.BlobId))
                result.TryAdd(file.Path, file.BlobId!.ToLowerInvariant());
        }

        return result;
    }

    /// <summary>
    /// The commit <paramref name="revision"/> of <paramref name="repoId"/> points at, from the Hub's revision endpoint;
    /// <paramref name="revision"/> itself when it already is a full commit id. <see langword="null"/> when the commit
    /// cannot be obtained — the caller then writes the layout of earlier versions.
    /// </summary>
    public static async Task<string?> TryResolveCommitAsync(
        HttpClient httpClient, string repoId, string revision, CancellationToken cancellationToken)
    {
        if (IsFullCommitId(revision))
            return revision.ToLowerInvariant();

        using var timeout = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        timeout.CancelAfter(s_revisionLookupTimeout);

        var url = $"{HuggingFaceBaseUrl}/api/models/{repoId}/revision/{Uri.EscapeDataString(revision)}";
        try
        {
            using var response = await httpClient.GetAsync(url, HttpCompletionOption.ResponseContentRead, timeout.Token).ConfigureAwait(false);
            if (!response.IsSuccessStatusCode)
            {
                Trace.TraceInformation(
                    $"[HubCache] Revision '{revision}' of '{repoId}' did not resolve (HTTP {(int)response.StatusCode}); writing snapshots/{revision}.");
                return null;
            }

            await using var stream = await response.Content.ReadAsStreamAsync(timeout.Token).ConfigureAwait(false);
            using var document = await JsonDocument.ParseAsync(stream, cancellationToken: timeout.Token).ConfigureAwait(false);
            if (document.RootElement.ValueKind == JsonValueKind.Object
                && document.RootElement.TryGetProperty("sha", out var sha)
                && sha.ValueKind == JsonValueKind.String
                && sha.GetString() is { } commit
                && IsFullCommitId(commit))
            {
                return commit.ToLowerInvariant();
            }

            Trace.TraceInformation($"[HubCache] The revision answer for '{repoId}' ({revision}) carries no commit id; writing snapshots/{revision}.");
            return null;
        }
        catch (OperationCanceledException) when (!cancellationToken.IsCancellationRequested)
        {
            Trace.TraceInformation($"[HubCache] Revision lookup for '{repoId}' ({revision}) timed out; writing snapshots/{revision}.");
            return null;
        }
        catch (Exception ex) when (ex is HttpRequestException or JsonException or IOException or InvalidOperationException)
        {
            Trace.TraceInformation(
                $"[HubCache] Revision lookup for '{repoId}' ({revision}) failed ({ex.GetType().Name}: {ex.Message}); writing snapshots/{revision}.");
            return null;
        }
    }

    /// <summary>
    /// Records that <paramref name="revision"/> points at <paramref name="commit"/>: <c>refs/{revision}</c> holds the
    /// commit id, without a trailing newline, as the hub cache writes it. Nothing is written for a revision that is
    /// the commit itself, or one that would resolve outside <c>refs/</c>.
    /// </summary>
    public static void WriteRef(string repoDir, string revision, string commit)
    {
        if (string.Equals(revision, commit, StringComparison.OrdinalIgnoreCase))
            return;

        var refsDir = Path.GetFullPath(Path.Combine(repoDir, "refs"));
        var refPath = Path.GetFullPath(Path.Combine(refsDir, revision.Replace('/', Path.DirectorySeparatorChar)));
        if (!refPath.StartsWith(refsDir + Path.DirectorySeparatorChar, StringComparison.OrdinalIgnoreCase))
            return;

        try
        {
            if (File.Exists(refPath) && string.Equals(File.ReadAllText(refPath).Trim(), commit, StringComparison.OrdinalIgnoreCase))
                return;

            Directory.CreateDirectory(Path.GetDirectoryName(refPath)!);
            File.WriteAllText(refPath, commit);
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            Trace.TraceWarning($"[HubCache] Could not write ref '{refPath}': {ex.Message}");
        }
    }

    /// <summary>
    /// Puts <paramref name="blobPath"/> at <paramref name="pointerPath"/> in a snapshot: a relative symbolic link to
    /// the blob, as the hub cache does. Where links cannot be created (Windows without developer mode) a blob this
    /// download just fetched is moved there — one copy on disk, as <c>huggingface_hub</c> does — and a blob that was
    /// already in the cache (another tool's, or another file's) is copied, so it stays where others find it.
    /// </summary>
    /// <returns><see langword="true"/> when a link was created (or one to the same blob was already there).</returns>
    public static bool LinkOrMove(string blobPath, string pointerPath, bool newBlob)
        => LinkOrMove(blobPath, pointerPath, newBlob, static (link, target) => File.CreateSymbolicLink(link, target));

    /// <summary><see cref="LinkOrMove(string, string, bool)"/> with the link creation supplied (a test seam).</summary>
    internal static bool LinkOrMove(string blobPath, string pointerPath, bool newBlob, Action<string, string> createSymbolicLink)
    {
        var blob = Path.GetFullPath(blobPath);
        var pointer = Path.GetFullPath(pointerPath);
        Directory.CreateDirectory(Path.GetDirectoryName(pointer)!);

        if (PointsAt(pointer, blob))
            return true;

        RemoveEntry(pointer);
        try
        {
            createSymbolicLink(pointer, Path.GetRelativePath(Path.GetDirectoryName(pointer)!, blob));
            return true;
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or PlatformNotSupportedException)
        {
            // Another caller placing the same file may have linked it in the meantime.
            if (PointsAt(pointer, blob))
                return true;

            Trace.TraceInformation(
                $"[HubCache] Cannot link '{pointer}' to its blob ({ex.Message}); {(newBlob ? "moving" : "copying")} the blob there instead.");
            RemoveEntry(pointer);
            if (newBlob)
                File.Move(blob, pointer, overwrite: true);
            else
                File.Copy(blob, pointer, overwrite: true);
            return false;
        }
    }

    /// <summary>
    /// Downloads one repository file to <paramref name="pointerPath"/> in a hub-layout snapshot: into
    /// <c>blobs/{blobId}</c> (resumable, the ".part" beside the blob) and then linked or moved into place
    /// (<see cref="LinkOrMove(string, string, bool)"/>). A blob already in the cache at the expected length is linked
    /// without a request. Without a blob id the file is downloaded straight to <paramref name="pointerPath"/>.
    /// </summary>
    /// <param name="httpClient">The client to download with.</param>
    /// <param name="repoDir">The repository's cache directory.</param>
    /// <param name="pointerPath">The file's path in the snapshot.</param>
    /// <param name="blobId">The file's blob id (see <see cref="RepoFile.BlobId"/>), or <see langword="null"/>.</param>
    /// <param name="expectedSize">The length the repository listed, when known.</param>
    /// <param name="request">Builds the download request for a destination path.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    public static async Task DownloadAsync(
        HttpClient httpClient,
        string repoDir,
        string pointerPath,
        string? blobId,
        long? expectedSize,
        Func<string, ResumableFileDownload.Request> request,
        CancellationToken cancellationToken)
    {
        if (!IsBlobId(blobId))
        {
            await ResumableFileDownload.DownloadAsync(httpClient, request(pointerPath), cancellationToken).ConfigureAwait(false);
            return;
        }

        var blobPath = GetBlobPath(repoDir, blobId!);
        var present = CacheManager.IsCachedFile(blobPath)
                      && (expectedSize is not { } expected || CacheManager.GetContentLength(blobPath) == expected);
        if (present)
        {
            Trace.TraceInformation($"[HubCache] '{pointerPath}' is already in the cache as blob '{blobId}'; nothing to download.");
        }
        else
        {
            await ResumableFileDownload.DownloadAsync(httpClient, request(blobPath), cancellationToken).ConfigureAwait(false);
        }

        LinkOrMove(blobPath, pointerPath, newBlob: !present);
    }

    /// <summary>
    /// Downloads one file of <paramref name="repoId"/> into the cache for a single-file loader (a GGUF model): into the
    /// snapshot of the commit <paramref name="revision"/> resolves to, in the hub layout, with the ref and a manifest
    /// entry that marks the snapshot as this library's; or, when the commit cannot be resolved, into
    /// <c>snapshots/{revision}/</c> as earlier versions did.
    /// </summary>
    /// <param name="httpClient">The client to download with.</param>
    /// <param name="cacheDir">The cache directory.</param>
    /// <param name="repoId">The repository.</param>
    /// <param name="revision">The revision requested.</param>
    /// <param name="repoPath">The file's path in the repository (<c>/</c>-separated).</param>
    /// <param name="expectedSize">The length the repository listed, when known.</param>
    /// <param name="blobId">The file's blob id, when the listing carried one.</param>
    /// <param name="request">Builds the download request for a resolve URL and a destination path.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>The file's path in the snapshot.</returns>
    public static async Task<string> DownloadFileAsync(
        HttpClient httpClient,
        string cacheDir,
        string repoId,
        string revision,
        string repoPath,
        long? expectedSize,
        string? blobId,
        Func<string, string, ResumableFileDownload.Request> request,
        CancellationToken cancellationToken)
    {
        var repoDir = CacheManager.GetRepositoryDirectory(cacheDir, repoId);
        var commit = await TryResolveCommitAsync(httpClient, repoId, revision, cancellationToken).ConfigureAwait(false);
        var snapshotDir = CacheManager.GetModelDirectory(cacheDir, repoId, commit ?? revision);
        var pointerPath = GetPathInSnapshot(snapshotDir, repoPath);
        var url = GetResolveUrl(repoId, commit ?? revision, repoPath);

        if (commit is null)
        {
            Directory.CreateDirectory(Path.GetDirectoryName(pointerPath)!);
            await ResumableFileDownload.DownloadAsync(httpClient, request(url, pointerPath), cancellationToken).ConfigureAwait(false);
            return pointerPath;
        }

        await DownloadAsync(httpClient, repoDir, pointerPath, blobId, expectedSize, destination => request(url, destination), cancellationToken)
            .ConfigureAwait(false);
        await RecordFileAsync(repoDir, snapshotDir, repoId, revision, repoPath, expectedSize ?? CacheManager.GetContentLength(pointerPath), verified: expectedSize is not null)
            .ConfigureAwait(false);
        WriteRef(repoDir, revision, commit);
        return pointerPath;
    }

    /// <summary>The <c>resolve</c> URL of a repository file at a revision.</summary>
    public static string GetResolveUrl(string repoId, string revision, string repoPath)
        => $"{HuggingFaceBaseUrl}/{repoId}/resolve/{Uri.EscapeDataString(revision)}/{repoPath}";

    /// <summary>
    /// The local path of <paramref name="repoPath"/> in <paramref name="snapshotDir"/>; refuses a path that resolves
    /// outside the snapshot.
    /// </summary>
    public static string GetPathInSnapshot(string snapshotDir, string repoPath)
    {
        var root = Path.TrimEndingDirectorySeparator(Path.GetFullPath(snapshotDir));
        var path = Path.GetFullPath(Path.Combine(root, repoPath.Replace('/', Path.DirectorySeparatorChar)));
        if (!path.StartsWith(root + Path.DirectorySeparatorChar, StringComparison.OrdinalIgnoreCase))
            throw new InvalidOperationException($"Path traversal detected in file path: {repoPath}");
        return path;
    }

    /// <summary>Adds (or updates) one file in the manifest of a hub-layout snapshot this library downloaded into.</summary>
    private static async Task RecordFileAsync(
        string repoDir, string snapshotDir, string repoId, string revision, string repoPath, long size, bool verified)
    {
        var manifest = DownloadManifest.ReadFile(DownloadManifest.GetPath(repoDir, snapshotDir, subfolder: null))
                       ?? new DownloadManifest { Version = DownloadManifest.VerifiedVersion };
        manifest.RepoId ??= repoId;
        manifest.Revision = revision;
        manifest.Files.RemoveAll(f => string.Equals(f.Path, repoPath, StringComparison.Ordinal));
        manifest.Files.Add(new ManifestFileEntry { Path = repoPath, Size = size });
        if (!verified)
            manifest.Version = 1;

        await DownloadManifest.WriteForSnapshotAsync(repoDir, snapshotDir, subfolder: null, manifest).ConfigureAwait(false);
    }

    // Whether the entry at pointer is a link that resolves to blob.
    private static bool PointsAt(string pointer, string blob)
    {
        try
        {
            var info = new FileInfo(pointer);
            return info.LinkTarget is not null
                   && info.ResolveLinkTarget(returnFinalTarget: true) is { Exists: true } target
                   && string.Equals(Path.GetFullPath(target.FullName), blob, StringComparison.OrdinalIgnoreCase);
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            return false;
        }
    }

    // Deletes whatever is at path — a file, or a link (even one whose target is gone). Never follows a link.
    private static void RemoveEntry(string path)
    {
        var info = new FileInfo(path);
        if (info.Exists || info.LinkTarget is not null)
            File.Delete(path);
    }
}
