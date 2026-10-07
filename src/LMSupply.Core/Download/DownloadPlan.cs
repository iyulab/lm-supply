namespace LMSupply.Download;

/// <summary>One repository file a download fetches, at the length the repository lists for it.</summary>
/// <param name="Path">The file's path in the repository (e.g. <c>onnx/encoder_model_int8.onnx</c>).</param>
/// <param name="SizeBytes">The length the repository listing gives for the file.</param>
public sealed record PlannedFile(string Path, long SizeBytes)
{
    /// <summary>
    /// Where the download stores the file, relative to the snapshot directory — <see cref="Path"/> unless the download
    /// puts it elsewhere: a load of a subfolder stores a tokenizer or config file the repository has only at its root
    /// beside the subfolder's model (<c>onnx/tokenizer.json</c> for the repository's <c>tokenizer.json</c>), where the
    /// load reads it.
    /// </summary>
    public string CachePath { get; init; } = Path;
}

/// <summary>
/// The files a download would fetch into an empty cache, chosen by the same rules the download itself uses —
/// what a consent screen or a disk/bandwidth budget needs before anything is fetched.
/// </summary>
public sealed class DownloadPlan
{
    /// <summary>The HuggingFace repository ID.</summary>
    public required string RepoId { get; init; }

    /// <summary>The revision the plan was made for.</summary>
    public required string Revision { get; init; }

    /// <summary>The files, in download order.</summary>
    public required IReadOnlyList<PlannedFile> Files { get; init; }

    /// <summary>
    /// Directories besides the repository's snapshots where the download also accepts a file at its repository path as
    /// already cached — the private tree earlier versions wrote GGUF files to. Empty for most plans.
    /// </summary>
    public IReadOnlyList<string> AlsoCachedIn { get; init; } = [];

    /// <summary>Bytes of all <see cref="Files"/> — the whole download, whatever the cache already holds.</summary>
    public long TotalBytes => Files.Sum(f => f.SizeBytes);

    /// <summary>
    /// Bytes a download of this plan into <paramref name="cacheDir"/> would still fetch: the files the cache does not
    /// hold where the download stores them (<see cref="PlannedFile.CachePath"/>; judged as
    /// <see cref="CacheManager.GetMissingFiles"/> judges them, or found at the repository path in <see cref="AlsoCachedIn"/>)
    /// and those it holds at a length other than the listing's, which the download discards and fetches again. A partly
    /// downloaded file counts in full. Reads the cache only; makes no request.
    /// </summary>
    /// <param name="cacheDir">The cache the download would write to.</param>
    public long GetRemainingBytes(string cacheDir)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(cacheDir);

        var missing = CacheManager.GetMissingFiles(cacheDir, RepoId, Files.Select(f => f.CachePath), revision: Revision)
            .ToHashSet(StringComparer.Ordinal);
        return Files
            .Where(f => !IsCachedAtListedLength(f, missing.Contains(f.CachePath) ? [] : CacheManager.GetSnapshotDirectories(cacheDir, RepoId, Revision)))
            .Sum(f => f.SizeBytes);
    }

    private bool IsCachedAtListedLength(PlannedFile file, IEnumerable<string> snapshots) =>
        snapshots.Select(directory => HubCache.GetPathInSnapshot(directory, file.CachePath))
            .Concat(AlsoCachedIn.Select(directory => HubCache.GetPathInSnapshot(directory, file.Path)))
            .Any(path => CacheManager.TryGetContentLength(path, out var length) && length == file.SizeBytes
                         && !CacheManager.IsLfsPointerFile(path));
}
