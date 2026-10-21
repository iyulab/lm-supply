using System.Collections.Concurrent;
using System.Diagnostics;

namespace LMSupply.Download;

/// <summary>
/// Finds a repository's GGUF files already in the cache for a single-file GGUF loader: in the repository's snapshots
/// for the revision (<see cref="CacheManager.GetSnapshotDirectories"/> — the hub layout downloads write, and
/// <c>snapshots/{revision}</c>). A loader whose earlier versions wrote a private tree
/// (<c>{cache}/{tree}/{org}_{name}/</c>) has that tree's files moved into <c>snapshots/{revision}</c> on first lookup, and
/// reads the tree where the move could not be made (a read-only cache).
/// </summary>
internal static class GgufCacheLookup
{
    private static readonly ConcurrentDictionary<string, bool> s_moved = new(StringComparer.Ordinal);

    // COMPAT(remove-after: 0.150.0) — the private trees GGUF files were written to before 0.98.0. Every lookup moves
    // them into the repository's snapshots; after this release they are neither moved nor read.
    /// <summary>The private tree earlier versions wrote a loader's GGUF files of <paramref name="repoId"/> to.</summary>
    public static string GetLegacyDirectory(string cacheDir, string legacyTreeName, string repoId)
        => Path.Combine(cacheDir, legacyTreeName, repoId.Replace('/', '_').Replace('\\', '_'));

    /// <summary>
    /// Every cached GGUF file of <paramref name="repoId"/> holding real content: the snapshots' first, then what is left
    /// of the private tree, each in ordinal path order.
    /// </summary>
    public static List<string> EnumerateCachedFiles(string cacheDir, string repoId, string revision, string? legacyTreeName)
    {
        MoveLegacyTree(cacheDir, legacyTreeName, repoId, revision);
        var result = new List<string>();
        foreach (var snapshot in CacheManager.GetSnapshotDirectories(cacheDir, repoId, revision))
            result.AddRange(EnumerateGguf(snapshot));

        if (legacyTreeName is not null && Directory.Exists(GetLegacyDirectory(cacheDir, legacyTreeName, repoId)))
            result.AddRange(EnumerateGguf(GetLegacyDirectory(cacheDir, legacyTreeName, repoId)));

        return result;
    }

    /// <summary>
    /// Where <paramref name="repoPath"/> may already be cached, in lookup order, each with whether it must be left
    /// as it is (a snapshot another tool owns) rather than deleted when it turns out to be of the wrong length.
    /// </summary>
    public static IEnumerable<(string Path, bool ReadOnly)> CandidatePaths(
        string cacheDir, string repoId, string revision, string repoPath, string? legacyTreeName)
    {
        MoveLegacyTree(cacheDir, legacyTreeName, repoId, revision);
        foreach (var snapshot in CacheManager.GetSnapshotDirectories(cacheDir, repoId, revision))
            yield return (HubCache.GetPathInSnapshot(snapshot, repoPath), CacheManager.IsForeignSnapshot(cacheDir, repoId, revision, snapshot));

        if (legacyTreeName is not null)
            yield return (HubCache.GetPathInSnapshot(GetLegacyDirectory(cacheDir, legacyTreeName, repoId), repoPath), false);
    }

    /// <summary>
    /// Moves the files of <paramref name="repoId"/>'s private tree into <c>snapshots/{revision}</c> — once per process —
    /// and removes the tree when nothing is left in it. A file already in the snapshot at the same length is not moved
    /// again (the tree's copy is deleted); one of another length stays where it is. Best effort: a cache that cannot
    /// be written keeps its tree, which <see cref="EnumerateCachedFiles"/> still reads.
    /// </summary>
    internal static void MoveLegacyTree(string cacheDir, string? legacyTreeName, string repoId, string revision)
    {
        if (legacyTreeName is null)
            return;

        var legacy = GetLegacyDirectory(cacheDir, legacyTreeName, repoId);
        if (!Directory.Exists(legacy) || !s_moved.TryAdd(Path.GetFullPath(legacy), true))
            return;

        var snapshot = CacheManager.GetModelDirectory(cacheDir, repoId, revision);
        try
        {
            foreach (var file in Directory.EnumerateFiles(legacy, "*", SearchOption.AllDirectories).ToList())
            {
                if (!CacheManager.IsCachedFile(file))
                    continue;

                var target = Path.Combine(snapshot, Path.GetRelativePath(legacy, file));
                if (File.Exists(target))
                {
                    if (CacheManager.GetContentLength(target) == CacheManager.GetContentLength(file))
                        File.Delete(file);
                    continue;
                }

                Directory.CreateDirectory(Path.GetDirectoryName(target)!);
                File.Move(file, target);
            }

            DeleteIfEmpty(legacy);
            DeleteIfEmpty(Path.GetDirectoryName(legacy)!);
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            Trace.TraceWarning($"[GgufCacheLookup] Left '{legacy}' in place: {ex.Message}");
        }
    }

    private static void DeleteIfEmpty(string directory)
    {
        if (Directory.Exists(directory) && !Directory.EnumerateFiles(directory, "*", SearchOption.AllDirectories).Any())
            Directory.Delete(directory, recursive: true);
    }

    private static IEnumerable<string> EnumerateGguf(string directory)
        => Directory.EnumerateFiles(directory, "*.gguf", SearchOption.AllDirectories)
            .Where(CacheManager.IsCachedFile)
            .OrderBy(f => f, StringComparer.Ordinal);
}
