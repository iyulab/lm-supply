namespace LMSupply.Download;

/// <summary>
/// Finds a repository's GGUF files already in the cache for a single-file GGUF loader: in the repository's snapshots
/// for the revision (<see cref="CacheManager.GetSnapshotDirectories"/> — the hub layout downloads write, and
/// <c>snapshots/{revision}</c>), then in the private tree earlier versions wrote the loader's files to
/// (<c>{cache}/{tree}/{org}_{name}/</c>). The private tree is read only; nothing is moved out of it.
/// </summary>
internal static class GgufCacheLookup
{
    /// <summary>The private tree earlier versions wrote a loader's GGUF files of <paramref name="repoId"/> to.</summary>
    public static string GetLegacyDirectory(string cacheDir, string legacyTreeName, string repoId)
        => Path.Combine(cacheDir, legacyTreeName, repoId.Replace('/', '_').Replace('\\', '_'));

    /// <summary>
    /// Every cached GGUF file of <paramref name="repoId"/> holding real content: the snapshots' first, then the
    /// private tree's, each in ordinal path order.
    /// </summary>
    public static List<string> EnumerateCachedFiles(string cacheDir, string repoId, string revision, string legacyTreeName)
    {
        var result = new List<string>();
        foreach (var snapshot in CacheManager.GetSnapshotDirectories(cacheDir, repoId, revision))
            result.AddRange(EnumerateGguf(snapshot));

        var legacy = GetLegacyDirectory(cacheDir, legacyTreeName, repoId);
        if (Directory.Exists(legacy))
            result.AddRange(EnumerateGguf(legacy));

        return result;
    }

    /// <summary>
    /// Where <paramref name="repoPath"/> may already be cached, in lookup order, each with whether it must be left
    /// as it is (a snapshot another tool owns) rather than deleted when it turns out to be of the wrong length.
    /// </summary>
    public static IEnumerable<(string Path, bool ReadOnly)> CandidatePaths(
        string cacheDir, string repoId, string revision, string repoPath, string legacyTreeName)
    {
        foreach (var snapshot in CacheManager.GetSnapshotDirectories(cacheDir, repoId, revision))
            yield return (HubCache.GetPathInSnapshot(snapshot, repoPath), CacheManager.IsForeignSnapshot(cacheDir, repoId, revision, snapshot));

        yield return (HubCache.GetPathInSnapshot(GetLegacyDirectory(cacheDir, legacyTreeName, repoId), repoPath), false);
    }

    private static IEnumerable<string> EnumerateGguf(string directory)
        => Directory.EnumerateFiles(directory, "*.gguf", SearchOption.AllDirectories)
            .Where(CacheManager.IsCachedFile)
            .OrderBy(f => f, StringComparer.Ordinal);
}
