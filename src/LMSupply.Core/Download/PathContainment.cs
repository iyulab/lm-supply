namespace LMSupply.Core.Download;

/// <summary>
/// Whether a path lies inside a directory — the one check every cache lookup and download makes before it reads,
/// writes or deletes a file. Both sides are compared in their full form, so a directory written as <c>C:/cache</c>,
/// <c>C:\cache\</c> or <c>sub/../cache</c> contains the same files, and a sibling whose name merely starts with the
/// directory's (<c>cache-old</c>) does not.
/// </summary>
internal static class PathContainment
{
    // Windows names files without regard to case; elsewhere a differently cased directory is another directory,
    // so ignoring case there would let "../Cache/x" pass as inside "cache".
    private static StringComparison Comparison =>
        OperatingSystem.IsWindows() ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal;

    /// <summary>True when <paramref name="path"/> is strictly inside <paramref name="directory"/>.</summary>
    public static bool IsInside(string directory, string path)
    {
        var root = Path.GetFullPath(directory);
        var full = Path.TrimEndingDirectorySeparator(Path.GetFullPath(path));

        // A directory given with a trailing separator, and a volume root ("C:\", "/"), already ends in one.
        var prefix = Path.EndsInDirectorySeparator(root) ? root : root + Path.DirectorySeparatorChar;
        return full.Length > prefix.Length && full.StartsWith(prefix, Comparison);
    }

    /// <summary>
    /// The full path of <paramref name="relativePath"/> (<c>/</c>-separated, as a repository names its files) inside
    /// <paramref name="directory"/>, or <see langword="null"/> when it resolves outside it.
    /// </summary>
    public static string? Resolve(string directory, string relativePath)
    {
        var full = Path.GetFullPath(Path.Combine(
            Path.GetFullPath(directory), relativePath.Replace('/', Path.DirectorySeparatorChar)));
        return IsInside(directory, full) ? full : null;
    }
}
