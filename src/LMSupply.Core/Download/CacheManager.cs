namespace LMSupply.Download;

/// <summary>
/// Manages the HuggingFace-compatible cache directory structure.
/// </summary>
public static class CacheManager
{
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
    /// Gets the model directory path for a given repository ID and revision.
    /// </summary>
    /// <param name="cacheDir">The base cache directory.</param>
    /// <param name="repoId">The HuggingFace repository ID (e.g., "sentence-transformers/all-MiniLM-L6-v2").</param>
    /// <param name="revision">The revision/branch (default: "main").</param>
    /// <returns>The full path to the model snapshot directory.</returns>
    public static string GetModelDirectory(string cacheDir, string repoId, string revision = "main")
    {
        // HuggingFace cache structure: models--{org}--{model}/snapshots/{revision}
        var sanitizedRepoId = repoId.Replace("/", "--");
        return Path.Combine(cacheDir, $"models--{sanitizedRepoId}", "snapshots", revision);
    }

    /// <summary>
    /// Gets the full path to a file within a cached model.
    /// </summary>
    public static string GetModelFilePath(string cacheDir, string repoId, string fileName, string revision = "main")
    {
        var modelDir = GetModelDirectory(cacheDir, repoId, revision);
        return Path.Combine(modelDir, fileName);
    }

    /// <summary>
    /// Checks if a model file exists in the cache.
    /// </summary>
    public static bool ModelFileExists(string cacheDir, string repoId, string fileName, string revision = "main")
    {
        var filePath = GetModelFilePath(cacheDir, repoId, fileName, revision);
        return File.Exists(filePath) && !IsLfsPointerFile(filePath);
    }

    /// <summary>
    /// Checks if a file is a Git LFS pointer file instead of actual content.
    /// </summary>
    public static bool IsLfsPointerFile(string filePath)
    {
        if (!File.Exists(filePath))
            return false;

        var fileInfo = new FileInfo(filePath);
        if (fileInfo.Length > 1024)
            return false;

        try
        {
            var content = File.ReadAllText(filePath);
            return content.StartsWith("version https://git-lfs.github.com/spec/v1");
        }
        catch
        {
            return false;
        }
    }

    /// <summary>
    /// Deletes a cached model.
    /// </summary>
    public static void DeleteModel(string cacheDir, string repoId)
    {
        var sanitizedRepoId = repoId.Replace("/", "--");
        var modelDir = Path.Combine(cacheDir, $"models--{sanitizedRepoId}");

        if (Directory.Exists(modelDir))
        {
            Directory.Delete(modelDir, recursive: true);
        }
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
}
