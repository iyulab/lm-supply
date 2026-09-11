using System.Diagnostics;

namespace LMSupply.Download;

/// <summary>
/// Manages the HuggingFace-compatible cache directory structure.
/// </summary>
public static class CacheManager
{
    private static readonly System.Text.Json.JsonSerializerOptions s_jsonOptions = new()
    {
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
        => IsCachedFile(GetModelFilePath(cacheDir, repoId, fileName, revision));

    /// <summary>
    /// Lists which of <paramref name="files"/> a download of <paramref name="repoId"/> would still have to
    /// fetch: the files that are not in the cache, or are cached only as Git LFS pointers.
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

        var directory = GetSubfolderDirectory(GetModelDirectory(cacheDir, repoId, revision), subfolder);
        return files.Where(file => !IsCachedFile(Path.Combine(directory, file))).ToList();
    }

    /// <summary>
    /// Whether a file holds real content: present, and not a Git LFS pointer. The downloader fetches a
    /// file exactly when this is false.
    /// </summary>
    internal static bool IsCachedFile(string filePath) => File.Exists(filePath) && !IsLfsPointerFile(filePath);

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
        if (!File.Exists(filePath))
            return false;

        var fileInfo = new FileInfo(filePath);
        if (fileInfo.Length > 1024)
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

            // Get the most recent snapshot
            var latestSnapshot = Directory.GetDirectories(snapshotsDir)
                .OrderByDescending(Directory.GetLastWriteTime)
                .FirstOrDefault();

            if (latestSnapshot == null)
                return null;

            // Calculate file list and total size
            var files = Directory.GetFiles(latestSnapshot, "*", SearchOption.AllDirectories);
            var totalSize = files.Sum(f => new FileInfo(f).Length);
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
            return System.Text.Json.JsonSerializer.Deserialize<ModelMetadata>(json, s_jsonOptions);
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
