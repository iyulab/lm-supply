using System.Net;
using System.Net.Http.Headers;
using System.Net.Http.Json;
using System.Text.Json;
using LMSupply.Exceptions;

namespace LMSupply.Core.Download;

/// <summary>
/// Service for automatically discovering model files in HuggingFace repositories.
/// Eliminates the need to hardcode subfolder paths and file lists.
/// </summary>
public sealed class ModelDiscoveryService : IDisposable
{
    private readonly HttpClient _httpClient;
    private readonly string? _cacheDir;
    private bool _disposed;

    private const string ApiBaseUrl = "https://huggingface.co/api/models";

    // Common subfolder patterns in priority order
    private static readonly string[] PreferredSubfolders =
        ["onnx", "cpu", "cpu-int4", "cpu-int8", "default"];

    // Diffusion pipeline model directories (Stable Diffusion, LCM, etc.)
    private static readonly HashSet<string> DiffusionPipelineDirectories = new(StringComparer.OrdinalIgnoreCase)
    {
        "text_encoder", "text_encoder_2", "unet", "vae_decoder", "vae_encoder", "vae"
    };

    // Config and tokenizer files that are typically in root or pipeline subdirectories
    private static readonly HashSet<string> ConfigFileNames = new(StringComparer.OrdinalIgnoreCase)
    {
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "vocab.txt",
        "merges.txt",
        "special_tokens_map.json",
        "sentencepiece.bpe.model",
        "generation_config.json",
        "preprocessor_config.json",
        "genai_config.json",
        // Diffusion pipeline specific
        "scheduler_config.json",
        "model_index.json"
    };

    /// <summary>
    /// Initializes a new ModelDiscoveryService.
    /// </summary>
    /// <param name="cacheDir">Optional cache directory for storing discovery results.</param>
    /// <param name="hfToken">Optional HuggingFace API token for private repositories.</param>
    public ModelDiscoveryService(string? cacheDir = null, string? hfToken = null)
    {
        _cacheDir = cacheDir;

        var handler = new HttpClientHandler
        {
            AutomaticDecompression = DecompressionMethods.All
        };

        _httpClient = new HttpClient(handler)
        {
            Timeout = TimeSpan.FromSeconds(30)
        };

        _httpClient.DefaultRequestHeaders.UserAgent.Add(
            new ProductInfoHeaderValue("LMSupply", "1.0"));

        // Use token from parameter or environment variable
        var token = hfToken ?? Environment.GetEnvironmentVariable("HF_TOKEN");
        if (!string.IsNullOrEmpty(token))
        {
            _httpClient.DefaultRequestHeaders.Authorization =
                new AuthenticationHeaderValue("Bearer", token);
        }
    }

    /// <summary>
    /// Lists all files in a HuggingFace repository.
    /// </summary>
    /// <param name="repoId">The repository ID (e.g., "microsoft/Phi-3-mini-4k-instruct-onnx").</param>
    /// <param name="revision">The revision/branch (default: "main").</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>List of files in the repository.</returns>
    public async Task<IReadOnlyList<RepoFile>> ListRepositoryFilesAsync(
        string repoId,
        string revision = "main",
        CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(repoId);

        // Check cache first
        var cached = await TryLoadFromCacheAsync(repoId, revision, cancellationToken);
        if (cached is not null)
            return cached;

        var url = $"{ApiBaseUrl}/{repoId}/tree/{revision}";

        try
        {
            var response = await _httpClient.GetAsync(url, cancellationToken);

            if (response.StatusCode == HttpStatusCode.NotFound)
            {
                throw new ModelNotFoundException(
                    $"Repository '{repoId}' not found on HuggingFace.",
                    repoId);
            }

            response.EnsureSuccessStatusCode();

            var files = await response.Content.ReadFromJsonAsync<List<RepoFile>>(cancellationToken)
                ?? throw new InvalidOperationException($"Failed to parse repository file list for '{repoId}'");

            // Recursively fetch subdirectories
            var allFiles = new List<RepoFile>();
            foreach (var file in files)
            {
                if (file.IsFile)
                {
                    allFiles.Add(file);
                }
                else if (file.IsDirectory)
                {
                    var subFiles = await ListDirectoryFilesAsync(repoId, file.Path, revision, cancellationToken);
                    allFiles.AddRange(subFiles);
                }
            }

            // Cache the result
            await SaveToCacheAsync(repoId, revision, allFiles, cancellationToken);

            return allFiles;
        }
        catch (HttpRequestException ex) when (ex.StatusCode == HttpStatusCode.Unauthorized)
        {
            throw new UnauthorizedAccessException(
                $"Access denied to repository '{repoId}'. Set HF_TOKEN environment variable for private repositories.",
                ex);
        }
    }

    /// <summary>
    /// Lists files in a specific directory of a repository.
    /// </summary>
    private async Task<List<RepoFile>> ListDirectoryFilesAsync(
        string repoId,
        string path,
        string revision,
        CancellationToken cancellationToken)
    {
        var url = $"{ApiBaseUrl}/{repoId}/tree/{revision}/{path}";

        try
        {
            var response = await _httpClient.GetAsync(url, cancellationToken);
            if (!response.IsSuccessStatusCode)
                return [];

            var files = await response.Content.ReadFromJsonAsync<List<RepoFile>>(cancellationToken) ?? [];

            var result = new List<RepoFile>();
            foreach (var file in files)
            {
                if (file.IsFile)
                {
                    result.Add(file);
                }
                else if (file.IsDirectory)
                {
                    var subFiles = await ListDirectoryFilesAsync(repoId, file.Path, revision, cancellationToken);
                    result.AddRange(subFiles);
                }
            }

            return result;
        }
        catch
        {
            return [];
        }
    }

    /// <summary>
    /// Automatically discovers model files from a HuggingFace repository.
    /// </summary>
    /// <param name="repoId">The repository ID.</param>
    /// <param name="preferences">Optional preferences for model selection.</param>
    /// <param name="revision">The revision/branch (default: "main").</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Discovery result with files to download.</returns>
    public async Task<ModelDiscoveryResult> DiscoverModelAsync(
        string repoId,
        ModelPreferences? preferences = null,
        string revision = "main",
        CancellationToken cancellationToken = default)
    {
        preferences ??= ModelPreferences.Default;

        var allFiles = await ListRepositoryFilesAsync(repoId, revision, cancellationToken);

        // Filter ONNX files
        var onnxFiles = allFiles
            .Where(f => f.Path.EndsWith(".onnx", StringComparison.OrdinalIgnoreCase))
            .ToList();

        if (onnxFiles.Count == 0)
        {
            throw new ModelNotFoundException(
                $"No ONNX files found in repository '{repoId}'. " +
                "Ensure the repository contains ONNX model files.",
                repoId);
        }

        // Detect subfolder
        var subfolder = preferences.PreferredSubfolder ?? DetectOnnxSubfolder(onnxFiles, preferences);

        // Select ONNX files based on subfolder and preferences
        var selectedOnnxFiles = SelectOnnxFiles(onnxFiles, subfolder, preferences);

        // Find external data files
        var externalDataFiles = FindExternalDataFiles(allFiles, selectedOnnxFiles);

        // Find config files (prefer root, but also check subfolder)
        var configFiles = FindConfigFiles(allFiles, subfolder);

        // Classify all variants for informational purposes
        var variants = ClassifyVariants(onnxFiles);

        return new ModelDiscoveryResult
        {
            RepoId = repoId,
            Subfolder = subfolder,
            OnnxFiles = selectedOnnxFiles,
            ExternalDataFiles = externalDataFiles,
            ConfigFiles = configFiles,
            AvailableVariants = variants
        };
    }

    /// <summary>
    /// Detects the best subfolder containing ONNX files.
    /// </summary>
    private static string? DetectOnnxSubfolder(List<RepoFile> onnxFiles, ModelPreferences preferences)
    {
        // Group ONNX files by their directory
        var folderGroups = onnxFiles
            .GroupBy(f => f.Directory ?? "")
            .ToDictionary(g => g.Key, g => g.ToList());

        // If all ONNX files are in root, return null
        if (folderGroups.Count == 1 && folderGroups.ContainsKey(""))
            return null;

        // Look for preferred subfolders first
        foreach (var preferred in PreferredSubfolders)
        {
            var match = folderGroups.Keys
                .FirstOrDefault(k => k.Equals(preferred, StringComparison.OrdinalIgnoreCase) ||
                                     k.EndsWith("/" + preferred, StringComparison.OrdinalIgnoreCase));
            if (match is not null)
                return match;
        }

        // Look for device-specific folders based on preferences
        var deviceKeywords = preferences.PreferredProvider switch
        {
            ExecutionProvider.Cuda => new[] { "cuda", "gpu" },
            ExecutionProvider.DirectML => new[] { "directml", "dml" },
            ExecutionProvider.CoreML => new[] { "coreml" },
            _ => new[] { "cpu" }
        };

        foreach (var keyword in deviceKeywords)
        {
            var match = folderGroups.Keys
                .FirstOrDefault(k => k.Contains(keyword, StringComparison.OrdinalIgnoreCase));
            if (match is not null)
                return match;
        }

        // Look for quantization-specific folders based on preferences
        foreach (var quant in preferences.QuantizationPriority)
        {
            var keyword = quant switch
            {
                Quantization.Int4 => "int4",
                Quantization.Int8 => "int8",
                Quantization.Fp16 => "fp16",
                _ => null
            };

            if (keyword is not null)
            {
                var match = folderGroups.Keys
                    .FirstOrDefault(k => k.Contains(keyword, StringComparison.OrdinalIgnoreCase));
                if (match is not null)
                    return match;
            }
        }

        // Return the folder with the most ONNX files
        return folderGroups
            .Where(g => g.Key != "")
            .OrderByDescending(g => g.Value.Count)
            .FirstOrDefault().Key;
    }

    /// <summary>
    /// Selects ONNX files to download based on subfolder and preferences.
    /// </summary>
    /// <remarks>
    /// For diffusion pipeline models (Stable Diffusion, LCM, etc.), ONNX files are spread
    /// across multiple subdirectories (text_encoder, unet, vae_decoder). This method detects
    /// such structures and selects all required pipeline components.
    /// </remarks>
    private static List<string> SelectOnnxFiles(
        List<RepoFile> onnxFiles,
        string? subfolder,
        ModelPreferences preferences)
    {
        // If specific files are requested, look for those
        if (preferences.PreferredOnnxFiles.Count > 0)
        {
            var requested = new List<string>();
            foreach (var preferred in preferences.PreferredOnnxFiles)
            {
                var match = onnxFiles.FirstOrDefault(f =>
                    f.FileName.Equals(preferred, StringComparison.OrdinalIgnoreCase) ||
                    f.Path.Equals(preferred, StringComparison.OrdinalIgnoreCase));

                if (match is not null)
                    requested.Add(match.Path);
            }

            if (requested.Count > 0)
                return requested;
        }

        // Check if this is a diffusion pipeline model
        if (IsDiffusionPipelineModel(onnxFiles))
        {
            return SelectDiffusionPipelineFiles(onnxFiles, preferences);
        }

        // Filter by subfolder for non-diffusion models
        var candidates = subfolder is null
            ? onnxFiles.Where(f => f.Directory is null).ToList()
            : onnxFiles.Where(f => f.Directory?.Equals(subfolder, StringComparison.OrdinalIgnoreCase) == true).ToList();

        // If subfolder has no files, fall back to all files
        if (candidates.Count == 0)
            candidates = onnxFiles;

        // Select best quantization variant
        return SelectBestQuantizationVariants(candidates, preferences);
    }

    /// <summary>
    /// Checks if the repository contains a diffusion pipeline model structure.
    /// </summary>
    private static bool IsDiffusionPipelineModel(List<RepoFile> onnxFiles)
    {
        // A diffusion pipeline model has ONNX files in at least 2 of these directories:
        // text_encoder, unet, vae_decoder (or vae)
        var directories = onnxFiles
            .Where(f => f.Directory is not null)
            .Select(f => f.Directory!.Split('/')[0]) // Get top-level directory
            .Distinct()
            .ToHashSet(StringComparer.OrdinalIgnoreCase);

        var pipelineComponentCount = DiffusionPipelineDirectories.Count(d => directories.Contains(d));
        return pipelineComponentCount >= 2;
    }

    /// <summary>
    /// Selects all required ONNX files for a diffusion pipeline model.
    /// </summary>
    private static List<string> SelectDiffusionPipelineFiles(
        List<RepoFile> onnxFiles,
        ModelPreferences preferences)
    {
        var selected = new List<string>();

        // Select files from each pipeline directory
        foreach (var pipelineDir in DiffusionPipelineDirectories)
        {
            var dirFiles = onnxFiles
                .Where(f => f.Directory?.Equals(pipelineDir, StringComparison.OrdinalIgnoreCase) == true ||
                            f.Directory?.StartsWith(pipelineDir + "/", StringComparison.OrdinalIgnoreCase) == true)
                .ToList();

            if (dirFiles.Count > 0)
            {
                // Select best quantization variant for this component
                selected.AddRange(SelectBestQuantizationVariants(dirFiles, preferences));
            }
        }

        // Also include any root-level ONNX files if present
        var rootFiles = onnxFiles.Where(f => f.Directory is null).ToList();
        if (rootFiles.Count > 0)
        {
            selected.AddRange(SelectBestQuantizationVariants(rootFiles, preferences));
        }

        return selected;
    }

    /// <summary>
    /// Selects the best quantization variants based on preferences.
    /// </summary>
    private static List<string> SelectBestQuantizationVariants(
        List<RepoFile> candidates,
        ModelPreferences preferences)
    {
        // Group by base name (without quantization suffix)
        var groups = candidates
            .GroupBy(f => GetBaseModelName(f.FileName))
            .ToDictionary(g => g.Key, g => g.ToList());

        var selected = new List<string>();

        foreach (var (baseName, variants) in groups)
        {
            // Find best variant for this base model
            RepoFile? bestVariant = null;

            foreach (var quant in preferences.QuantizationPriority)
            {
                var suffix = quant switch
                {
                    Quantization.Int4 => "_int4",
                    Quantization.Int8 => "_int8",
                    Quantization.Fp16 => "_fp16",
                    _ => ""
                };

                bestVariant = quant == Quantization.Default
                    ? variants.FirstOrDefault(v => !HasQuantizationSuffix(v.FileName))
                    : variants.FirstOrDefault(v => v.FileName.Contains(suffix, StringComparison.OrdinalIgnoreCase));

                if (bestVariant is not null)
                    break;
            }

            // Fall back to first variant if no match
            bestVariant ??= variants.First();
            selected.Add(bestVariant.Path);
        }

        return selected;
    }

    /// <summary>
    /// Gets the base model name without quantization suffix.
    /// </summary>
    private static string GetBaseModelName(string fileName)
    {
        var name = Path.GetFileNameWithoutExtension(fileName);

        // Remove common suffixes
        foreach (var suffix in new[] { "_int4", "_int8", "_fp16", "_uint8", "_quantized", "_q4", "_q8" })
        {
            if (name.EndsWith(suffix, StringComparison.OrdinalIgnoreCase))
                return name[..^suffix.Length];
        }

        return name;
    }

    /// <summary>
    /// Checks if a file name has a quantization suffix.
    /// </summary>
    private static bool HasQuantizationSuffix(string fileName)
    {
        var suffixes = new[] { "_int4", "_int8", "_fp16", "_uint8", "_quantized", "_q4", "_q8" };
        return suffixes.Any(s => fileName.Contains(s, StringComparison.OrdinalIgnoreCase));
    }

    /// <summary>
    /// Finds external data files associated with ONNX files.
    /// </summary>
    private static List<string> FindExternalDataFiles(
        IReadOnlyList<RepoFile> allFiles,
        List<string> onnxFiles)
    {
        var dataFiles = new List<string>();

        foreach (var onnxPath in onnxFiles)
        {
            // Check for .onnx_data file
            var dataPath = onnxPath + "_data";
            if (allFiles.Any(f => f.Path.Equals(dataPath, StringComparison.OrdinalIgnoreCase)))
            {
                dataFiles.Add(dataPath);
            }

            // Check for chunked data files (.onnx_data_0, .onnx_data_1, ...)
            var dataPathPrefix = dataPath + "_";
            var chunks = allFiles
                .Where(f => f.Path.StartsWith(dataPathPrefix, StringComparison.OrdinalIgnoreCase))
                .OrderBy(f => f.Path)
                .Select(f => f.Path);

            dataFiles.AddRange(chunks);
        }

        return dataFiles;
    }

    // Common subdirectories for diffusion pipeline models (all config-containing directories)
    private static readonly string[] PipelineSubdirectories =
        ["tokenizer", "scheduler", "feature_extractor", "text_encoder", "text_encoder_2",
         "safety_checker", "unet", "vae_decoder", "vae_encoder", "vae"];

    /// <summary>
    /// Finds configuration and tokenizer files.
    /// </summary>
    /// <remarks>
    /// For diffusion pipeline models (LCM, Stable Diffusion, etc.), config files are often
    /// located in subdirectories like tokenizer/, scheduler/, etc. This method searches
    /// all relevant locations including root, main subfolder, and common pipeline subdirectories.
    /// </remarks>
    private static List<string> FindConfigFiles(IReadOnlyList<RepoFile> allFiles, string? subfolder)
    {
        var configFiles = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

        foreach (var fileName in ConfigFileNames)
        {
            // 1. Look in root
            var rootPath = allFiles.FirstOrDefault(f =>
                f.Path.Equals(fileName, StringComparison.OrdinalIgnoreCase));
            if (rootPath is not null)
            {
                configFiles.Add(rootPath.Path);
            }

            // 2. Look in main subfolder if specified
            if (subfolder is not null)
            {
                var subPath = $"{subfolder}/{fileName}";
                var subFile = allFiles.FirstOrDefault(f =>
                    f.Path.Equals(subPath, StringComparison.OrdinalIgnoreCase));
                if (subFile is not null)
                {
                    configFiles.Add(subFile.Path);
                }
            }

            // 3. Look in common pipeline subdirectories (for diffusion models)
            foreach (var pipelineDir in PipelineSubdirectories)
            {
                var pipelinePath = $"{pipelineDir}/{fileName}";
                var pipelineFile = allFiles.FirstOrDefault(f =>
                    f.Path.Equals(pipelinePath, StringComparison.OrdinalIgnoreCase));
                if (pipelineFile is not null)
                {
                    configFiles.Add(pipelineFile.Path);
                }
            }
        }

        return configFiles.ToList();
    }

    /// <summary>
    /// Classifies all ONNX variants by category.
    /// </summary>
    private static ModelVariants ClassifyVariants(List<RepoFile> onnxFiles)
    {
        var paths = onnxFiles.Select(f => f.Path).ToList();

        return new ModelVariants
        {
            Default = paths.Where(p => !HasQuantizationSuffix(Path.GetFileName(p))).ToList(),
            Fp16 = paths.Where(p => p.Contains("fp16", StringComparison.OrdinalIgnoreCase)).ToList(),
            Int8 = paths.Where(p => p.Contains("int8", StringComparison.OrdinalIgnoreCase)).ToList(),
            Int4 = paths.Where(p => p.Contains("int4", StringComparison.OrdinalIgnoreCase)).ToList(),
            Cpu = paths.Where(p => p.Contains("cpu", StringComparison.OrdinalIgnoreCase)).ToList(),
            Cuda = paths.Where(p => p.Contains("cuda", StringComparison.OrdinalIgnoreCase) ||
                                    p.Contains("gpu", StringComparison.OrdinalIgnoreCase)).ToList()
        };
    }

    #region Caching

    private string GetCachePath(string repoId, string revision)
    {
        if (_cacheDir is null)
            return string.Empty;

        var sanitizedRepo = repoId.Replace('/', '_').Replace('\\', '_');
        return Path.Combine(_cacheDir, ".discovery-cache", $"{sanitizedRepo}_{revision}.json");
    }

    private async Task<IReadOnlyList<RepoFile>?> TryLoadFromCacheAsync(
        string repoId,
        string revision,
        CancellationToken cancellationToken)
    {
        var cachePath = GetCachePath(repoId, revision);
        if (string.IsNullOrEmpty(cachePath) || !File.Exists(cachePath))
            return null;

        try
        {
            var fileInfo = new FileInfo(cachePath);
            // Cache expires after 24 hours
            if (DateTime.UtcNow - fileInfo.LastWriteTimeUtc > TimeSpan.FromHours(24))
                return null;

            await using var stream = File.OpenRead(cachePath);
            return await JsonSerializer.DeserializeAsync<List<RepoFile>>(stream, cancellationToken: cancellationToken);
        }
        catch
        {
            return null;
        }
    }

    private async Task SaveToCacheAsync(
        string repoId,
        string revision,
        List<RepoFile> files,
        CancellationToken cancellationToken)
    {
        var cachePath = GetCachePath(repoId, revision);
        if (string.IsNullOrEmpty(cachePath))
            return;

        try
        {
            var dir = Path.GetDirectoryName(cachePath);
            if (!string.IsNullOrEmpty(dir))
                Directory.CreateDirectory(dir);

            await using var stream = File.Create(cachePath);
            await JsonSerializer.SerializeAsync(stream, files, cancellationToken: cancellationToken);
        }
        catch
        {
            // Ignore cache write failures
        }
    }

    #endregion

    public void Dispose()
    {
        if (_disposed) return;
        _httpClient.Dispose();
        _disposed = true;
    }
}
