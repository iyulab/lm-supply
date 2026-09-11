using System.Diagnostics;
using System.Net;
using System.Net.Http.Headers;
using System.Net.Http.Json;
using System.Text.Json;
using LMSupply.Download;
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

    // Encoder model file prefixes for encoder-decoder architectures (prefix-based matching)
    private static readonly string[] EncoderPrefixes =
    [
        "encoder_model",  // matches encoder_model.onnx, encoder_model_fp16.onnx, encoder_model_bnb4.onnx, etc.
        "encoder"         // matches encoder.onnx
    ];

    // Decoder model file prefixes for encoder-decoder architectures (priority order by variant)
    private static readonly (string Prefix, DecoderVariant Variant)[] DecoderPrefixes =
    [
        ("decoder_model_merged", DecoderVariant.Merged),       // merged encoder+decoder
        ("decoder_with_past_model", DecoderVariant.WithPast),  // KV-cache variant
        ("decoder_with_past", DecoderVariant.WithPast),        // alternate naming
        ("decoder_model", DecoderVariant.Standard),            // standalone decoder
        ("decoder", DecoderVariant.Standard)                   // minimal naming
    ];

    // Legacy exact-match patterns kept only for IsEncoderDecoderModel (architecture detection)
    private static readonly string[] EncoderPatterns =
    [
        "encoder_model.onnx",
        "encoder_model_quantized.onnx",
        "encoder_model_fp16.onnx",
        "encoder_model_int8.onnx",
        "encoder_model_int4.onnx",
        "encoder.onnx"
    ];

    private static readonly string[] DecoderPatterns =
    [
        "decoder_model_merged.onnx",
        "decoder_model.onnx",
        "decoder_with_past_model.onnx",
        "decoder.onnx"
    ];

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
        // MarianMT/OPUS-MT SentencePiece tokenizer files
        "source.spm",
        "target.spm",
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
        : this(cacheDir, hfToken, new HttpClientHandler { AutomaticDecompression = DecompressionMethods.All }, disposeHandler: true)
    {
    }

    /// <summary>Test seam: discovery over a caller-owned transport, which this service does not dispose.</summary>
    internal ModelDiscoveryService(string? cacheDir, string? hfToken, HttpMessageHandler handler)
        : this(cacheDir, hfToken, handler, disposeHandler: false)
    {
    }

    private ModelDiscoveryService(string? cacheDir, string? hfToken, HttpMessageHandler handler, bool disposeHandler)
    {
        _cacheDir = cacheDir;

        _httpClient = new HttpClient(handler, disposeHandler)
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

        if (LocalFilesOnly)
            return await ListCachedFilesAsync(repoId, revision, cancellationToken);

        // Check cache first
        var cached = await TryLoadFromCacheAsync(repoId, revision, ignoreExpiry: false, cancellationToken);
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
        catch (Exception ex)
        {
            Trace.TraceInformation($"[ModelDiscoveryService] Directory listing failed: {ex.Message}");
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
        preferences ??= ModelPreferences.ForCurrentHardware();

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

        // Detect architecture type
        var architecture = DetectArchitecture(onnxFiles);

        // Select ONNX files based on subfolder and preferences
        var selectedOnnxFiles = SelectOnnxFiles(onnxFiles, subfolder, preferences);

        // Find external data files
        var externalDataFiles = FindExternalDataFiles(allFiles, selectedOnnxFiles);

        // Find config files (prefer root, but also check subfolder)
        var configFiles = FindConfigFiles(allFiles, subfolder);

        // Classify all variants for informational purposes
        var variants = ClassifyVariants(onnxFiles);

        // Classify encoder-decoder files if applicable
        List<string> encoderFiles = [];
        List<string> decoderFiles = [];
        var decoderVariant = DecoderVariant.Standard;

        if (architecture == ModelArchitecture.EncoderDecoder)
        {
            var (encoders, decoders, variant) = ClassifyEncoderDecoderFiles(onnxFiles, preferences);
            encoderFiles = encoders;
            decoderFiles = decoders;
            decoderVariant = variant;

            // If explicit files specified in preferences, use those
            if (!string.IsNullOrEmpty(preferences.ExplicitEncoderFile))
            {
                var explicitEncoder = onnxFiles.FirstOrDefault(f =>
                    f.Path.EndsWith(preferences.ExplicitEncoderFile, StringComparison.OrdinalIgnoreCase));
                if (explicitEncoder is not null)
                    encoderFiles = [explicitEncoder.Path];
            }

            if (!string.IsNullOrEmpty(preferences.ExplicitDecoderFile))
            {
                var explicitDecoder = onnxFiles.FirstOrDefault(f =>
                    f.Path.EndsWith(preferences.ExplicitDecoderFile, StringComparison.OrdinalIgnoreCase));
                if (explicitDecoder is not null)
                {
                    decoderFiles = [explicitDecoder.Path];
                    decoderVariant = DetectDecoderVariant(explicitDecoder.Path);
                }
            }
        }

        // For encoder-decoder models, download only the selected encoder+decoder pair,
        // not all quantization variants discovered by SelectOnnxFiles.
        var onnxFilesForDownload = architecture == ModelArchitecture.EncoderDecoder &&
                                   (encoderFiles.Count > 0 || decoderFiles.Count > 0)
            ? encoderFiles.Concat(decoderFiles).ToList()
            : selectedOnnxFiles;

        return new ModelDiscoveryResult
        {
            RepoId = repoId,
            Subfolder = subfolder,
            OnnxFiles = onnxFilesForDownload,
            ExternalDataFiles = externalDataFiles,
            ConfigFiles = configFiles,
            AvailableVariants = variants,
            Architecture = architecture,
            EncoderFiles = encoderFiles,
            DecoderFiles = decoderFiles,
            DetectedDecoderVariant = decoderVariant
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
                Quantization.Quant4 => "int4",  // folder names typically use int4/int8 convention
                Quantization.Quant8 => "int8",
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
    /// Checks if the file list contains an encoder-decoder model pattern.
    /// </summary>
    public static bool IsEncoderDecoderModel(IEnumerable<string> filePaths)
    {
        var files = filePaths.ToList();
        var hasEncoder = files.Any(f => EncoderPatterns.Any(p =>
            f.EndsWith(p, StringComparison.OrdinalIgnoreCase)));
        var hasDecoder = files.Any(f => DecoderPatterns.Any(p =>
            f.EndsWith(p, StringComparison.OrdinalIgnoreCase)));
        return hasEncoder && hasDecoder;
    }

    /// <summary>
    /// Detects the architecture type from a list of ONNX files.
    /// </summary>
    public static ModelArchitecture DetectArchitecture(List<RepoFile> onnxFiles)
    {
        if (IsDiffusionPipelineModel(onnxFiles))
            return ModelArchitecture.DiffusionPipeline;

        var paths = onnxFiles.Select(f => f.Path).ToList();
        if (IsEncoderDecoderModel(paths))
            return ModelArchitecture.EncoderDecoder;

        if (onnxFiles.Count == 1)
            return ModelArchitecture.SingleModel;

        return ModelArchitecture.Unknown;
    }

    /// <summary>
    /// Classifies encoder and decoder files from a list of ONNX files using prefix-based matching.
    /// Supports all quantization suffixes (_fp16, _int8, _int4, _bnb4, _q4, _q4f16, _uint8, etc.).
    /// </summary>
    private static (List<string> encoderFiles, List<string> decoderFiles, DecoderVariant variant)
        ClassifyEncoderDecoderFiles(List<RepoFile> onnxFiles, ModelPreferences preferences)
    {
        // Classify all files by role using prefix matching
        var allEncoders = new List<RepoFile>();
        var decodersByVariant = new Dictionary<DecoderVariant, List<RepoFile>>();

        foreach (var file in onnxFiles)
        {
            var baseName = Path.GetFileNameWithoutExtension(file.FileName);

            // Check encoder prefixes (longest match first to avoid "encoder" matching "encoder_model_*")
            if (EncoderPrefixes.Any(prefix => baseName.Equals(prefix, StringComparison.OrdinalIgnoreCase) ||
                                               baseName.StartsWith(prefix + "_", StringComparison.OrdinalIgnoreCase)))
            {
                allEncoders.Add(file);
                continue;
            }

            // Check decoder prefixes (ordered by specificity: merged > with_past > standard)
            foreach (var (prefix, variant) in DecoderPrefixes)
            {
                if (baseName.Equals(prefix, StringComparison.OrdinalIgnoreCase) ||
                    baseName.StartsWith(prefix + "_", StringComparison.OrdinalIgnoreCase))
                {
                    if (!decodersByVariant.TryGetValue(variant, out var list))
                    {
                        list = [];
                        decodersByVariant[variant] = list;
                    }
                    list.Add(file);
                    break; // Don't match shorter prefixes
                }
            }
        }

        // Select best decoder variant type, then best quantization within that variant
        string? selectedDecoder = null;
        var selectedVariant = DecoderVariant.Standard;

        // Check explicit override first
        if (!string.IsNullOrEmpty(preferences.ExplicitDecoderFile))
        {
            var explicitMatch = onnxFiles.FirstOrDefault(f =>
                f.Path.EndsWith(preferences.ExplicitDecoderFile, StringComparison.OrdinalIgnoreCase));
            if (explicitMatch is not null)
            {
                selectedDecoder = explicitMatch.Path;
                selectedVariant = DetectDecoderVariant(selectedDecoder);
            }
        }

        if (selectedDecoder is null)
        {
            foreach (var variant in preferences.DecoderVariantPriority)
            {
                if (decodersByVariant.TryGetValue(variant, out var candidates) && candidates.Count > 0)
                {
                    // Apply quantization preference to select best variant
                    var bestFiles = SelectBestQuantizationVariants(candidates, preferences);
                    if (bestFiles.Count > 0)
                    {
                        selectedDecoder = bestFiles[0];
                        selectedVariant = variant;
                        break;
                    }
                }
            }
        }

        // Select encoder with matching quantization
        string? selectedEncoder = null;
        if (allEncoders.Count > 0)
        {
            if (preferences.RequireMatchedQuantization && selectedDecoder is not null)
            {
                selectedEncoder = SelectMatchingEncoder(allEncoders, selectedDecoder, preferences);
            }
            else
            {
                var bestEncoders = SelectBestQuantizationVariants(allEncoders, preferences);
                selectedEncoder = bestEncoders.Count > 0 ? bestEncoders[0] : null;
            }
        }

        return (
            selectedEncoder is not null ? [selectedEncoder] : [],
            selectedDecoder is not null ? [selectedDecoder] : [],
            selectedVariant
        );
    }

    /// <summary>
    /// Selects an encoder that matches the quantization of the selected decoder.
    /// Falls back to quantization preference if no exact match.
    /// </summary>
    private static string? SelectMatchingEncoder(List<RepoFile> encoderFiles, string selectedDecoder, ModelPreferences preferences)
    {
        if (encoderFiles.Count == 0)
            return null;

        var decoderQuant = DetectFileQuantization(Path.GetFileNameWithoutExtension(selectedDecoder));

        if (decoderQuant is not null)
        {
            // Find encoder with same quantization suffix
            var match = encoderFiles.FirstOrDefault(f =>
                DetectFileQuantization(Path.GetFileNameWithoutExtension(f.FileName)) == decoderQuant);
            if (match is not null)
                return match.Path;
        }
        else
        {
            // Decoder is FP32, prefer encoder without quantization
            var match = encoderFiles.FirstOrDefault(f =>
                !HasQuantizationSuffix(f.FileName));
            if (match is not null)
                return match.Path;
        }

        // Fall back to preference-based selection
        var best = SelectBestQuantizationVariants(encoderFiles, preferences);
        return best.Count > 0 ? best[0] : encoderFiles[0].Path;
    }

    /// <summary>
    /// Detects the quantization suffix from a file name (without extension).
    /// Returns the suffix string (e.g., "_fp16", "_bnb4") or null for FP32.
    /// </summary>
    private static string? DetectFileQuantization(string nameWithoutExtension)
    {
        foreach (var suffix in QuantizationSuffixes)
        {
            if (nameWithoutExtension.EndsWith(suffix, StringComparison.OrdinalIgnoreCase))
                return suffix;
        }
        return null;
    }

    /// <summary>
    /// All known quantization suffixes, ordered longest-first to avoid partial matches.
    /// </summary>
    private static readonly string[] QuantizationSuffixes =
        ["_q4f16", "_quantized", "_uint8", "_bnb4", "_int4", "_int8", "_fp16", "_q4", "_q8"];

    /// <summary>
    /// Detects the decoder variant type from a decoder file path.
    /// </summary>
    private static DecoderVariant DetectDecoderVariant(string? decoderPath)
    {
        if (string.IsNullOrEmpty(decoderPath))
            return DecoderVariant.Standard;

        var fileName = Path.GetFileName(decoderPath);

        if (fileName.Contains("merged", StringComparison.OrdinalIgnoreCase))
            return DecoderVariant.Merged;

        if (fileName.Contains("with_past", StringComparison.OrdinalIgnoreCase))
            return DecoderVariant.WithPast;

        return DecoderVariant.Standard;
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
    /// Test-accessible wrapper for SelectBestQuantizationVariants.
    /// </summary>
    internal static List<string> SelectBestVariantsForTest(
        List<RepoFile> candidates, ModelPreferences preferences)
        => SelectBestQuantizationVariants(candidates, preferences);

    /// <summary>
    /// Test-accessible wrapper for ClassifyEncoderDecoderFiles.
    /// Returns (encoderFiles, decoderFiles, variant) without requiring network calls.
    /// </summary>
    internal static (List<string> EncoderFiles, List<string> DecoderFiles, DecoderVariant Variant)
        ClassifyEncoderDecoderForTest(List<RepoFile> onnxFiles, ModelPreferences preferences)
        => ClassifyEncoderDecoderFiles(onnxFiles, preferences);

    /// <summary>
    /// Selects the best quantization variants based on preferences.
    /// Supports all common suffixes: _int4, _int8, _fp16, _bnb4, _q4, _q4f16, _uint8, _quantized.
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
                if (quant == Quantization.Default)
                {
                    bestVariant = variants.FirstOrDefault(v => !HasQuantizationSuffix(v.FileName));
                }
                else
                {
                    // Try all suffixes that map to this quantization level
                    var suffixes = QuantizationLevelSuffixes(quant);
                    foreach (var suffix in suffixes)
                    {
                        bestVariant = variants.FirstOrDefault(v =>
                            Path.GetFileNameWithoutExtension(v.FileName)
                                .EndsWith(suffix, StringComparison.OrdinalIgnoreCase));
                        if (bestVariant is not null)
                            break;
                    }
                }

                if (bestVariant is not null)
                    break;
            }

            // Fallback: size-based selection when no suffix match
            // PreferLowMemory (Low tier) → smallest file, otherwise → largest file
            if (bestVariant is null && variants.Count > 0)
            {
                var withSize = variants.Where(v => v.Size > 0).ToList();
                var pool = withSize.Count > 0 ? withSize : variants;
                bestVariant = preferences.PreferLowMemory
                    ? pool.MinBy(v => v.Size)
                    : pool.MaxBy(v => v.Size);
            }

            bestVariant ??= variants.First();
            selected.Add(bestVariant.Path);
        }

        return selected;
    }

    /// <summary>
    /// Returns all file suffixes that correspond to a given quantization level.
    /// Ordered by preference (standard naming first).
    /// </summary>
    private static string[] QuantizationLevelSuffixes(Quantization quant) => quant switch
    {
        Quantization.Fp16 => ["_fp16"],
        Quantization.Quant8 => ["_int8", "_uint8", "_quantized", "_q8"],
        Quantization.Quant4 => ["_int4", "_bnb4", "_q4", "_q4f16"],
        _ => []
    };

    /// <summary>
    /// Gets the base model name without quantization suffix.
    /// </summary>
    private static string GetBaseModelName(string fileName)
    {
        var name = Path.GetFileNameWithoutExtension(fileName);

        // Remove common quantization suffixes (uses centralized list, longest-first)
        foreach (var suffix in QuantizationSuffixes)
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
        var name = Path.GetFileNameWithoutExtension(fileName);
        return QuantizationSuffixes.Any(s => name.EndsWith(s, StringComparison.OrdinalIgnoreCase));
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
            // Check for .onnx.data file (HuggingFace format: model.onnx → model.onnx.data)
            var dotDataPath = onnxPath + ".data";
            if (allFiles.Any(f => f.Path.Equals(dotDataPath, StringComparison.OrdinalIgnoreCase)))
            {
                dataFiles.Add(dotDataPath);
            }

            // Check for .onnx_data file (alternative format: model.onnx → model.onnx_data)
            var underscoreDataPath = onnxPath + "_data";
            if (allFiles.Any(f => f.Path.Equals(underscoreDataPath, StringComparison.OrdinalIgnoreCase)))
            {
                dataFiles.Add(underscoreDataPath);
            }

            // Check for chunked data files (.onnx_data_0, .onnx_data_1, ...)
            var dataPathPrefix = underscoreDataPath + "_";
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
            Quant8 = paths.Where(p => p.Contains("int8", StringComparison.OrdinalIgnoreCase)).ToList(),
            Quant4 = paths.Where(p => p.Contains("int4", StringComparison.OrdinalIgnoreCase)).ToList(),
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

    /// <summary>
    /// When true, the repository's file list comes from the local cache only and no request is made. Set
    /// by a <see cref="HuggingFaceDownloader"/> created with <c>localFilesOnly</c>.
    /// </summary>
    internal bool LocalFilesOnly { get; init; }

    // Local files only: the cached file list at any age — its 24-hour expiry exists to pick up changes to
    // the repository, which cannot be fetched now — and failing that, the manifest the downloader wrote
    // beside the files. With neither, the model was never downloaded into this cache.
    private async Task<IReadOnlyList<RepoFile>> ListCachedFilesAsync(
        string repoId,
        string revision,
        CancellationToken cancellationToken)
    {
        var cached = await TryLoadFromCacheAsync(repoId, revision, ignoreExpiry: true, cancellationToken);
        if (cached is not null)
            return cached;

        if (_cacheDir is not null)
        {
            var manifest = await DownloadManifest.ReadAsync(CacheManager.GetModelDirectory(_cacheDir, repoId, revision));
            if (manifest is { Files.Count: > 0 })
                return [.. manifest.Files.Select(f => new RepoFile { Path = f.Path, Type = "file", Size = f.Size })];
        }

        throw new ModelNotFoundException(
            $"Model '{repoId}' is not in the local cache and downloads are disabled.", repoId);
    }

    private async Task<IReadOnlyList<RepoFile>?> TryLoadFromCacheAsync(
        string repoId,
        string revision,
        bool ignoreExpiry,
        CancellationToken cancellationToken)
    {
        var cachePath = GetCachePath(repoId, revision);
        if (string.IsNullOrEmpty(cachePath) || !File.Exists(cachePath))
            return null;

        try
        {
            var fileInfo = new FileInfo(cachePath);
            // Cache expires after 24 hours
            if (!ignoreExpiry && DateTime.UtcNow - fileInfo.LastWriteTimeUtc > TimeSpan.FromHours(24))
                return null;

            await using var stream = File.OpenRead(cachePath);
            return await JsonSerializer.DeserializeAsync<List<RepoFile>>(stream, cancellationToken: cancellationToken);
        }
        catch (Exception ex)
        {
            Trace.TraceInformation($"[ModelDiscoveryService] File list cache load failed: {ex.Message}");
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
        catch (Exception ex)
        {
            Trace.TraceInformation($"[ModelDiscoveryService] File list cache write failed: {ex.Message}");
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
