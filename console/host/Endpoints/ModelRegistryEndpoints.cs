using LMSupply.Console.Host.Infrastructure;
using LMSupply.Console.Host.Services;
using LMSupply.Captioner;
using LMSupply.Detector;
using LMSupply.Download;
using LMSupply.Embedder;
using LMSupply.Ocr;
using LMSupply.Reranker;
using LMSupply.Segmenter;
using LMSupply.Synthesizer;
using LMSupply.Transcriber;
using LMSupply.Generator;
using LMSupply.Generator.Gguf;
using LMSupply.ImageGenerator.Models;

namespace LMSupply.Console.Host.Endpoints;

/// <summary>
/// Provides model registry information for all model types.
/// Maps aliases (default, fast, quality) to actual HuggingFace repo IDs.
/// </summary>
public static class ModelRegistryEndpoints
{
    public static void MapModelRegistryEndpoints(this WebApplication app)
    {
        var group = app.MapGroup("/api/registry")
            .WithTags("Registry")
            ;

        // GET /api/registry/models - Get all model types with their aliases
        group.MapGet("/models", async (CacheService cache) =>
        {
            var cachedModels = cache.GetCachedModels();
            var cachedRepoIds = cachedModels.Select(m => m.RepoId).ToHashSet(StringComparer.OrdinalIgnoreCase);

            var registry = new ModelRegistry
            {
                ModelTypes = await GetAllModelTypesAsync(cachedRepoIds, cache.CacheDirectory)
            };

            return Results.Ok(registry);
        })
        .WithName("GetModelRegistry")
        .WithSummary("Get all model types with their aliases and repo IDs")
        .WithDescription("Returns a comprehensive list of all model types, their available aliases, and the actual HuggingFace repository IDs.");

        // GET /api/registry/models/{type} - Get models for specific type
        group.MapGet("/models/{type}", async (string type, CacheService cache) =>
        {
            var cachedModels = cache.GetCachedModels();
            var cachedRepoIds = cachedModels.Select(m => m.RepoId).ToHashSet(StringComparer.OrdinalIgnoreCase);

            var modelType = await GetModelTypeByNameAsync(type, cachedRepoIds, cache.CacheDirectory);
            if (modelType == null)
            {
                return ApiHelper.Error($"Unknown model type: {type}", "invalid_request_error", 404);
            }

            return Results.Ok(modelType);
        })
        .WithName("GetRegistryModelsByType")
        .WithSummary("Get models for a specific type")
        .WithDescription("Returns all aliases and repo IDs for a specific model type.");
    }

    private static async Task<List<ModelTypeInfo>> GetAllModelTypesAsync(HashSet<string> cachedRepoIds, string cacheDirectory)
    {
        return
        [
            await CreateGeneratorModelsAsync(cacheDirectory),
            CreateEmbedderModels(cachedRepoIds),
            CreateRerankerModels(cachedRepoIds),
            CreateTranscriberModels(cachedRepoIds),
            CreateSynthesizerModels(cachedRepoIds),
            CreateTranslatorModels(cachedRepoIds),
            CreateCaptionerModels(cachedRepoIds),
            CreateOcrModels(cacheDirectory),
            CreateDetectorModels(cachedRepoIds),
            CreateSegmenterModels(cachedRepoIds),
            CreateImageGeneratorModels(cachedRepoIds),
        ];
    }

    private static async Task<ModelTypeInfo?> GetModelTypeByNameAsync(string type, HashSet<string> cachedRepoIds, string cacheDirectory)
    {
        return type.ToLowerInvariant() switch
        {
            "generator" => await CreateGeneratorModelsAsync(cacheDirectory),
            "embedder" => CreateEmbedderModels(cachedRepoIds),
            "reranker" => CreateRerankerModels(cachedRepoIds),
            "transcriber" => CreateTranscriberModels(cachedRepoIds),
            "synthesizer" => CreateSynthesizerModels(cachedRepoIds),
            "translator" => CreateTranslatorModels(cachedRepoIds),
            "captioner" => CreateCaptionerModels(cachedRepoIds),
            "ocr" => CreateOcrModels(cacheDirectory),
            "detector" => CreateDetectorModels(cachedRepoIds),
            "segmenter" => CreateSegmenterModels(cachedRepoIds),
            "imagegenerator" or "imagegen" or "image" => CreateImageGeneratorModels(cachedRepoIds),
            _ => null
        };
    }

    // The generator catalogs are two: the GGUF aliases (served by llama-server) and the ONNX entries. Both are listed from the library, and
    // whether an alias is cached is asked of the library too — it knows which file a GGUF alias picks for this host's memory and what
    // "default" resolves to here, which no repository id this endpoint could name says.
    private static async Task<ModelTypeInfo> CreateGeneratorModelsAsync(string cacheDirectory)
    {
        var options = new GeneratorOptions { CacheDirectory = cacheDirectory };
        var models = new List<ModelAliasInfo>
        {
            // "default" is picked by the hardware at load time, so it names no one repository
            new()
            {
                AliasName = "default",
                RepoId = "default",
                Description = "Chosen by the hardware when it is loaded",
                IsCached = await LocalGenerator.IsModelDownloadedAsync("default", options)
            }
        };

        foreach (var alias in LocalGenerator.Registry.GetAliases().Where(a => a.Kind == AliasKind.System && a.Name != "auto"))
        {
            var entry = LocalGenerator.Registry.Resolve(alias.Name);
            models.Add(new ModelAliasInfo
            {
                AliasName = alias.Name,
                RepoId = entry.ModelId,
                Description = entry.Description ?? alias.Name,
                IsCached = await LocalGenerator.IsModelDownloadedAsync(alias.Name, options)
            });
        }

        foreach (var alias in GgufModelRegistry.GetAliases().Where(a => !a.Equals("gguf:auto", StringComparison.OrdinalIgnoreCase)))
        {
            var info = GgufModelRegistry.Resolve(alias)!;
            models.Add(new ModelAliasInfo
            {
                AliasName = alias,
                RepoId = info.RepoId,
                Description = info.DisplayName,
                IsCached = await LocalGenerator.IsModelDownloadedAsync(alias, options)
            });
        }

        return new ModelTypeInfo
        {
            Type = "generator",
            DisplayName = "Text Generator",
            Description = "LLM text generation models",
            Models = models
        };
    }

    private static ModelTypeInfo CreateEmbedderModels(HashSet<string> cachedRepoIds) => FromRegistry(
        "embedder", "Text Embedder", "Text embedding models for semantic search", LocalEmbedder.Registry.GetAvailableModels(), cachedRepoIds);

    private static ModelTypeInfo CreateRerankerModels(HashSet<string> cachedRepoIds) => FromRegistry(
        "reranker", "Reranker", "Cross-encoder reranking models", LocalReranker.Registry.GetAvailableModels(), cachedRepoIds);

    // The domains below list their registry, so the console names exactly what each Local* loads. They used to be
    // hand-written lists that had drifted: OCR named Tesseract, the detector YOLOv8, the segmenter SAM ViT and the
    // transcriber openai/whisper-*, none of which the library loads.
    private static ModelTypeInfo FromRegistry<T>(
        string type, string displayName, string description, IEnumerable<T> models, HashSet<string> cachedRepoIds)
        where T : IModelInfoBase => new()
    {
        Type = type,
        DisplayName = displayName,
        Description = description,
        Models = [.. models.Select(m =>
        {
            var repoId = m.Id.Split(':')[0];   // «org/name:variant» picks a file inside the repository
            return new ModelAliasInfo
            {
                AliasName = m.AliasName,
                RepoId = repoId,
                Description = m.Description ?? m.AliasName,
                IsCached = cachedRepoIds.Contains(repoId)
            };
        })]
    };

    private static ModelTypeInfo CreateTranscriberModels(HashSet<string> cachedRepoIds) => FromRegistry(
        "transcriber", "Transcriber", "Speech-to-text models", LocalTranscriber.Registry.GetAvailableModels(), cachedRepoIds);

    private static ModelTypeInfo CreateSynthesizerModels(HashSet<string> cachedRepoIds) => FromRegistry(
        "synthesizer", "Synthesizer", "Text-to-speech voices (Piper): English text is spoken as it is; other languages need phonemes or a phonemizer",
        LocalSynthesizer.Registry.GetAvailableModels(), cachedRepoIds);

    private static ModelTypeInfo CreateTranslatorModels(HashSet<string> cachedRepoIds) => FromRegistry(
        "translator", "Translator", "Neural machine translation models", LMSupply.Translator.LocalTranslator.Registry.GetAvailableModels(), cachedRepoIds);

    private static ModelTypeInfo CreateCaptionerModels(HashSet<string> cachedRepoIds) => FromRegistry(
        "captioner", "Captioner", "Image captioning models", LocalCaptioner.Registry.GetAvailableModels(), cachedRepoIds);

    // The OCR models are not listed through FromRegistry: one repository holds many of them, each in a subfolder, so the registry's id
    // is «repository/subfolder» — no repository, and never in the cache listing, which names repositories. Each model is cached
    // when the files its loader reads (the model, and a recognizer's dictionary) are in its subfolder.
    private static ModelTypeInfo CreateOcrModels(string cacheDirectory) => new()
    {
        Type = "ocr",
        DisplayName = "OCR",
        Description = "Text detection and recognition models",
        Models =
        [
            .. LocalOcr.DetectionRegistry.GetAvailableModels().Select(m => OcrModel(
                m.AliasName, m.RepoId, m.Subfolder, m.Description, [m.ModelFile], cacheDirectory)),
            .. LocalOcr.RecognitionRegistry.GetAvailableModels().Select(m => OcrModel(
                m.AliasName, m.RepoId, m.Subfolder, m.Description, [m.ModelFile, m.DictFile], cacheDirectory)),
        ]
    };

    private static ModelAliasInfo OcrModel(
        string alias, string repoId, string? subfolder, string? description, string[] files, string cacheDirectory) => new()
    {
        AliasName = alias,
        RepoId = repoId,
        Description = description ?? alias,
        IsCached = CacheManager.GetMissingFiles(cacheDirectory, repoId, files, subfolder).Count == 0
    };

    private static ModelTypeInfo CreateDetectorModels(HashSet<string> cachedRepoIds) => FromRegistry(
        "detector", "Object Detector", "Object detection models", LocalDetector.Registry.GetAvailableModels(), cachedRepoIds);

    private static ModelTypeInfo CreateSegmenterModels(HashSet<string> cachedRepoIds) => FromRegistry(
        "segmenter", "Segmenter", "Semantic and interactive (LoadInteractiveAsync) segmentation models",
        LocalSegmenter.Registry.GetAvailableModels(), cachedRepoIds);

    private static ModelTypeInfo CreateImageGeneratorModels(HashSet<string> cachedRepoIds)
    {
        var registry = ImageGeneratorModelRegistry.Default;
        var models = registry.GetAvailableModels().Select(info =>
        {
            return new ModelAliasInfo
            {
                AliasName = info.AliasName,
                RepoId = info.ModelId,
                Description = $"LCM ({info.RecommendedSteps} steps, {info.RecommendedGuidanceScale:F1} guidance)",
                IsCached = cachedRepoIds.Contains(info.ModelId)
            };
        }).ToList();

        return new ModelTypeInfo
        {
            Type = "imagegenerator",
            DisplayName = "Image Generator",
            Description = "Text-to-image generation models (LCM)",
            Models = models
        };
    }
}

/// <summary>
/// Model registry containing all model types.
/// </summary>
public record ModelRegistry
{
    public required List<ModelTypeInfo> ModelTypes { get; init; }
}

/// <summary>
/// Information about a model type.
/// </summary>
public record ModelTypeInfo
{
    public required string Type { get; init; }
    public required string DisplayName { get; init; }
    public required string Description { get; init; }
    public required List<ModelAliasInfo> Models { get; init; }
}

/// <summary>
/// Information about a model alias.
/// </summary>
public record ModelAliasInfo
{
    public required string AliasName { get; init; }
    public required string RepoId { get; init; }
    public required string Description { get; init; }
    public bool IsCached { get; init; }
}
