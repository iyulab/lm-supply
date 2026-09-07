using LMSupply.Console.Host.Infrastructure;
using LMSupply.Console.Host.Services;
using LMSupply.Generator;
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
        group.MapGet("/models", (CacheService cache) =>
        {
            var cachedModels = cache.GetCachedModels();
            var cachedRepoIds = cachedModels.Select(m => m.RepoId).ToHashSet(StringComparer.OrdinalIgnoreCase);

            var registry = new ModelRegistry
            {
                ModelTypes = GetAllModelTypes(cachedRepoIds)
            };

            return Results.Ok(registry);
        })
        .WithName("GetModelRegistry")
        .WithSummary("Get all model types with their aliases and repo IDs")
        .WithDescription("Returns a comprehensive list of all model types, their available aliases, and the actual HuggingFace repository IDs.");

        // GET /api/registry/models/{type} - Get models for specific type
        group.MapGet("/models/{type}", (string type, CacheService cache) =>
        {
            var cachedModels = cache.GetCachedModels();
            var cachedRepoIds = cachedModels.Select(m => m.RepoId).ToHashSet(StringComparer.OrdinalIgnoreCase);

            var modelType = GetModelTypeByName(type, cachedRepoIds);
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

    private static List<ModelTypeInfo> GetAllModelTypes(HashSet<string> cachedRepoIds)
    {
        return
        [
            CreateGeneratorModels(cachedRepoIds),
            CreateEmbedderModels(cachedRepoIds),
            CreateRerankerModels(cachedRepoIds),
            CreateTranscriberModels(cachedRepoIds),
            CreateSynthesizerModels(cachedRepoIds),
            CreateTranslatorModels(cachedRepoIds),
            CreateCaptionerModels(cachedRepoIds),
            CreateOcrModels(cachedRepoIds),
            CreateDetectorModels(cachedRepoIds),
            CreateSegmenterModels(cachedRepoIds),
            CreateImageGeneratorModels(cachedRepoIds),
        ];
    }

    private static ModelTypeInfo? GetModelTypeByName(string type, HashSet<string> cachedRepoIds)
    {
        return type.ToLowerInvariant() switch
        {
            "generator" => CreateGeneratorModels(cachedRepoIds),
            "embedder" => CreateEmbedderModels(cachedRepoIds),
            "reranker" => CreateRerankerModels(cachedRepoIds),
            "transcriber" => CreateTranscriberModels(cachedRepoIds),
            "synthesizer" => CreateSynthesizerModels(cachedRepoIds),
            "translator" => CreateTranslatorModels(cachedRepoIds),
            "captioner" => CreateCaptionerModels(cachedRepoIds),
            "ocr" => CreateOcrModels(cachedRepoIds),
            "detector" => CreateDetectorModels(cachedRepoIds),
            "segmenter" => CreateSegmenterModels(cachedRepoIds),
            "imagegenerator" or "imagegen" or "image" => CreateImageGeneratorModels(cachedRepoIds),
            _ => null
        };
    }

    /// <summary>
    /// Resolves a generator alias to the repository id the cache is actually keyed by.
    ///
    /// <para>
    /// <c>WellKnownModels.Generator</c> deliberately names registry aliases rather than repository
    /// ids for some tiers (<c>"phi-4-mini"</c>), and the cache index is built from directory names
    /// of the form <c>models--org--name</c> — so it only ever contains <c>org/name</c> strings.
    /// Comparing an alias against it therefore reported "not cached" for a model sitting on disk,
    /// and the response's <c>RepoId</c> field carried something that was not a repository id.
    /// </para>
    ///
    /// <para>
    /// <c>"default"</c> stays unresolved on purpose: the generator picks it by hardware at load
    /// time, so there is no single repository id this endpoint could name for it.
    /// </para>
    /// </summary>
    private static string ResolveGeneratorRepoId(string idOrAlias)
        => GeneratorModelRegistry.Default.TryResolve(idOrAlias, out var model) && model is not null
            ? model.ModelId
            : idOrAlias;

    private static ModelTypeInfo CreateGeneratorModels(HashSet<string> cachedRepoIds) => new()
    {
        Type = "generator",
        DisplayName = "Text Generator",
        Description = "LLM text generation models",
        Models =
        [
            new ModelAliasInfo { AliasName = "default", RepoId = ResolveGeneratorRepoId(WellKnownModels.Generator.Default), Description = "Microsoft Phi-4 Mini (3.8B, MIT)", IsCached = cachedRepoIds.Contains(ResolveGeneratorRepoId(WellKnownModels.Generator.Default)) },
            new ModelAliasInfo { AliasName = "fast", RepoId = ResolveGeneratorRepoId(WellKnownModels.Generator.Fast), Description = "Phi-4 Mini (3.8B, fastest FC-capable)", IsCached = cachedRepoIds.Contains(ResolveGeneratorRepoId(WellKnownModels.Generator.Fast)) },
            new ModelAliasInfo { AliasName = "quality", RepoId = ResolveGeneratorRepoId(WellKnownModels.Generator.Quality), Description = "Microsoft Phi-4 (14B, highest quality)", IsCached = cachedRepoIds.Contains(ResolveGeneratorRepoId(WellKnownModels.Generator.Quality)) },
            new ModelAliasInfo { AliasName = "medium", RepoId = ResolveGeneratorRepoId(WellKnownModels.Generator.Medium), Description = "Phi-3.5 Mini (3.8B, 128K context)", IsCached = cachedRepoIds.Contains(ResolveGeneratorRepoId(WellKnownModels.Generator.Medium)) },
            new ModelAliasInfo { AliasName = "large", RepoId = ResolveGeneratorRepoId(WellKnownModels.Generator.Large), Description = "Microsoft Phi-4 (14B, highest quality)", IsCached = cachedRepoIds.Contains(ResolveGeneratorRepoId(WellKnownModels.Generator.Large)) },
        ]
    };

    private static ModelTypeInfo CreateEmbedderModels(HashSet<string> cachedRepoIds) => new()
    {
        Type = "embedder",
        DisplayName = "Text Embedder",
        Description = "Text embedding models for semantic search",
        Models =
        [
            new ModelAliasInfo { AliasName = "default", RepoId = WellKnownModels.Embedder.Default, Description = "BGE-M3 (568M, 1024 dims, 100+ languages)", IsCached = cachedRepoIds.Contains(WellKnownModels.Embedder.Default) },
            new ModelAliasInfo { AliasName = "fast", RepoId = WellKnownModels.Embedder.Fast, Description = "multilingual-e5-small (118M, 384 dims, 100+ languages)", IsCached = cachedRepoIds.Contains(WellKnownModels.Embedder.Fast) },
            new ModelAliasInfo { AliasName = "quality", RepoId = WellKnownModels.Embedder.Quality, Description = "GTE Large EN v1.5 (434M, 1024 dims, 8K context)", IsCached = cachedRepoIds.Contains(WellKnownModels.Embedder.Quality) },
            new ModelAliasInfo { AliasName = "large", RepoId = WellKnownModels.Embedder.Large, Description = "Nomic Embed v1.5 (137M, 8K context)", IsCached = cachedRepoIds.Contains(WellKnownModels.Embedder.Large) },
            new ModelAliasInfo { AliasName = "multilingual", RepoId = WellKnownModels.Embedder.Multilingual, Description = "E5 Base (278M, 100+ languages)", IsCached = cachedRepoIds.Contains(WellKnownModels.Embedder.Multilingual) },
            new ModelAliasInfo { AliasName = "multilingual-large", RepoId = WellKnownModels.Embedder.MultilingualLarge, Description = "BGE M3 (568M, best multilingual)", IsCached = cachedRepoIds.Contains(WellKnownModels.Embedder.MultilingualLarge) },
        ]
    };

    private static ModelTypeInfo CreateRerankerModels(HashSet<string> cachedRepoIds) => new()
    {
        Type = "reranker",
        DisplayName = "Reranker",
        Description = "Cross-encoder reranking models",
        Models =
        [
            new ModelAliasInfo { AliasName = "default", RepoId = WellKnownModels.Reranker.Default, Description = "MS MARCO MiniLM L6 (22M)", IsCached = cachedRepoIds.Contains(WellKnownModels.Reranker.Default) },
            new ModelAliasInfo { AliasName = "fast", RepoId = WellKnownModels.Reranker.Fast, Description = "TinyBERT L2 (4.4M, ultra-fast)", IsCached = cachedRepoIds.Contains(WellKnownModels.Reranker.Fast) },
            new ModelAliasInfo { AliasName = "quality", RepoId = WellKnownModels.Reranker.Quality, Description = "BGE Reranker Base (278M)", IsCached = cachedRepoIds.Contains(WellKnownModels.Reranker.Quality) },
            new ModelAliasInfo { AliasName = "large", RepoId = WellKnownModels.Reranker.Large, Description = "BGE Reranker Large (560M)", IsCached = cachedRepoIds.Contains(WellKnownModels.Reranker.Large) },
            new ModelAliasInfo { AliasName = "multilingual", RepoId = WellKnownModels.Reranker.Multilingual, Description = "BGE Reranker v2 M3 (8K context)", IsCached = cachedRepoIds.Contains(WellKnownModels.Reranker.Multilingual) },
        ]
    };

    private static ModelTypeInfo CreateTranscriberModels(HashSet<string> cachedRepoIds) => new()
    {
        Type = "transcriber",
        DisplayName = "Speech Transcriber",
        Description = "Speech-to-text (Whisper) models",
        Models =
        [
            new ModelAliasInfo { AliasName = "default", RepoId = "openai/whisper-base", Description = "Whisper Base (74M)", IsCached = cachedRepoIds.Contains("openai/whisper-base") },
            new ModelAliasInfo { AliasName = "fast", RepoId = "openai/whisper-tiny", Description = "Whisper Tiny (39M, fastest)", IsCached = cachedRepoIds.Contains("openai/whisper-tiny") },
            new ModelAliasInfo { AliasName = "quality", RepoId = "openai/whisper-small", Description = "Whisper Small (244M)", IsCached = cachedRepoIds.Contains("openai/whisper-small") },
            new ModelAliasInfo { AliasName = "large", RepoId = "openai/whisper-medium", Description = "Whisper Medium (769M)", IsCached = cachedRepoIds.Contains("openai/whisper-medium") },
        ]
    };

    private static ModelTypeInfo CreateSynthesizerModels(HashSet<string> cachedRepoIds) => new()
    {
        Type = "synthesizer",
        DisplayName = "Speech Synthesizer",
        Description = "Text-to-speech (Piper) models",
        Models =
        [
            new ModelAliasInfo { AliasName = "default", RepoId = "rhasspy/piper-voices", Description = "Piper Voices (multi-language)", IsCached = cachedRepoIds.Contains("rhasspy/piper-voices") },
            new ModelAliasInfo { AliasName = "en-us", RepoId = "rhasspy/piper-voices", Description = "English US voices", IsCached = cachedRepoIds.Contains("rhasspy/piper-voices") },
        ]
    };

    private static ModelTypeInfo CreateTranslatorModels(HashSet<string> cachedRepoIds) => new()
    {
        Type = "translator",
        DisplayName = "Translator",
        Description = "Neural machine translation models",
        Models =
        [
            new ModelAliasInfo { AliasName = "default", RepoId = "facebook/nllb-200-distilled-600M", Description = "NLLB 200 (200+ languages)", IsCached = cachedRepoIds.Contains("facebook/nllb-200-distilled-600M") },
            new ModelAliasInfo { AliasName = "fast", RepoId = "Helsinki-NLP/opus-mt-en-ko", Description = "OPUS MT (language pairs)", IsCached = cachedRepoIds.Contains("Helsinki-NLP/opus-mt-en-ko") },
        ]
    };

    private static ModelTypeInfo CreateCaptionerModels(HashSet<string> cachedRepoIds) => new()
    {
        Type = "captioner",
        DisplayName = "Image Captioner",
        Description = "Image captioning and VQA models",
        Models =
        [
            new ModelAliasInfo { AliasName = "default", RepoId = "microsoft/Florence-2-base", Description = "Florence 2 Base", IsCached = cachedRepoIds.Contains("microsoft/Florence-2-base") },
            new ModelAliasInfo { AliasName = "quality", RepoId = "microsoft/Florence-2-large", Description = "Florence 2 Large", IsCached = cachedRepoIds.Contains("microsoft/Florence-2-large") },
        ]
    };

    private static ModelTypeInfo CreateOcrModels(HashSet<string> cachedRepoIds) => new()
    {
        Type = "ocr",
        DisplayName = "OCR",
        Description = "Optical character recognition models",
        Models =
        [
            new ModelAliasInfo { AliasName = "default", RepoId = "tesseract-ocr/tessdata", Description = "Tesseract (100+ languages)", IsCached = cachedRepoIds.Contains("tesseract-ocr/tessdata") },
        ]
    };

    private static ModelTypeInfo CreateDetectorModels(HashSet<string> cachedRepoIds) => new()
    {
        Type = "detector",
        DisplayName = "Object Detector",
        Description = "Object detection models",
        Models =
        [
            new ModelAliasInfo { AliasName = "default", RepoId = "ultralytics/yolov8n", Description = "YOLOv8 Nano (3.2M)", IsCached = cachedRepoIds.Contains("ultralytics/yolov8n") },
            new ModelAliasInfo { AliasName = "quality", RepoId = "ultralytics/yolov8s", Description = "YOLOv8 Small (11M)", IsCached = cachedRepoIds.Contains("ultralytics/yolov8s") },
        ]
    };

    private static ModelTypeInfo CreateSegmenterModels(HashSet<string> cachedRepoIds) => new()
    {
        Type = "segmenter",
        DisplayName = "Image Segmenter",
        Description = "Image segmentation models",
        Models =
        [
            new ModelAliasInfo { AliasName = "default", RepoId = "facebook/sam-vit-base", Description = "SAM ViT Base", IsCached = cachedRepoIds.Contains("facebook/sam-vit-base") },
        ]
    };

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
