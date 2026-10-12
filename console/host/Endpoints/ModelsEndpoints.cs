using LMSupply;
using LMSupply.Captioner;
using LMSupply.Console.Host.Infrastructure;
using LMSupply.Console.Host.Models.OpenAI;
using LMSupply.Console.Host.Models.Requests;
using LMSupply.Console.Host.Services;
using LMSupply.Detector;
using LMSupply.Download;
using LMSupply.Embedder;
using LMSupply.Generator;
using LMSupply.ImageGenerator;
using LMSupply.Ocr;
using LMSupply.Reranker;
using LMSupply.Segmenter;
using LMSupply.Synthesizer;
using LMSupply.Transcriber;
using LMSupply.Translator;


namespace LMSupply.Console.Host.Endpoints;

public static class ModelsEndpoints
{
    public static void MapModelsEndpoints(this WebApplication app)
    {
        // OpenAI-compatible /v1/models endpoint
        var v1Group = app.MapGroup("/v1")
            .WithTags("Models");

        // GET /v1/models - List available models (OpenAI compatible)
        v1Group.MapGet("/models", () =>
        {
            var models = new List<Models.OpenAI.ModelInfo>();

            // Embedder — text embedding
            foreach (var m in LocalEmbedder.Registry.GetAvailableModels())
            {
                models.Add(new Models.OpenAI.ModelInfo
                {
                    Id = m.AliasName,
                    Capabilities = ["embeddings"],
                    ContextLength = ((IModelInfoBase)m).ContextLength
                });
            }

            // Reranker — semantic reranking
            foreach (var m in LocalReranker.Registry.GetAvailableModels())
            {
                models.Add(new Models.OpenAI.ModelInfo
                {
                    Id = m.AliasName,
                    Capabilities = ["reranking"],
                    ContextLength = ((IModelInfoBase)m).ContextLength
                });
            }

            // Generator — text generation / chat completions
            foreach (var m in LocalGenerator.Registry.GetAvailableModels())
            {
                models.Add(new Models.OpenAI.ModelInfo
                {
                    Id = m.AliasName,
                    Capabilities = ["chat.completions", "text.generation"],
                    ContextLength = ((IModelInfoBase)m).ContextLength
                });
            }

            // Translator — neural machine translation
            foreach (var m in LocalTranslator.Registry.GetAvailableModels())
            {
                models.Add(new Models.OpenAI.ModelInfo
                {
                    Id = m.AliasName,
                    Capabilities = ["translation"],
                    ContextLength = ((IModelInfoBase)m).ContextLength
                });
            }

            // Transcriber — speech-to-text
            foreach (var m in LocalTranscriber.Registry.GetAvailableModels())
            {
                models.Add(new Models.OpenAI.ModelInfo
                {
                    Id = m.AliasName,
                    Capabilities = ["transcription"],
                    ContextLength = ((IModelInfoBase)m).ContextLength
                });
            }

            // Synthesizer — text-to-speech
            foreach (var m in LocalSynthesizer.Registry.GetAvailableModels())
            {
                models.Add(new Models.OpenAI.ModelInfo
                {
                    Id = m.AliasName,
                    Capabilities = ["speech.synthesis"],
                    ContextLength = ((IModelInfoBase)m).ContextLength
                });
            }

            // Captioner — image-to-text captioning
            foreach (var m in LocalCaptioner.Registry.GetAvailableModels())
            {
                models.Add(new Models.OpenAI.ModelInfo
                {
                    Id = m.AliasName,
                    Capabilities = ["image.captioning"],
                    ContextLength = ((IModelInfoBase)m).ContextLength
                });
            }

            // Detector — object detection
            foreach (var m in LocalDetector.Registry.GetAvailableModels())
            {
                models.Add(new Models.OpenAI.ModelInfo
                {
                    Id = m.AliasName,
                    Capabilities = ["object.detection"],
                    ContextLength = ((IModelInfoBase)m).ContextLength
                });
            }

            // Segmenter — image segmentation
            foreach (var m in LocalSegmenter.Registry.GetAvailableModels())
            {
                models.Add(new Models.OpenAI.ModelInfo
                {
                    Id = m.AliasName,
                    Capabilities = ["image.segmentation"],
                    ContextLength = ((IModelInfoBase)m).ContextLength
                });
            }

            // ImageGenerator — text-to-image generation
            foreach (var m in LocalImageGenerator.Registry.GetAvailableModels())
            {
                models.Add(new Models.OpenAI.ModelInfo
                {
                    Id = m.AliasName,
                    Capabilities = ["image.generation"],
                    ContextLength = ((IModelInfoBase)m).ContextLength
                });
            }

            // OCR — separate detection and recognition registries (no unified Registry property)
            foreach (var m in LocalOcr.DetectionRegistry.GetAvailableModels())
            {
                models.Add(new Models.OpenAI.ModelInfo
                {
                    Id = m.AliasName,
                    Capabilities = ["ocr.detection"],
                    ContextLength = ((IModelInfoBase)m).ContextLength
                });
            }
            foreach (var m in LocalOcr.RecognitionRegistry.GetAvailableModels())
            {
                models.Add(new Models.OpenAI.ModelInfo
                {
                    Id = m.AliasName,
                    Capabilities = ["ocr.recognition"],
                    ContextLength = ((IModelInfoBase)m).ContextLength
                });
            }

            return Results.Ok(new ModelList { Data = models });
        })
        .WithName("ListModels")
        .WithSummary("List all available models (OpenAI compatible)")
        .WithDescription("Returns all available model aliases across all domains.")
        .Produces<ModelList>()
        .Produces<ErrorResponse>(400);

        // GET /v1/models/{model} - Get model info (OpenAI compatible)
        v1Group.MapGet("/models/{*model}", (string model) =>
        {
            // Search all registries for the model alias
            var allModels = Enumerable.Empty<IModelInfoBase>()
                .Concat(LocalEmbedder.Registry.GetAvailableModels())
                .Concat(LocalReranker.Registry.GetAvailableModels())
                .Concat(LocalGenerator.Registry.GetAvailableModels())
                .Concat(LocalTranslator.Registry.GetAvailableModels())
                .Concat(LocalTranscriber.Registry.GetAvailableModels())
                .Concat(LocalSynthesizer.Registry.GetAvailableModels())
                .Concat(LocalCaptioner.Registry.GetAvailableModels())
                .Concat(LocalDetector.Registry.GetAvailableModels())
                .Concat(LocalSegmenter.Registry.GetAvailableModels())
                .Concat(LocalImageGenerator.Registry.GetAvailableModels())
                .Concat(LocalOcr.DetectionRegistry.GetAvailableModels())
                .Concat(LocalOcr.RecognitionRegistry.GetAvailableModels());

            var found = allModels.FirstOrDefault(m =>
                string.Equals(m.AliasName, model, StringComparison.OrdinalIgnoreCase) ||
                string.Equals(m.CatalogKey, model, StringComparison.OrdinalIgnoreCase) ||
                string.Equals(m.Id, model, StringComparison.OrdinalIgnoreCase));

            if (found is null)
                return ApiHelper.Error($"Model '{model}' not found", "model_not_found", 404);

            return Results.Ok(new Models.OpenAI.ModelInfo
            {
                Id = found.AliasName,
                Capabilities = [],
                ContextLength = found.ContextLength
            });
        })
        .WithName("GetModel")
        .WithSummary("Get model information (OpenAI compatible)")
        .WithDescription("Returns information about a specific model.")
        .Produces<Models.OpenAI.ModelInfo>()
        .Produces<ErrorResponse>(404);

        // Cache management endpoints (LMSupply-specific)
        var cacheGroup = app.MapGroup("/api/cache")
            .WithTags("Cache");

        // GET /api/cache/models - List cached models
        cacheGroup.MapGet("/models", (CacheService cache) =>
        {
            var models = cache.GetCachedModels();
            return Results.Ok(new
            {
                models,
                totalCount = models.Count,
                totalSizeMB = models.Sum(m => m.SizeMB)
            });
        })
        .WithName("GetCachedModels")
        .WithSummary("List all cached models")
        .WithDescription("Returns all models stored in the local cache directory.");

        // GET /api/cache/models/type/{type} - List cached models by type
        cacheGroup.MapGet("/models/type/{type}", (string type, CacheService cache) =>
        {
            if (!Enum.TryParse<ModelType>(type, ignoreCase: true, out var modelType))
            {
                return ApiHelper.Error($"Invalid model type: {type}");
            }

            var models = cache.GetCachedModelsByType(modelType);
            return Results.Ok(models);
        })
        .WithName("GetModelsByType")
        .WithSummary("List cached models by type")
        .WithDescription("Returns cached models filtered by type (embedder, reranker, generator, etc.).")
        .Produces<ErrorResponse>(400);

        // GET /api/cache/loaded - List currently loaded models
        cacheGroup.MapGet("/loaded", (ModelManagerService manager) =>
        {
            var models = manager.GetLoadedModels();
            return Results.Ok(models);
        })
        .WithName("GetLoadedModels")
        .WithSummary("List currently loaded models")
        .WithDescription("Returns models currently loaded in memory and ready for inference.");

        // DELETE /api/cache/loaded/{key} - Unload a specific model from memory
        cacheGroup.MapDelete("/loaded/{*key}", async (string key, ModelManagerService manager) =>
        {
            var decodedKey = Uri.UnescapeDataString(key);
            var loadedModels = manager.GetLoadedModels();
            var exists = loadedModels.Any(m => string.Equals($"{m.ModelType}:{m.ModelId}", decodedKey, StringComparison.OrdinalIgnoreCase));

            if (!exists)
                return ApiHelper.Error($"Model not loaded: {decodedKey}", "not_found", 404);

            await manager.UnloadModelAsync(decodedKey);
            return Results.Ok(new { message = $"Model unloaded: {decodedKey}" });
        })
        .WithName("UnloadModel")
        .WithSummary("Unload a model from memory")
        .WithDescription("Unloads a loaded model from memory. Key format: {type}:{modelId} (e.g., 'generator:default').");

        // POST /api/cache/load - Pre-load a model with options
        cacheGroup.MapPost("/load", async (ModelLoadRequest request, ModelManagerService manager, CancellationToken ct) =>
        {
            try
            {
                if (string.IsNullOrWhiteSpace(request.Type))
                {
                    return ApiHelper.Error("'type' field is required");
                }

                var key = await manager.LoadModelAsync(request, ct);
                var loadedModels = manager.GetLoadedModels();
                var info = loadedModels.FirstOrDefault(m => string.Equals($"{m.ModelType}:{m.ModelId}", key, StringComparison.OrdinalIgnoreCase));
                return Results.Ok(new { key, model = info });
            }
            catch (ArgumentException ex)
            {
                return ApiHelper.Error(ex.Message);
            }
            catch (Exception ex)
            {
                return ApiHelper.InternalError(ex);
            }
        })
        .WithName("LoadModel")
        .WithSummary("Pre-load a model with options")
        .WithDescription("Pre-loads a model into memory with explicit configuration. Useful for warming up models before use or loading with specific provider/options.");

        // DELETE /api/cache/models/{repoId} - Delete cached model
        cacheGroup.MapDelete("/models/{*repoId}", async (string repoId, CacheService cache, ModelManagerService manager) =>
        {
            // URL decode the repoId (handles nvidia%2Fmodel → nvidia/model)
            var decodedRepoId = Uri.UnescapeDataString(repoId);

            // Unload model first if loaded
            var loadedModels = manager.GetLoadedModels();
            foreach (var loaded in loadedModels.Where(m => m.ModelId == decodedRepoId))
            {
                await manager.UnloadModelAsync($"{loaded.ModelType}:{decodedRepoId}");
            }

            var success = cache.DeleteModel(decodedRepoId);
            if (success)
            {
                return Results.Ok(new { message = $"Model deleted: {decodedRepoId}" });
            }

            return ApiHelper.Error($"Model not found: {decodedRepoId}", "not_found", 404);
        })
        .WithName("DeleteModel")
        .WithSummary("Delete a cached model")
        .WithDescription("Deletes a model from the local cache. The model is unloaded first if currently loaded.");

        // GET /api/cache/stats - Cache statistics
        cacheGroup.MapGet("/stats", (CacheService cache) =>
        {
            var models = cache.GetCachedModels();
            var byType = models.GroupBy(m => m.DetectedType)
                .ToDictionary(g => g.Key.ToString(), g => g.Count());

            return Results.Ok(new
            {
                totalModels = models.Count,
                totalSizeMB = models.Sum(m => m.SizeMB),
                cacheDirectory = cache.CacheDirectory,
                byType
            });
        })
        .WithName("GetCacheStats")
        .WithSummary("Cache statistics")
        .WithDescription("Returns statistics about the model cache including total count, size, and breakdown by type.");

        // Download endpoints
        var downloadGroup = app.MapGroup("/api/download")
            .WithTags("Download");

        // POST /api/download/check - Check model availability on HuggingFace
        downloadGroup.MapPost("/check", async (ModelCheckRequest request, DownloadService download, CancellationToken ct) =>
        {
            if (string.IsNullOrWhiteSpace(request.RepoId))
            {
                return ApiHelper.Error("RepoId is required");
            }

            var result = await download.CheckModelAsync(request.RepoId, ct);
            return Results.Ok(result);
        })
        .WithName("CheckModel")
        .WithSummary("Check model availability on HuggingFace")
        .WithDescription("Checks if a model repository exists on HuggingFace Hub and returns metadata.")
        .Produces<ErrorResponse>(400);

        // POST /api/download/model - Download model from HuggingFace (SSE progress)
        downloadGroup.MapPost("/model", async (ModelDownloadRequest request, DownloadService download, HttpContext context, CancellationToken ct) =>
        {
            if (string.IsNullOrWhiteSpace(request.RepoId))
            {
                context.Response.StatusCode = 400;
                await context.Response.WriteAsJsonAsync(new ErrorResponse
                {
                    Error = new ErrorDetail { Message = "RepoId is required", Type = "invalid_request_error" }
                }, ct);
                return;
            }

            // Set CORS headers manually before SSE response (prevents middleware conflict)
            var origin = context.Request.Headers.Origin.ToString();
            if (!string.IsNullOrEmpty(origin))
            {
                context.Response.Headers.AccessControlAllowOrigin = origin;
                context.Response.Headers.AccessControlAllowCredentials = "true";
            }

            // SSE headers with proxy compatibility
            context.Response.ContentType = "text/event-stream; charset=utf-8";
            context.Response.Headers.CacheControl = "no-cache";
            context.Response.Headers.Connection = "keep-alive";
            context.Response.Headers["X-Accel-Buffering"] = "no"; // Disable nginx/proxy buffering

            // Disable response buffering for real-time streaming
            var responseBodyFeature = context.Features.Get<Microsoft.AspNetCore.Http.Features.IHttpResponseBodyFeature>();
            responseBodyFeature?.DisableBuffering();

            // Start response to prevent CORS middleware from trying to add headers later
            await context.Response.StartAsync(ct);

            // Use separate cancellation for download - allows download to continue even if SSE stream fails
            using var downloadCts = new CancellationTokenSource();
            using var linkedCts = CancellationTokenSource.CreateLinkedTokenSource(ct, downloadCts.Token);

            // Throttle progress updates to reduce SSE traffic (max 2 updates per second)
            var lastUpdate = DateTime.MinValue;
            var updateInterval = TimeSpan.FromMilliseconds(500);
            var writeLock = new object();
            var lastPercentReported = -1;

            try
            {
                await download.DownloadModelAsync(
                    request.RepoId,
                    async progress =>
                    {
                        // Throttle: only send update if enough time has passed or significant progress
                        var now = DateTime.UtcNow;
                        var percentInt = (int)progress.PercentComplete;

                        lock (writeLock)
                        {
                            if (now - lastUpdate < updateInterval && percentInt == lastPercentReported && percentInt < 100)
                                return;

                            lastUpdate = now;
                            lastPercentReported = percentInt;
                        }

                        if (ct.IsCancellationRequested)
                            return;

                        var data = System.Text.Json.JsonSerializer.Serialize(new
                        {
                            fileName = progress.FileName,
                            bytesDownloaded = progress.BytesDownloaded,
                            totalBytes = progress.TotalBytes,
                            percentComplete = progress.PercentComplete
                        });
                        await context.Response.WriteAsync($"data: {data}\n\n", ct);
                        await context.Response.Body.FlushAsync(ct);
                    },
                    fileName: request.FileName,
                    cancellationToken: linkedCts.Token);

                await context.Response.WriteAsync("data: {\"status\":\"Completed\",\"percentComplete\":100}\n\n", ct);
                await context.Response.Body.FlushAsync(ct);
            }
            catch (OperationCanceledException)
            {
                // Client disconnected - don't try to write to closed response
            }
            catch (Exception ex)
            {
                try
                {
                    var escapedError = ex.Message.Replace("\\", "\\\\").Replace("\"", "\\\"");
                    await context.Response.WriteAsync($"data: {{\"status\":\"Failed\",\"error\":\"{escapedError}\"}}\n\n", ct);
                    await context.Response.Body.FlushAsync(ct);
                }
                catch
                {
                    // Ignore write errors if client disconnected
                }
            }
        })
        .WithName("DownloadModel")
        .WithSummary("Download model from HuggingFace (SSE progress)")
        .WithDescription("Downloads a model from HuggingFace Hub with real-time progress via Server-Sent Events.");
    }
}

public record ModelCheckRequest(string RepoId);
public record ModelDownloadRequest(string RepoId, string? FileName = null);
