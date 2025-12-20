using LMSupply.Console.Host.Infrastructure;
using LMSupply.Console.Host.Models.OpenAI;
using LMSupply.Console.Host.Services;
using LMSupply.Download;

namespace LMSupply.Console.Host.Endpoints;

public static class ModelsEndpoints
{
    public static void MapModelsEndpoints(this WebApplication app)
    {
        // OpenAI-compatible /v1/models endpoint
        var v1Group = app.MapGroup("/v1")
            .WithTags("Models")
            .WithOpenApi();

        // GET /v1/models - List available models (OpenAI compatible)
        v1Group.MapGet("/models", (ModelManagerService manager) =>
        {
            var loadedModels = manager.GetLoadedModels();
            var models = loadedModels.Select(m => new ModelInfo
            {
                Id = $"{m.ModelType.ToString().ToLower()}:{m.ModelId}",
                OwnedBy = "lmsupply"
            }).ToList();

            // Add well-known model aliases
            var aliases = new[]
            {
                new ModelInfo { Id = "generator:default", OwnedBy = "lmsupply" },
                new ModelInfo { Id = "embedder:default", OwnedBy = "lmsupply" },
                new ModelInfo { Id = "reranker:default", OwnedBy = "lmsupply" },
                new ModelInfo { Id = "transcriber:default", OwnedBy = "lmsupply" },
                new ModelInfo { Id = "synthesizer:default", OwnedBy = "lmsupply" },
                new ModelInfo { Id = "translator:default", OwnedBy = "lmsupply" },
                new ModelInfo { Id = "captioner:default", OwnedBy = "lmsupply" },
                new ModelInfo { Id = "ocr:default", OwnedBy = "lmsupply" },
                new ModelInfo { Id = "detector:default", OwnedBy = "lmsupply" },
                new ModelInfo { Id = "segmenter:default", OwnedBy = "lmsupply" }
            };

            // Merge loaded models with aliases (avoid duplicates)
            var allModels = aliases
                .Where(a => !models.Any(m => m.Id == a.Id))
                .Concat(models)
                .ToList();

            return Results.Ok(new ModelListResponse { Data = allModels });
        })
        .WithName("ListModels")
        .WithSummary("List available models (OpenAI compatible)")
        .WithDescription("Returns a list of models available in the local cache and loaded models.")
        .Produces<ModelListResponse>();

        // GET /v1/models/{model} - Get model info (OpenAI compatible)
        v1Group.MapGet("/models/{*model}", (string model) =>
        {
            return Results.Ok(new ModelInfo
            {
                Id = model,
                OwnedBy = "lmsupply"
            });
        })
        .WithName("GetModel")
        .WithSummary("Get model information (OpenAI compatible)")
        .WithDescription("Returns information about a specific model.")
        .Produces<ModelInfo>();

        // Cache management endpoints (LMSupply-specific)
        var cacheGroup = app.MapGroup("/api/cache")
            .WithTags("Cache")
            .WithOpenApi();

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

            return Results.NotFound(new { error = $"Model not found: {decodedRepoId}" });
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
            .WithTags("Download")
            .WithOpenApi();

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
                await context.Response.WriteAsJsonAsync(new { error = "RepoId is required" }, ct);
                return;
            }

            // Set CORS headers manually before SSE response (prevents middleware conflict)
            var origin = context.Request.Headers.Origin.ToString();
            if (!string.IsNullOrEmpty(origin))
            {
                context.Response.Headers.AccessControlAllowOrigin = origin;
                context.Response.Headers.AccessControlAllowCredentials = "true";
            }

            context.Response.ContentType = "text/event-stream";
            context.Response.Headers.CacheControl = "no-cache";
            context.Response.Headers.Connection = "keep-alive";

            try
            {
                await download.DownloadModelAsync(
                    request.RepoId,
                    progress =>
                    {
                        var data = System.Text.Json.JsonSerializer.Serialize(new
                        {
                            fileName = progress.FileName,
                            bytesDownloaded = progress.BytesDownloaded,
                            totalBytes = progress.TotalBytes,
                            percentComplete = progress.PercentComplete
                        });
                        context.Response.WriteAsync($"data: {data}\n\n", ct).GetAwaiter().GetResult();
                        context.Response.Body.FlushAsync(ct).GetAwaiter().GetResult();
                    },
                    ct);

                await context.Response.WriteAsync("data: {\"status\":\"Completed\",\"percentComplete\":100}\n\n", ct);
            }
            catch (Exception ex)
            {
                var escapedError = ex.Message.Replace("\\", "\\\\").Replace("\"", "\\\"");
                await context.Response.WriteAsync($"data: {{\"status\":\"Failed\",\"error\":\"{escapedError}\"}}\n\n", ct);
            }
        })
        .WithName("DownloadModel")
        .WithSummary("Download model from HuggingFace (SSE progress)")
        .WithDescription("Downloads a model from HuggingFace Hub with real-time progress via Server-Sent Events.");
    }
}

public record ModelCheckRequest(string RepoId);
public record ModelDownloadRequest(string RepoId);
