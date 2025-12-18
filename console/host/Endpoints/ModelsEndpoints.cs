using LMSupply.Console.Host.Services;
using LMSupply.Download;

namespace LMSupply.Console.Host.Endpoints;

public static class ModelsEndpoints
{
    public static void MapModelsEndpoints(this WebApplication app)
    {
        var group = app.MapGroup("/api/models")
            .WithTags("Models")
            .WithOpenApi();

        // 캐시된 모델 목록
        group.MapGet("/", (CacheService cache) =>
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
        .WithSummary("캐시된 모든 모델 목록 조회");

        // 타입별 모델 목록
        group.MapGet("/type/{type}", (string type, CacheService cache) =>
        {
            if (!Enum.TryParse<ModelType>(type, ignoreCase: true, out var modelType))
            {
                return Results.BadRequest(new { error = $"Invalid model type: {type}" });
            }

            var models = cache.GetCachedModelsByType(modelType);
            return Results.Ok(models);
        })
        .WithName("GetModelsByType")
        .WithSummary("타입별 모델 목록 조회");

        // 로드된 모델 목록
        group.MapGet("/loaded", (ModelManagerService manager) =>
        {
            var models = manager.GetLoadedModels();
            return Results.Ok(models);
        })
        .WithName("GetLoadedModels")
        .WithSummary("현재 로드된 모델 목록");

        // 모델 삭제
        group.MapDelete("/{*repoId}", async (string repoId, CacheService cache, ModelManagerService manager) =>
        {
            // 로드된 모델이면 먼저 언로드
            var loadedModels = manager.GetLoadedModels();
            foreach (var loaded in loadedModels.Where(m => m.ModelId == repoId))
            {
                await manager.UnloadModelAsync($"{loaded.ModelType}:{repoId}");
            }

            var success = cache.DeleteModel(repoId);
            if (success)
            {
                return Results.Ok(new { message = $"Model deleted: {repoId}" });
            }

            return Results.NotFound(new { error = $"Model not found: {repoId}" });
        })
        .WithName("DeleteModel")
        .WithSummary("캐시된 모델 삭제");

        // 캐시 통계
        group.MapGet("/stats", (CacheService cache) =>
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
        .WithSummary("캐시 통계");

        // 모델 검증 (HuggingFace 존재 여부 및 지원 여부)
        group.MapPost("/check", async (ModelCheckRequest request, DownloadService download, CancellationToken ct) =>
        {
            if (string.IsNullOrWhiteSpace(request.RepoId))
            {
                return Results.BadRequest(new { error = "RepoId is required" });
            }

            var result = await download.CheckModelAsync(request.RepoId, ct);
            return Results.Ok(result);
        })
        .WithName("CheckModel")
        .WithSummary("HuggingFace 모델 검증");

        // 모델 다운로드 (SSE 진행률)
        group.MapPost("/download", async (ModelDownloadRequest request, DownloadService download, HttpContext context, CancellationToken ct) =>
        {
            if (string.IsNullOrWhiteSpace(request.RepoId))
            {
                context.Response.StatusCode = 400;
                await context.Response.WriteAsJsonAsync(new { error = "RepoId is required" }, ct);
                return;
            }

            // SSE 응답 전에 CORS 헤더를 수동으로 설정 (응답 시작 후 CORS 미들웨어 충돌 방지)
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
                        // Synchronous write from Progress<T> callback
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
        .WithSummary("HuggingFace 모델 다운로드 (SSE)");
    }
}

public record ModelCheckRequest(string RepoId);
public record ModelDownloadRequest(string RepoId);
