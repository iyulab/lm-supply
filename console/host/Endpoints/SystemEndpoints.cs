using LMSupply.Console.Host.Infrastructure;
using LMSupply.Console.Host.Services;

namespace LMSupply.Console.Host.Endpoints;

public static class SystemEndpoints
{
    public static void MapSystemEndpoints(this WebApplication app)
    {
        var group = app.MapGroup("/api/system")
            .WithTags("System")
            ;

        // System status
        group.MapGet("/status", (SystemMonitorService monitor, ModelManagerService modelManager) =>
        {
            var status = monitor.GetStatus();
            var loadedModels = modelManager.GetLoadedModels();

            return Results.Ok(new
            {
                status,
                loadedModels = loadedModels.Count,
                models = loadedModels
            });
        })
        .WithName("GetSystemStatus")
        .WithSummary("Get system status");

        // GPU info
        group.MapGet("/gpu", (SystemMonitorService monitor) =>
        {
            var gpuInfo = monitor.GetGpuInfo();
            return Results.Ok(gpuInfo);
        })
        .WithName("GetGpuInfo")
        .WithSummary("Get GPU info");

        // Memory metrics
        group.MapGet("/memory", (SystemMonitorService monitor) =>
        {
            var memory = monitor.GetMemoryMetrics();
            return Results.Ok(memory);
        })
        .WithName("GetMemoryMetrics")
        .WithSummary("Get memory metrics");

        // Real-time metrics stream (SSE)
        group.MapGet("/metrics/stream", async (HttpContext context, SystemMonitorService monitor) =>
        {
            var cancellationToken = context.RequestAborted;
            var metrics = monitor.StreamMetricsAsync(cancellationToken, intervalMs: 1000);

            await SseHelper.StreamAsync(context, metrics, cancellationToken);
        })
        .WithName("StreamMetrics")
        .WithSummary("Real-time metrics stream (SSE)");

        // Current version info
        group.MapGet("/version", (UpdateService updateService) =>
        {
            return Results.Ok(new { version = updateService.CurrentVersion, rid = updateService.CurrentRid });
        })
        .WithName("GetVersion")
        .WithSummary("Get current version info");

        // Check for updates
        group.MapGet("/update", async (UpdateService updateService) =>
        {
            var result = await updateService.CheckForUpdateAsync();
            return Results.Ok(result);
        })
        .WithName("CheckUpdate")
        .WithSummary("Check for a newer version");

        // Apply update (SSE stream)
        group.MapPost("/update", async (HttpContext context, UpdateService updateService) =>
        {
            var ct = context.RequestAborted;
            var progress = updateService.ApplyUpdateAsync(ct);
            await SseHelper.StreamAsync(context, progress, ct);
        })
        .WithName("ApplyUpdate")
        .WithSummary("Download and apply the update");
    }
}
