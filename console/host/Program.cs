using System.Reflection;
using System.Text.Json;
using System.Text.Json.Serialization;
using Microsoft.Extensions.FileProviders;
using LMSupply.Console.Host.Data;
using LMSupply.Console.Host.Endpoints;
using LMSupply.Console.Host.Infrastructure;
using LMSupply.Console.Host.Services;
using Microsoft.EntityFrameworkCore;

// This host lets users load an arbitrary model id via LocalGenerator.LoadAsync, including
// explicit ONNX Runtime GenAI repos — register the backend so that path keeps working.
LMSupply.Generator.Onnx.OnnxGeneratorBackend.Register();

// CLI mode: lm-supply.exe update
if (args.Length > 0 && args[0].Equals("update", StringComparison.OrdinalIgnoreCase))
{
    await RunCliUpdateAsync();
    return;
}

var builder = WebApplication.CreateBuilder(args);

// Kestrel configuration - allow large file uploads (up to 500 MB)
builder.WebHost.ConfigureKestrel(options =>
{
    options.Limits.MaxRequestBodySize = 500 * 1024 * 1024; // 500MB
});

// JSON serialization settings
builder.Services.ConfigureHttpJsonOptions(options =>
{
    options.SerializerOptions.PropertyNamingPolicy = JsonNamingPolicy.CamelCase;
    options.SerializerOptions.DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull;
    options.SerializerOptions.Converters.Add(new JsonStringEnumConverter(JsonNamingPolicy.CamelCase));
});

// CORS configuration (for development)
builder.Services.AddCors(options =>
{
    options.AddDefaultPolicy(policy =>
    {
        policy.WithOrigins("http://localhost:5173", "http://localhost:3000")
              .AllowAnyHeader()
              .AllowAnyMethod()
              .AllowCredentials();
    });
    // SSE endpoints handle CORS manually to avoid middleware conflicts
    options.AddPolicy("SSE", policy =>
    {
        policy.WithOrigins("http://localhost:5173", "http://localhost:3000")
              .AllowAnyHeader()
              .AllowAnyMethod()
              .AllowCredentials()
              .SetPreflightMaxAge(TimeSpan.FromMinutes(10));
    });
});

// OpenAPI/Swagger
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen(c =>
{
    c.SwaggerDoc("v1", new() { Title = "LMSupply Console API", Version = "v1" });
    c.AddSecurityDefinition("Bearer", new Microsoft.OpenApi.OpenApiSecurityScheme
    {
        Name = "Authorization",
        Type = Microsoft.OpenApi.SecuritySchemeType.Http,
        Scheme = "bearer",
        BearerFormat = "API Key",
        In = Microsoft.OpenApi.ParameterLocation.Header,
        Description = "Enter your API key: lms-...",
    });
    c.AddSecurityRequirement(doc => new Microsoft.OpenApi.OpenApiSecurityRequirement
    {
        {
            new Microsoft.OpenApi.OpenApiSecuritySchemeReference("Bearer", doc),
            []
        }
    });
});

// Service registration
builder.Services.AddSingleton<CacheService>();
builder.Services.AddSingleton<SystemMonitorService>();
builder.Services.AddSingleton<ModelManagerService>();
builder.Services.AddSingleton<DownloadService>();
builder.Services.AddSingleton<UpdateService>();
builder.Services.AddSingleton<TempFileService>();
builder.Services.AddHostedService(sp => sp.GetRequiredService<TempFileService>());

// API Key storage (SQLite)
var dbPath = Path.Combine(
    Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
    ".lmsupply", "api-keys.db");
Directory.CreateDirectory(Path.GetDirectoryName(dbPath)!);
builder.Services.AddDbContextFactory<ApiKeyDbContext>(options =>
    options.UseSqlite($"Data Source={dbPath}"));
builder.Services.AddSingleton<ApiKeyService>();

var app = builder.Build();

// Ensure API key database is created and clean up old logs
using (var scope = app.Services.CreateScope())
{
    var dbFactory = scope.ServiceProvider.GetRequiredService<IDbContextFactory<ApiKeyDbContext>>();
    await using var ctx = await dbFactory.CreateDbContextAsync();
    await ctx.Database.EnsureCreatedAsync();
}
var apiKeyService = app.Services.GetRequiredService<ApiKeyService>();
await apiKeyService.CleanupOldLogsAsync();

// Swagger UI
app.UseSwagger();
app.UseSwaggerUI(c =>
{
    c.SwaggerEndpoint("/swagger/v1/swagger.json", "LMSupply Console API v1");
    c.RoutePrefix = "swagger";
});

app.UseCors();
app.UseMiddleware<RequestIdMiddleware>();
app.UseMiddleware<ApiKeyMiddleware>();
app.UseMiddleware<ErrorMiddleware>();

// Serve static files from embedded resources (no manifest exists if wwwroot was missing at build time)
var assembly = Assembly.GetExecutingAssembly();
ManifestEmbeddedFileProvider? embeddedProvider = null;
try
{
    embeddedProvider = new ManifestEmbeddedFileProvider(assembly, "wwwroot");
    app.UseDefaultFiles(new DefaultFilesOptions { FileProvider = embeddedProvider });
    app.UseStaticFiles(new StaticFileOptions { FileProvider = embeddedProvider });
}
catch (InvalidOperationException)
{
    // No manifest (built without wwwroot) — only swagger is served
}

// Map API endpoints
app.MapModelsEndpoints();
app.MapSystemEndpoints();
app.MapChatEndpoints();
app.MapEmbedEndpoints();
app.MapRerankEndpoints();
app.MapTranscribeEndpoints();
app.MapSynthesizeEndpoints();
app.MapCaptionEndpoints();
app.MapOcrEndpoints();
app.MapDetectEndpoints();
app.MapSegmentEndpoints();
app.MapTranslateEndpoints();
app.MapImageEndpoints();
app.MapFileEndpoints();
app.MapModelRegistryEndpoints();
app.MapApiKeyEndpoints();

// Health check
app.MapGet("/health", () => Results.Ok(new { status = "healthy", timestamp = DateTime.UtcNow }));

// Root endpoint
app.MapGet("/", () =>
{
    var indexFile = embeddedProvider?.GetFileInfo("index.html");
    if (indexFile is { Exists: true })
    {
        return Results.Stream(indexFile.CreateReadStream(), "text/html");
    }
    return Results.Redirect("/swagger");
});

// SPA fallback: non-API GET paths go to index.html (client-side routing)
// Only match GET requests — POST/PUT/DELETE to unknown routes should return 404, not HTML
app.MapFallback(context =>
{
    if (!HttpMethods.IsGet(context.Request.Method))
    {
        context.Response.StatusCode = 404;
        return Task.CompletedTask;
    }

    var indexFile = embeddedProvider?.GetFileInfo("index.html");
    if (indexFile is { Exists: true })
    {
        context.Response.ContentType = "text/html";
        return indexFile.CreateReadStream().CopyToAsync(context.Response.Body);
    }

    context.Response.StatusCode = 404;
    return Task.CompletedTask;
});

app.Run();

static async Task RunCliUpdateAsync()
{
    using var host = Host.CreateDefaultBuilder()
        .ConfigureLogging(l => l.ClearProviders().AddConsole().SetMinimumLevel(LogLevel.Warning))
        .ConfigureServices(s => s.AddSingleton<UpdateService>())
        .Build();

    await host.StartAsync();
    var updateService = host.Services.GetRequiredService<UpdateService>();

    Console.WriteLine($"LMSupply Console v{updateService.CurrentVersion} ({updateService.CurrentRid})");

    var check = await updateService.CheckForUpdateAsync();
    if (!check.UpdateAvailable)
    {
        Console.WriteLine(check.Error is not null
            ? $"Update check failed: {check.Error}"
            : $"Already up to date (v{check.CurrentVersion})");
        await host.StopAsync();
        return;
    }

    Console.WriteLine($"Update available: v{check.CurrentVersion} → v{check.LatestVersion}");

    var lastPercent = -1;
    await foreach (var progress in updateService.ApplyUpdateAsync())
    {
        switch (progress.Status)
        {
            case "Downloading":
                if (progress.Percent != lastPercent)
                {
                    Console.Write($"\rDownloading... {progress.Percent}%   ");
                    lastPercent = progress.Percent;
                }
                break;
            case "Extracting":
                Console.WriteLine("\nExtracting...");
                break;
            case "Replacing":
                Console.WriteLine("Replacing files...");
                break;
            case "Restarting":
                Console.WriteLine("Launching new version...");
                break;
            case "Error":
                Console.Error.WriteLine($"\nError: {progress.Error}");
                await host.StopAsync();
                Environment.Exit(1);
                break;
        }
    }

    // Shuts down cleanly once UpdateService calls StopApplication() after 1.5 seconds
    await host.WaitForShutdownAsync();
}
