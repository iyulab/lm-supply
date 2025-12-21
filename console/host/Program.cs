using System.Reflection;
using System.Text.Json;
using System.Text.Json.Serialization;
using Microsoft.Extensions.FileProviders;
using LMSupply.Console.Host.Endpoints;
using LMSupply.Console.Host.Services;

var builder = WebApplication.CreateBuilder(args);

// JSON 직렬화 설정
builder.Services.ConfigureHttpJsonOptions(options =>
{
    options.SerializerOptions.PropertyNamingPolicy = JsonNamingPolicy.CamelCase;
    options.SerializerOptions.DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull;
    options.SerializerOptions.Converters.Add(new JsonStringEnumConverter(JsonNamingPolicy.CamelCase));
});

// CORS 설정 (개발용)
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
});

// 서비스 등록
builder.Services.AddSingleton<CacheService>();
builder.Services.AddSingleton<SystemMonitorService>();
builder.Services.AddSingleton<ModelManagerService>();
builder.Services.AddSingleton<DownloadService>();

var app = builder.Build();

// Swagger UI
app.UseSwagger();
app.UseSwaggerUI(c =>
{
    c.SwaggerEndpoint("/swagger/v1/swagger.json", "LMSupply Console API v1");
    c.RoutePrefix = "swagger";
});

app.UseCors();

// 임베디드 리소스에서 정적 파일 제공
var assembly = Assembly.GetExecutingAssembly();
var embeddedProvider = new ManifestEmbeddedFileProvider(assembly, "wwwroot");
var hasEmbeddedFiles = embeddedProvider.GetDirectoryContents("/").Exists;

if (hasEmbeddedFiles)
{
    app.UseDefaultFiles(new DefaultFilesOptions { FileProvider = embeddedProvider });
    app.UseStaticFiles(new StaticFileOptions { FileProvider = embeddedProvider });
}

// API 엔드포인트 매핑
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
app.MapModelRegistryEndpoints();

// Health check
app.MapGet("/health", () => Results.Ok(new { status = "healthy", timestamp = DateTime.UtcNow }));

// 루트 엔드포인트
app.MapGet("/", () =>
{
    var indexFile = embeddedProvider.GetFileInfo("index.html");
    if (indexFile.Exists)
    {
        return Results.Stream(indexFile.CreateReadStream(), "text/html");
    }
    return Results.Redirect("/swagger");
});

// SPA 폴백: API 이외의 경로는 index.html로 (클라이언트 사이드 라우팅)
app.MapFallback(() =>
{
    var indexFile = embeddedProvider.GetFileInfo("index.html");
    if (indexFile.Exists)
    {
        return Results.Stream(indexFile.CreateReadStream(), "text/html");
    }
    return Results.NotFound();
});

app.Run();
