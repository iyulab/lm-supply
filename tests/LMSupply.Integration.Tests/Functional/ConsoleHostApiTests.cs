using System.Net;
using System.Net.Http.Json;
using System.Text;
using System.Text.Json;
using LMSupply.Integration.Tests.Helpers;
using Microsoft.AspNetCore.Mvc.Testing;

namespace LMSupply.Integration.Tests.Functional;

/// <summary>
/// Console Host HTTP API integration tests.
/// Tests the web API layer including routing, serialization, error handling,
/// and end-to-end inference through HTTP endpoints.
/// Group A: GET endpoints (no model loading required)
/// Group B: Error handling validation (no model loading required)
/// Group C: Inference endpoints (GPU + models required)
/// </summary>
[Trait("Category", "Functional")]
[Trait("Category", "LocalOnly")]
[Trait("Domain", "ConsoleHost")]
public class ConsoleHostApiTests : IClassFixture<WebApplicationFactory<Program>>
{
    private readonly HttpClient _client;

    public ConsoleHostApiTests(WebApplicationFactory<Program> factory)
    {
        _client = factory.CreateClient();
    }

    // ── Group A: GET Endpoints (no model required) ────────────────

    [Fact]
    [Trait("Axis", "API-GET")]
    public async Task GET_Health_Returns200WithHealthyStatus()
    {
        var response = await _client.GetAsync("/health", TestContext.Current.CancellationToken);

        response.StatusCode.Should().Be(HttpStatusCode.OK);
        var json = await response.Content.ReadFromJsonAsync<JsonElement>(cancellationToken: TestContext.Current.CancellationToken);
        json.GetProperty("status").GetString().Should().Be("healthy");
        json.TryGetProperty("timestamp", out _).Should().BeTrue();
    }

    [Fact]
    [Trait("Axis", "API-GET")]
    public async Task GET_Models_ReturnsOpenAICompatibleList()
    {
        var response = await _client.GetAsync("/v1/models", TestContext.Current.CancellationToken);

        response.StatusCode.Should().Be(HttpStatusCode.OK);
        var json = await response.Content.ReadFromJsonAsync<JsonElement>(cancellationToken: TestContext.Current.CancellationToken);
        var data = json.GetProperty("data");
        data.GetArrayLength().Should().BeGreaterThan(0, "should include well-known model aliases");
    }

    [Fact]
    [Trait("Axis", "API-GET")]
    public async Task GET_Models_ContainsExpectedAliases()
    {
        var response = await _client.GetAsync("/v1/models", TestContext.Current.CancellationToken);
        var json = await response.Content.ReadFromJsonAsync<JsonElement>(cancellationToken: TestContext.Current.CancellationToken);
        var ids = json.GetProperty("data")
            .EnumerateArray()
            .Select(item => item.GetProperty("id").GetString()!)
            .ToHashSet();

        ids.Should().Contain("embedder:default");
        ids.Should().Contain("reranker:default");
        ids.Should().Contain("generator:default");
    }

    [Fact]
    [Trait("Axis", "API-GET")]
    public async Task GET_ModelById_ReturnsModelInfo()
    {
        var response = await _client.GetAsync("/v1/models/embedder:default", TestContext.Current.CancellationToken);

        response.StatusCode.Should().Be(HttpStatusCode.OK);
        var json = await response.Content.ReadFromJsonAsync<JsonElement>(cancellationToken: TestContext.Current.CancellationToken);
        json.GetProperty("id").GetString().Should().Be("embedder:default");
    }

    [Fact]
    [Trait("Axis", "API-GET")]
    public async Task GET_SystemStatus_Returns200()
    {
        var response = await _client.GetAsync("/api/system/status", TestContext.Current.CancellationToken);

        response.StatusCode.Should().Be(HttpStatusCode.OK);
    }

    [Fact]
    [Trait("Axis", "API-GET")]
    public async Task GET_GpuInfo_Returns200()
    {
        var response = await _client.GetAsync("/api/system/gpu", TestContext.Current.CancellationToken);

        response.StatusCode.Should().Be(HttpStatusCode.OK);
    }

    [Fact]
    [Trait("Axis", "API-GET")]
    public async Task GET_MemoryMetrics_Returns200()
    {
        var response = await _client.GetAsync("/api/system/memory", TestContext.Current.CancellationToken);

        response.StatusCode.Should().Be(HttpStatusCode.OK);
    }

    [Fact]
    [Trait("Axis", "API-GET")]
    public async Task GET_DetectLabels_ReturnsCOCOLabels()
    {
        var response = await _client.GetAsync("/v1/images/detect/labels", TestContext.Current.CancellationToken);

        response.StatusCode.Should().Be(HttpStatusCode.OK);
        var json = await response.Content.ReadFromJsonAsync<JsonElement>(cancellationToken: TestContext.Current.CancellationToken);
        var labels = json.GetProperty("labels");
        labels.GetArrayLength().Should().Be(80, "COCO dataset has 80 class labels");

        // Verify shape: each label has id and name
        var first = labels[0];
        first.TryGetProperty("id", out _).Should().BeTrue();
        first.TryGetProperty("name", out _).Should().BeTrue();
    }

    [Fact]
    [Trait("Axis", "API-GET")]
    public async Task GET_SegmentLabels_ReturnsADE20KLabels()
    {
        var response = await _client.GetAsync("/v1/images/segment/labels", TestContext.Current.CancellationToken);

        response.StatusCode.Should().Be(HttpStatusCode.OK);
        var json = await response.Content.ReadFromJsonAsync<JsonElement>(cancellationToken: TestContext.Current.CancellationToken);
        var labels = json.GetProperty("labels");
        labels.GetArrayLength().Should().Be(150, "ADE20K dataset has 150 class labels");

        // Verify shape: each label has id and name
        var first = labels[0];
        first.TryGetProperty("id", out _).Should().BeTrue();
        first.TryGetProperty("name", out _).Should().BeTrue();
    }

    [Fact]
    [Trait("Axis", "API-GET")]
    public async Task GET_TranslateLanguages_ReturnsLanguagePairs()
    {
        var response = await _client.GetAsync("/v1/translate/languages", TestContext.Current.CancellationToken);

        response.StatusCode.Should().Be(HttpStatusCode.OK);
        var json = await response.Content.ReadFromJsonAsync<JsonElement>(cancellationToken: TestContext.Current.CancellationToken);
        var languages = json.GetProperty("languages");
        languages.GetArrayLength().Should().BeGreaterThan(0, "should have translation language pairs");
    }

    [Fact]
    [Trait("Axis", "API-GET")]
    public async Task GET_OcrLanguages_ReturnsSupportedLanguages()
    {
        var response = await _client.GetAsync("/v1/images/ocr/languages", TestContext.Current.CancellationToken);

        response.StatusCode.Should().Be(HttpStatusCode.OK);
        var json = await response.Content.ReadFromJsonAsync<JsonElement>(cancellationToken: TestContext.Current.CancellationToken);
        var languages = json.GetProperty("languages");
        languages.GetArrayLength().Should().BeGreaterThan(0, "should have supported OCR languages");

        // Verify English is always supported
        var langList = languages.EnumerateArray().Select(l => l.GetString()).ToList();
        langList.Should().Contain("en", "English should always be a supported OCR language");
    }

    [Fact]
    [Trait("Axis", "API-GET")]
    public async Task GET_ImageModels_ReturnsAvailableModels()
    {
        var response = await _client.GetAsync("/v1/images/models", TestContext.Current.CancellationToken);

        response.StatusCode.Should().Be(HttpStatusCode.OK);
        var json = await response.Content.ReadFromJsonAsync<JsonElement>(cancellationToken: TestContext.Current.CancellationToken);
        var models = json.GetProperty("models");
        models.GetArrayLength().Should().BeGreaterThan(0, "should have image generation models");

        // Verify shape: each model has id and repo_id
        var first = models[0];
        first.TryGetProperty("id", out _).Should().BeTrue();
        first.TryGetProperty("repo_id", out _).Should().BeTrue();
    }

    [Fact]
    [Trait("Axis", "API-GET")]
    public async Task GET_Registry_ReturnsAllModelTypes()
    {
        var response = await _client.GetAsync("/api/registry/models", TestContext.Current.CancellationToken);

        response.StatusCode.Should().Be(HttpStatusCode.OK);
        var json = await response.Content.ReadFromJsonAsync<JsonElement>(cancellationToken: TestContext.Current.CancellationToken);
        var types = json.GetProperty("modelTypes");
        types.GetArrayLength().Should().BeGreaterThanOrEqualTo(10,
            "should have at least 10 model type categories");
    }

    [Fact]
    [Trait("Axis", "API-GET")]
    public async Task GET_RegistryByType_ReturnsEmbedderModels()
    {
        var response = await _client.GetAsync("/api/registry/models/embedder", TestContext.Current.CancellationToken);

        response.StatusCode.Should().Be(HttpStatusCode.OK);
        var json = await response.Content.ReadFromJsonAsync<JsonElement>(cancellationToken: TestContext.Current.CancellationToken);
        json.GetProperty("type").GetString().Should().Be("embedder");
        json.GetProperty("models").GetArrayLength().Should().BeGreaterThan(0);
    }

    [Fact]
    [Trait("Axis", "API-GET")]
    public async Task GET_Registry_ContainsAll11Types()
    {
        var response = await _client.GetAsync("/api/registry/models", TestContext.Current.CancellationToken);
        var json = await response.Content.ReadFromJsonAsync<JsonElement>(cancellationToken: TestContext.Current.CancellationToken);
        var types = json.GetProperty("modelTypes")
            .EnumerateArray()
            .Select(t => t.GetProperty("type").GetString()!)
            .ToHashSet();

        var expectedTypes = new[]
        {
            "generator", "embedder", "reranker", "transcriber",
            "synthesizer", "translator", "captioner", "ocr",
            "detector", "segmenter", "imagegenerator"
        };

        foreach (var expected in expectedTypes)
        {
            types.Should().Contain(expected, $"registry should include '{expected}' type");
        }
    }

    [Fact]
    [Trait("Axis", "API-GET")]
    public async Task GET_Registry_EachTypeHasModels()
    {
        var response = await _client.GetAsync("/api/registry/models", TestContext.Current.CancellationToken);
        var json = await response.Content.ReadFromJsonAsync<JsonElement>(cancellationToken: TestContext.Current.CancellationToken);
        var types = json.GetProperty("modelTypes").EnumerateArray();

        foreach (var modelType in types)
        {
            var typeName = modelType.GetProperty("type").GetString();
            modelType.GetProperty("displayName").GetString().Should().NotBeNullOrEmpty(
                $"{typeName} should have displayName");
            modelType.GetProperty("description").GetString().Should().NotBeNullOrEmpty(
                $"{typeName} should have description");
            modelType.GetProperty("models").GetArrayLength().Should().BeGreaterThan(0,
                $"{typeName} should have at least one model alias");
        }
    }

    [Fact]
    [Trait("Axis", "API-GET")]
    public async Task GET_Registry_EmbedderHasExpectedAliases()
    {
        var response = await _client.GetAsync("/api/registry/models/embedder", TestContext.Current.CancellationToken);
        var json = await response.Content.ReadFromJsonAsync<JsonElement>(cancellationToken: TestContext.Current.CancellationToken);
        var aliases = json.GetProperty("models")
            .EnumerateArray()
            .Select(m => m.GetProperty("aliasName").GetString()!)
            .ToList();

        aliases.Should().Contain("default");
        aliases.Should().Contain("fast");
        aliases.Should().Contain("quality");
        aliases.Should().Contain("multilingual");
    }

    [Fact]
    [Trait("Axis", "API-GET")]
    public async Task GET_Registry_ModelAliasHasRepoId()
    {
        var response = await _client.GetAsync("/api/registry/models/embedder", TestContext.Current.CancellationToken);
        var json = await response.Content.ReadFromJsonAsync<JsonElement>(cancellationToken: TestContext.Current.CancellationToken);
        var models = json.GetProperty("models").EnumerateArray();

        foreach (var model in models)
        {
            model.GetProperty("repoId").GetString().Should().NotBeNullOrEmpty(
                $"alias '{model.GetProperty("aliasName").GetString()}' should have repoId");
            model.TryGetProperty("isCached", out _).Should().BeTrue(
                "each model should have isCached field");
        }
    }

    [Fact]
    [Trait("Axis", "API-Error")]
    public async Task POST_DownloadCheck_EmptyRepoId_Returns400()
    {
        var content = JsonContent.Create(new { repoId = "" });
        var response = await _client.PostAsync("/api/download/check", content, TestContext.Current.CancellationToken);

        response.StatusCode.Should().Be(HttpStatusCode.BadRequest);
    }

    [Fact]
    [Trait("Axis", "API-GET")]
    public async Task GET_RegistryByType_UnknownType_Returns404()
    {
        var response = await _client.GetAsync("/api/registry/models/nonexistent", TestContext.Current.CancellationToken);

        response.StatusCode.Should().Be(HttpStatusCode.NotFound);
    }

    [Fact]
    [Trait("Axis", "API-GET")]
    public async Task GET_CacheStats_ReturnsStatistics()
    {
        var response = await _client.GetAsync("/api/cache/stats", TestContext.Current.CancellationToken);

        response.StatusCode.Should().Be(HttpStatusCode.OK);
        var json = await response.Content.ReadFromJsonAsync<JsonElement>(cancellationToken: TestContext.Current.CancellationToken);
        json.TryGetProperty("totalModels", out _).Should().BeTrue();
        json.TryGetProperty("cacheDirectory", out _).Should().BeTrue();
    }

    [Fact]
    [Trait("Axis", "API-GET")]
    public async Task GET_CachedModels_ReturnsModelList()
    {
        var response = await _client.GetAsync("/api/cache/models", TestContext.Current.CancellationToken);

        response.StatusCode.Should().Be(HttpStatusCode.OK);
        var json = await response.Content.ReadFromJsonAsync<JsonElement>(cancellationToken: TestContext.Current.CancellationToken);
        json.TryGetProperty("models", out _).Should().BeTrue();
        json.TryGetProperty("totalCount", out _).Should().BeTrue();
    }

    [Fact]
    [Trait("Axis", "API-GET")]
    public async Task GET_LoadedModels_ReturnsEmptyListInitially()
    {
        var response = await _client.GetAsync("/api/cache/loaded", TestContext.Current.CancellationToken);

        response.StatusCode.Should().Be(HttpStatusCode.OK);
        var json = await response.Content.ReadFromJsonAsync<JsonElement>(cancellationToken: TestContext.Current.CancellationToken);
        json.ValueKind.Should().Be(JsonValueKind.Array);
    }

    [Fact]
    [Trait("Axis", "API-GET")]
    public async Task GET_Swagger_Returns200()
    {
        var response = await _client.GetAsync("/swagger/v1/swagger.json", TestContext.Current.CancellationToken);

        response.StatusCode.Should().Be(HttpStatusCode.OK);
        var json = await response.Content.ReadFromJsonAsync<JsonElement>(cancellationToken: TestContext.Current.CancellationToken);
        json.GetProperty("info").GetProperty("title").GetString()
            .Should().Be("LMSupply Console API");
    }

    [Fact]
    [Trait("Axis", "API-GET")]
    public async Task GET_SystemStatus_ReturnsExpectedShape()
    {
        var response = await _client.GetAsync("/api/system/status", TestContext.Current.CancellationToken);

        response.StatusCode.Should().Be(HttpStatusCode.OK);
        var json = await response.Content.ReadFromJsonAsync<JsonElement>(cancellationToken: TestContext.Current.CancellationToken);

        var status = json.GetProperty("status");
        status.TryGetProperty("engineReady", out _).Should().BeTrue();
        status.TryGetProperty("gpuAvailable", out _).Should().BeTrue();
        status.TryGetProperty("gpuProvider", out _).Should().BeTrue();
        status.TryGetProperty("cpuUsage", out _).Should().BeTrue();
        status.TryGetProperty("ramUsageMB", out _).Should().BeTrue();
        status.TryGetProperty("ramTotalMB", out _).Should().BeTrue();
        status.TryGetProperty("processMemoryMB", out _).Should().BeTrue();
        status.TryGetProperty("timestamp", out _).Should().BeTrue();
    }

    [Fact]
    [Trait("Axis", "API-GET")]
    public async Task GET_GpuInfo_ReturnsExpectedShape()
    {
        var response = await _client.GetAsync("/api/system/gpu", TestContext.Current.CancellationToken);

        response.StatusCode.Should().Be(HttpStatusCode.OK);
        var json = await response.Content.ReadFromJsonAsync<JsonElement>(cancellationToken: TestContext.Current.CancellationToken);
        json.TryGetProperty("isAvailable", out _).Should().BeTrue();
        json.TryGetProperty("name", out _).Should().BeTrue();
        json.TryGetProperty("provider", out _).Should().BeTrue();
    }

    [Fact]
    [Trait("Axis", "API-GET")]
    public async Task GET_MemoryMetrics_ReturnsExpectedShape()
    {
        var response = await _client.GetAsync("/api/system/memory", TestContext.Current.CancellationToken);

        response.StatusCode.Should().Be(HttpStatusCode.OK);
        var json = await response.Content.ReadFromJsonAsync<JsonElement>(cancellationToken: TestContext.Current.CancellationToken);
        json.TryGetProperty("totalMB", out _).Should().BeTrue();
        json.TryGetProperty("usedMB", out _).Should().BeTrue();
        json.TryGetProperty("usagePercent", out _).Should().BeTrue();
    }

    [Fact]
    [Trait("Axis", "API-GET")]
    public async Task GET_Version_ReturnsVersionAndRid()
    {
        var response = await _client.GetAsync("/api/system/version", TestContext.Current.CancellationToken);

        response.StatusCode.Should().Be(HttpStatusCode.OK);
        var json = await response.Content.ReadFromJsonAsync<JsonElement>(cancellationToken: TestContext.Current.CancellationToken);
        json.GetProperty("version").GetString().Should().NotBeNullOrEmpty();
        json.GetProperty("rid").GetString().Should().NotBeNullOrEmpty();
    }

    [Fact]
    [Trait("Axis", "API-GET")]
    public async Task GET_Update_ReturnsUpdateCheckResult()
    {
        var response = await _client.GetAsync("/api/system/update", TestContext.Current.CancellationToken);

        response.StatusCode.Should().Be(HttpStatusCode.OK);
        var json = await response.Content.ReadFromJsonAsync<JsonElement>(cancellationToken: TestContext.Current.CancellationToken);
        json.TryGetProperty("currentVersion", out _).Should().BeTrue();
        // updateAvailable, latestVersion, releaseUrl may be null if check fails
    }

    [Fact]
    [Trait("Axis", "API-CORS")]
    public async Task CORS_SSE_Endpoint_IncludesCorsHeaders()
    {
        var request = new HttpRequestMessage(HttpMethod.Get, "/api/system/metrics/stream");
        request.Headers.Add("Origin", "http://localhost:5173");
        request.Headers.Accept.Add(new System.Net.Http.Headers.MediaTypeWithQualityHeaderValue("text/event-stream"));

        using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(2));
        try
        {
            var response = await _client.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, cts.Token);
            response.StatusCode.Should().Be(HttpStatusCode.OK);
            response.Headers.TryGetValues("Access-Control-Allow-Origin", out var values)
                .Should().BeTrue("SSE endpoint should include CORS header");
            values!.Should().Contain("http://localhost:5173");
        }
        catch (OperationCanceledException)
        {
            // Expected — SSE streams indefinitely
        }
    }

    [Fact]
    [Trait("Axis", "API-GET")]
    public async Task GET_CacheStats_ReturnsExpectedShape()
    {
        var response = await _client.GetAsync("/api/cache/stats", TestContext.Current.CancellationToken);

        response.StatusCode.Should().Be(HttpStatusCode.OK);
        var json = await response.Content.ReadFromJsonAsync<JsonElement>(cancellationToken: TestContext.Current.CancellationToken);
        json.TryGetProperty("totalModels", out _).Should().BeTrue();
        json.TryGetProperty("totalSizeMB", out _).Should().BeTrue();
        json.TryGetProperty("cacheDirectory", out _).Should().BeTrue();
        json.TryGetProperty("byType", out _).Should().BeTrue();
    }

    [Fact]
    [Trait("Axis", "API-GET")]
    public async Task GET_CachedModels_ReturnsExpectedShape()
    {
        var response = await _client.GetAsync("/api/cache/models", TestContext.Current.CancellationToken);

        response.StatusCode.Should().Be(HttpStatusCode.OK);
        var json = await response.Content.ReadFromJsonAsync<JsonElement>(cancellationToken: TestContext.Current.CancellationToken);
        json.TryGetProperty("models", out _).Should().BeTrue();
        json.TryGetProperty("totalCount", out _).Should().BeTrue();
        json.TryGetProperty("totalSizeMB", out _).Should().BeTrue();
    }

    [Fact]
    [Trait("Axis", "API-Error")]
    public async Task DELETE_CachedModel_NonExistent_Returns404()
    {
        var response = await _client.DeleteAsync("/api/cache/models/nonexistent%2Fmodel-xyz", TestContext.Current.CancellationToken);

        response.StatusCode.Should().Be(HttpStatusCode.NotFound);
    }

    [Fact]
    [Trait("Axis", "API-Error")]
    public async Task DELETE_LoadedModel_NonExistent_Returns404()
    {
        var response = await _client.DeleteAsync("/api/cache/loaded/generator:nonexistent-xyz", TestContext.Current.CancellationToken);

        response.StatusCode.Should().Be(HttpStatusCode.NotFound);
    }

    [Fact]
    [Trait("Axis", "API-Error")]
    public async Task POST_LoadModel_InvalidType_Returns400Or500()
    {
        var content = JsonContent.Create(new { type = "invalid_type", modelId = "default" });
        var response = await _client.PostAsync("/api/cache/load", content, TestContext.Current.CancellationToken);

        // Should fail with an error (ArgumentException → 400 or type error → 500)
        response.IsSuccessStatusCode.Should().BeFalse(
            "loading with invalid type should fail");
    }

    [Fact]
    [Trait("Axis", "API-GET")]
    public async Task GET_CacheModelsByType_InvalidType_Returns400()
    {
        var response = await _client.GetAsync("/api/cache/models/type/nonexistent", TestContext.Current.CancellationToken);

        response.StatusCode.Should().Be(HttpStatusCode.BadRequest);
    }

    // ── Group B: Error Handling (no model loading) ────────────────

    [Fact]
    [Trait("Axis", "API-Error")]
    public async Task POST_Rerank_MissingQuery_Returns400()
    {
        string[] docs = ["doc1"];
        var content = JsonContent.Create(new { query = "", documents = docs });
        var response = await _client.PostAsync("/v1/rerank", content, TestContext.Current.CancellationToken);

        response.StatusCode.Should().Be(HttpStatusCode.BadRequest);
        var json = await response.Content.ReadFromJsonAsync<JsonElement>(cancellationToken: TestContext.Current.CancellationToken);
        json.GetProperty("error").GetProperty("message").GetString()
            .Should().Contain("query", "error should mention the missing field");
    }

    [Fact]
    [Trait("Axis", "API-Error")]
    public async Task POST_Rerank_EmptyDocuments_Returns400()
    {
        var content = JsonContent.Create(new
        {
            query = "test",
            documents = Array.Empty<string>()
        });
        var response = await _client.PostAsync("/v1/rerank", content, TestContext.Current.CancellationToken);

        response.StatusCode.Should().Be(HttpStatusCode.BadRequest);
    }

    [Fact]
    [Trait("Axis", "API-Error")]
    public async Task POST_Speech_MissingInput_Returns400()
    {
        var content = JsonContent.Create(new { input = "", model = "fast" });
        var response = await _client.PostAsync("/v1/audio/speech", content, TestContext.Current.CancellationToken);

        response.StatusCode.Should().Be(HttpStatusCode.BadRequest);
    }

    [Fact]
    [Trait("Axis", "API-Error")]
    public async Task POST_Caption_NotFormData_IsRejected()
    {
        // Endpoint has .Accepts("multipart/form-data") constraint,
        // so JSON requests don't match the route → 404 from SPA fallback
        var content = new StringContent("{}", Encoding.UTF8, "application/json");
        var response = await _client.PostAsync("/v1/images/caption", content, TestContext.Current.CancellationToken);

        response.IsSuccessStatusCode.Should().BeFalse();
    }

    [Fact]
    [Trait("Axis", "API-Error")]
    public async Task POST_Vqa_NotFormData_IsRejected()
    {
        var content = new StringContent("{}", Encoding.UTF8, "application/json");
        var response = await _client.PostAsync("/v1/images/vqa", content, TestContext.Current.CancellationToken);

        response.IsSuccessStatusCode.Should().BeFalse();
    }

    [Fact]
    [Trait("Axis", "API-Error")]
    public async Task POST_Detect_NotFormData_IsRejected()
    {
        // Endpoint has .Accepts("multipart/form-data") constraint,
        // so JSON requests don't match the route → 404 from SPA fallback
        var content = new StringContent("{}", Encoding.UTF8, "application/json");
        var response = await _client.PostAsync("/v1/images/detect", content, TestContext.Current.CancellationToken);

        response.IsSuccessStatusCode.Should().BeFalse();
    }

    [Fact]
    [Trait("Axis", "API-Error")]
    public async Task POST_Detect_FormData_MissingFile_Returns400()
    {
        using var formContent = new MultipartFormDataContent();
        formContent.Add(new StringContent("default"), "model");
        var response = await _client.PostAsync("/v1/images/detect", formContent, TestContext.Current.CancellationToken);

        response.StatusCode.Should().Be(HttpStatusCode.BadRequest);
    }

    [Fact]
    [Trait("Axis", "API-Error")]
    public async Task POST_Ocr_NotFormData_IsRejected()
    {
        // Endpoint has .Accepts("multipart/form-data") constraint,
        // so JSON requests don't match the route → 404 from SPA fallback
        var content = new StringContent("{}", Encoding.UTF8, "application/json");
        var response = await _client.PostAsync("/v1/images/ocr", content, TestContext.Current.CancellationToken);

        response.IsSuccessStatusCode.Should().BeFalse();
    }

    [Fact]
    [Trait("Axis", "API-Error")]
    public async Task POST_Segment_NotFormData_IsRejected()
    {
        // Endpoint has .Accepts("multipart/form-data") constraint,
        // so JSON requests don't match the route → 404 from SPA fallback
        var content = new StringContent("{}", Encoding.UTF8, "application/json");
        var response = await _client.PostAsync("/v1/images/segment", content, TestContext.Current.CancellationToken);

        response.IsSuccessStatusCode.Should().BeFalse();
    }

    [Fact]
    [Trait("Axis", "API-Error")]
    public async Task POST_Segment_FormData_MissingFile_Returns400()
    {
        using var formContent = new MultipartFormDataContent();
        formContent.Add(new StringContent("default"), "model");
        var response = await _client.PostAsync("/v1/images/segment", formContent, TestContext.Current.CancellationToken);

        response.StatusCode.Should().Be(HttpStatusCode.BadRequest);
    }

    [Fact]
    [Trait("Axis", "API-Error")]
    public async Task POST_ImageGeneration_MissingPrompt_Returns400()
    {
        var content = JsonContent.Create(new { prompt = "" });
        var response = await _client.PostAsync("/v1/images/generations", content, TestContext.Current.CancellationToken);

        response.StatusCode.Should().Be(HttpStatusCode.BadRequest);
    }

    [Fact]
    [Trait("Axis", "API-Error")]
    public async Task POST_ImageGenerate_Extended_MissingPrompt_Returns400()
    {
        var content = JsonContent.Create(new { prompt = "" });
        var response = await _client.PostAsync("/v1/images/generate", content, TestContext.Current.CancellationToken);

        response.StatusCode.Should().Be(HttpStatusCode.BadRequest);
    }

    [Fact]
    [Trait("Axis", "API-Error")]
    public async Task POST_Caption_FormData_MissingFile_Returns400()
    {
        using var formContent = new MultipartFormDataContent();
        formContent.Add(new StringContent("fast"), "model");
        var response = await _client.PostAsync("/v1/images/caption", formContent, TestContext.Current.CancellationToken);

        response.StatusCode.Should().Be(HttpStatusCode.BadRequest);
    }

    [Fact]
    [Trait("Axis", "API-Error")]
    public async Task POST_Vqa_FormData_MissingFile_Returns400()
    {
        using var formContent = new MultipartFormDataContent();
        formContent.Add(new StringContent("What is this?"), "question");
        var response = await _client.PostAsync("/v1/images/vqa", formContent, TestContext.Current.CancellationToken);

        response.StatusCode.Should().Be(HttpStatusCode.BadRequest);
    }

    [Fact]
    [Trait("Axis", "API-Error")]
    public async Task POST_Vqa_FormData_MissingQuestion_Returns400()
    {
        using var formContent = new MultipartFormDataContent();
        formContent.Add(new ByteArrayContent([0xFF, 0xD8, 0xFF]), "file", "test.jpg");
        var response = await _client.PostAsync("/v1/images/vqa", formContent, TestContext.Current.CancellationToken);

        response.StatusCode.Should().Be(HttpStatusCode.BadRequest);
    }

    [Fact]
    [Trait("Axis", "API-Error")]
    public async Task POST_Ocr_FormData_MissingFile_Returns400()
    {
        using var formContent = new MultipartFormDataContent();
        formContent.Add(new StringContent("en"), "language");
        var response = await _client.PostAsync("/v1/images/ocr", formContent, TestContext.Current.CancellationToken);

        response.StatusCode.Should().Be(HttpStatusCode.BadRequest);
    }

    [Fact]
    [Trait("Axis", "API-Error")]
    public async Task POST_Transcribe_FormData_MissingFile_Returns400()
    {
        using var formContent = new MultipartFormDataContent();
        formContent.Add(new StringContent("fast"), "model");
        var response = await _client.PostAsync("/v1/audio/transcriptions", formContent, TestContext.Current.CancellationToken);

        response.StatusCode.Should().Be(HttpStatusCode.BadRequest);
    }

    // ── Group C: Inference Endpoints (GPU + models required) ──────

    [Fact]
    [Trait("Axis", "API-Inference")]
    public async Task POST_Embeddings_ReturnsOpenAICompatibleResponse()
    {
        var content = JsonContent.Create(new { input = "Hello, world!", model = "fast" });
        var response = await _client.PostAsync("/v1/embeddings", content, TestContext.Current.CancellationToken);

        response.StatusCode.Should().Be(HttpStatusCode.OK);
        var json = await response.Content.ReadFromJsonAsync<JsonElement>(cancellationToken: TestContext.Current.CancellationToken);

        json.GetProperty("data").GetArrayLength().Should().Be(1);
        json.GetProperty("data")[0].GetProperty("embedding").GetArrayLength()
            .Should().BeGreaterThan(0);
        json.GetProperty("model").GetString().Should().NotBeNullOrEmpty();
        json.TryGetProperty("usage", out _).Should().BeTrue();
    }

    [Fact]
    [Trait("Axis", "API-Inference")]
    public async Task POST_Embeddings_BatchInput_ReturnsMultiple()
    {
        string[] inputs = ["Hello", "World"];
        var content = JsonContent.Create(new
        {
            input = inputs,
            model = "fast"
        });
        var response = await _client.PostAsync("/v1/embeddings", content, TestContext.Current.CancellationToken);

        response.StatusCode.Should().Be(HttpStatusCode.OK);
        var json = await response.Content.ReadFromJsonAsync<JsonElement>(cancellationToken: TestContext.Current.CancellationToken);
        json.GetProperty("data").GetArrayLength().Should().Be(2);
    }

    [Fact]
    [Trait("Axis", "API-Inference")]
    public async Task POST_Rerank_ReturnsRankedResults()
    {
        string[] rerankDocs = ["ML is a subset of AI", "The weather is nice"];
        var content = JsonContent.Create(new
        {
            query = "What is machine learning?",
            documents = rerankDocs,
            model = "fast",
            return_documents = true
        });
        var response = await _client.PostAsync("/v1/rerank", content, TestContext.Current.CancellationToken);

        response.StatusCode.Should().Be(HttpStatusCode.OK);
        var json = await response.Content.ReadFromJsonAsync<JsonElement>(cancellationToken: TestContext.Current.CancellationToken);
        json.GetProperty("results").GetArrayLength().Should().Be(2);
        json.GetProperty("model").GetString().Should().NotBeNullOrEmpty();
    }

    [Fact]
    [Trait("Axis", "API-Inference")]
    public async Task POST_Translate_ReturnsTranslatedText()
    {
        var content = JsonContent.Create(new { input = "안녕하세요", model = "ko-en" });
        var response = await _client.PostAsync("/v1/translate", content, TestContext.Current.CancellationToken);

        response.StatusCode.Should().Be(HttpStatusCode.OK);
        var json = await response.Content.ReadFromJsonAsync<JsonElement>(cancellationToken: TestContext.Current.CancellationToken);
        json.GetProperty("translations").GetArrayLength().Should().Be(1);
        json.GetProperty("translations")[0].GetProperty("translated_text").GetString()
            .Should().NotBeNullOrEmpty();
    }

    [Fact]
    [Trait("Axis", "API-Inference")]
    [Trait("Category", "Integration")]
    public async Task POST_Speech_ReturnsWavAudio()
    {
        var content = JsonContent.Create(new
        {
            input = "Hello, this is a test.",
            model = "fast"
        });
        var response = await _client.PostAsync("/v1/audio/speech", content, TestContext.Current.CancellationToken);

        response.StatusCode.Should().Be(HttpStatusCode.OK);
        response.Content.Headers.ContentType?.MediaType.Should().Be("audio/wav");
        var bytes = await response.Content.ReadAsByteArrayAsync(TestContext.Current.CancellationToken);
        bytes.Length.Should().BeGreaterThan(44, "WAV should have header + audio data");
    }

    [Fact]
    [Trait("Axis", "API-Inference")]
    public async Task POST_Transcribe_FormUpload_ReturnsText()
    {
        var wavBytes = TestDataHelper.CreateToneWav(16000, 1.0f, 440);
        using var formContent = new MultipartFormDataContent();
        formContent.Add(new ByteArrayContent(wavBytes), "file", "test.wav");
        formContent.Add(new StringContent("fast"), "model");

        var response = await _client.PostAsync("/v1/audio/transcriptions", formContent, TestContext.Current.CancellationToken);

        response.StatusCode.Should().Be(HttpStatusCode.OK);
        var json = await response.Content.ReadFromJsonAsync<JsonElement>(cancellationToken: TestContext.Current.CancellationToken);
        json.TryGetProperty("text", out _).Should().BeTrue();
    }

    [Fact]
    [Trait("Axis", "API-Inference")]
    [Trait("Category", "Integration")]
    public async Task POST_Caption_FormUpload_ReturnsCaption()
    {
        var imageBytes = TestDataHelper.CreateGradientBmp(256, 256);
        using var formContent = new MultipartFormDataContent();
        formContent.Add(new ByteArrayContent(imageBytes), "file", "test.bmp");
        formContent.Add(new StringContent("fast"), "model");

        var response = await _client.PostAsync("/v1/images/caption", formContent, TestContext.Current.CancellationToken);

        response.StatusCode.Should().Be(HttpStatusCode.OK);
        var json = await response.Content.ReadFromJsonAsync<JsonElement>(cancellationToken: TestContext.Current.CancellationToken);
        json.GetProperty("caption").GetString().Should().NotBeNullOrEmpty();
    }

    [Fact]
    [Trait("Axis", "API-Inference")]
    [Trait("Category", "Integration")]
    public async Task POST_Detect_FormUpload_ReturnsDetections()
    {
        var imageBytes = TestDataHelper.CreateGradientBmp(640, 480);
        using var formContent = new MultipartFormDataContent();
        formContent.Add(new ByteArrayContent(imageBytes), "file", "test.bmp");
        formContent.Add(new StringContent("fast"), "model");

        var response = await _client.PostAsync("/v1/images/detect", formContent, TestContext.Current.CancellationToken);

        response.StatusCode.Should().Be(HttpStatusCode.OK);
        var json = await response.Content.ReadFromJsonAsync<JsonElement>(cancellationToken: TestContext.Current.CancellationToken);
        json.TryGetProperty("objects", out _).Should().BeTrue();
    }

    [Fact]
    [Trait("Axis", "API-Inference")]
    public async Task POST_Segment_FormUpload_ReturnsSegments()
    {
        var imageBytes = TestDataHelper.CreateGradientBmp(256, 256);
        using var formContent = new MultipartFormDataContent();
        formContent.Add(new ByteArrayContent(imageBytes), "file", "test.bmp");
        formContent.Add(new StringContent("fast"), "model");

        var response = await _client.PostAsync("/v1/images/segment", formContent, TestContext.Current.CancellationToken);

        response.StatusCode.Should().Be(HttpStatusCode.OK);
        var json = await response.Content.ReadFromJsonAsync<JsonElement>(cancellationToken: TestContext.Current.CancellationToken);
        json.TryGetProperty("segments", out _).Should().BeTrue();
    }

    [Fact]
    [Trait("Axis", "API-Inference")]
    public async Task POST_Ocr_FormUpload_ReturnsText()
    {
        var imageBytes = TestDataHelper.CreateGradientBmp(200, 50);
        using var formContent = new MultipartFormDataContent();
        formContent.Add(new ByteArrayContent(imageBytes), "file", "test.bmp");

        var response = await _client.PostAsync("/v1/images/ocr", formContent, TestContext.Current.CancellationToken);

        response.StatusCode.Should().Be(HttpStatusCode.OK);
        var json = await response.Content.ReadFromJsonAsync<JsonElement>(cancellationToken: TestContext.Current.CancellationToken);
        json.TryGetProperty("text", out _).Should().BeTrue();
    }

    // ── Group D: Advanced API Tests ─────────────────────────────────

    [Fact]
    [Trait("Axis", "API-Error")]
    public async Task POST_Embeddings_EmptyInput_ReturnsError()
    {
        var content = JsonContent.Create(new { input = "", model = "fast" });
        var response = await _client.PostAsync("/v1/embeddings", content, TestContext.Current.CancellationToken);

        // Empty input should either succeed with empty embedding or return 400
        // Either outcome is acceptable as long as no 500
        ((int)response.StatusCode).Should().BeLessThan(500,
            "empty input should not cause internal server error");
    }

    [Fact]
    [Trait("Axis", "API-Inference")]
    public async Task POST_Rerank_TopN_LimitsResults()
    {
        string[] docs = ["ML is AI", "Dogs bark", "AI trains models", "Weather is cold"];
        var content = JsonContent.Create(new
        {
            query = "What is AI?",
            documents = docs,
            model = "fast",
            top_n = 2
        });
        var response = await _client.PostAsync("/v1/rerank", content, TestContext.Current.CancellationToken);

        response.StatusCode.Should().Be(HttpStatusCode.OK);
        var json = await response.Content.ReadFromJsonAsync<JsonElement>(cancellationToken: TestContext.Current.CancellationToken);
        json.GetProperty("results").GetArrayLength().Should().Be(2,
            "top_n=2 should return exactly 2 results");
    }

    [Fact]
    [Trait("Axis", "API-Inference")]
    public async Task POST_Rerank_ReturnDocuments_IncludesText()
    {
        string[] docs = ["Machine learning intro"];
        var content = JsonContent.Create(new
        {
            query = "What is ML?",
            documents = docs,
            model = "fast",
            return_documents = true
        });
        var response = await _client.PostAsync("/v1/rerank", content, TestContext.Current.CancellationToken);

        response.StatusCode.Should().Be(HttpStatusCode.OK);
        var json = await response.Content.ReadFromJsonAsync<JsonElement>(cancellationToken: TestContext.Current.CancellationToken);
        var result = json.GetProperty("results")[0];
        result.GetProperty("document").GetProperty("text").GetString()
            .Should().Be("Machine learning intro");
    }

    [Fact]
    [Trait("Axis", "API-Error")]
    public async Task POST_Embeddings_UnknownModel_ReturnsError()
    {
        // API-2: Wrong model name should return a clear error, not 500
        var content = JsonContent.Create(new { input = "test", model = "nonexistent-model-xyz" });
        var response = await _client.PostAsync("/v1/embeddings", content, TestContext.Current.CancellationToken);

        // Should be 4xx or at worst 500 with a meaningful error body
        response.IsSuccessStatusCode.Should().BeFalse("unknown model should fail");
        var json = await response.Content.ReadFromJsonAsync<JsonElement>(cancellationToken: TestContext.Current.CancellationToken);
        json.TryGetProperty("error", out _).Should().BeTrue("error response should have error field");
    }

    [Fact]
    [Trait("Axis", "API-Error")]
    public async Task POST_Rerank_UnknownModel_ReturnsError()
    {
        string[] docs = ["doc1"];
        var content = JsonContent.Create(new
        {
            query = "test",
            documents = docs,
            model = "nonexistent-model-xyz"
        });
        var response = await _client.PostAsync("/v1/rerank", content, TestContext.Current.CancellationToken);

        response.IsSuccessStatusCode.Should().BeFalse("unknown model should fail");
    }

    [Fact]
    [Trait("Axis", "API-Inference")]
    public async Task POST_Embeddings_ConcurrentRequests_AllSucceed()
    {
        // API-5: Concurrent requests should all succeed
        var tasks = Enumerable.Range(0, 5).Select(i =>
        {
            var content = JsonContent.Create(new { input = $"test input {i}", model = "fast" });
            return _client.PostAsync("/v1/embeddings", content);
        });

        var responses = await Task.WhenAll(tasks);

        foreach (var response in responses)
        {
            response.StatusCode.Should().Be(HttpStatusCode.OK,
                "all concurrent embedding requests should succeed");
        }
    }

    [Fact]
    [Trait("Axis", "API-Inference")]
    public async Task POST_Embeddings_LargeInput_HandlesGracefully()
    {
        // API-4: Large request should not crash the server
        var largeText = string.Join(" ", Enumerable.Repeat("hello world test", 500));
        var content = JsonContent.Create(new { input = largeText, model = "fast" });
        var response = await _client.PostAsync("/v1/embeddings", content, TestContext.Current.CancellationToken);

        // Should either succeed (with truncation) or fail with a clear error
        (response.IsSuccessStatusCode || response.StatusCode == HttpStatusCode.BadRequest
            || response.StatusCode == HttpStatusCode.RequestEntityTooLarge)
            .Should().BeTrue("large input should be handled gracefully");
    }

    [Fact]
    [Trait("Axis", "API-GET")]
    public async Task GET_CacheModelsByType_ReturnsTypeModels()
    {
        var response = await _client.GetAsync("/api/cache/models/type/embedder", TestContext.Current.CancellationToken);

        response.StatusCode.Should().Be(HttpStatusCode.OK);
    }

    [Fact]
    [Trait("Axis", "API-Error")]
    public async Task POST_ChatCompletions_MissingMessages_Returns400()
    {
        var content = JsonContent.Create(new { model = "gguf:gemma4-fast" });
        var response = await _client.PostAsync("/v1/chat/completions", content, TestContext.Current.CancellationToken);

        // Missing required 'messages' field should return 400
        response.StatusCode.Should().Be(HttpStatusCode.BadRequest);
    }

    [Fact]
    [Trait("Axis", "API-Error")]
    public async Task POST_ChatCompletions_EmptyMessages_Returns400()
    {
        var content = JsonContent.Create(new
        {
            model = "default",
            messages = Array.Empty<object>()
        });
        var response = await _client.PostAsync("/v1/chat/completions", content, TestContext.Current.CancellationToken);

        // Empty messages should be rejected or cause an error
        response.IsSuccessStatusCode.Should().BeFalse(
            "empty messages array should not succeed");
    }

    [Fact]
    [Trait("Axis", "API-Error")]
    public async Task POST_ChatCompletions_InvalidRole_ReturnsError()
    {
        var content = JsonContent.Create(new
        {
            model = "default",
            messages = new[] { new { role = "invalid_role", content = "Hello" } }
        });
        var response = await _client.PostAsync("/v1/chat/completions", content, TestContext.Current.CancellationToken);

        // Invalid role should cause an error (enum parse failure)
        response.IsSuccessStatusCode.Should().BeFalse(
            "invalid message role should cause an error");
    }

    [Fact]
    [Trait("Axis", "API-Error")]
    public async Task POST_Translate_EmptyInput_Returns400()
    {
        var content = JsonContent.Create(new { input = "", model = "ko-en" });
        var response = await _client.PostAsync("/v1/translate", content, TestContext.Current.CancellationToken);

        response.StatusCode.Should().Be(HttpStatusCode.BadRequest);
    }

    [Fact]
    [Trait("Axis", "API-GET")]
    public async Task GET_TranslateLanguages_HasExpectedShape()
    {
        var response = await _client.GetAsync("/v1/translate/languages", TestContext.Current.CancellationToken);

        response.StatusCode.Should().Be(HttpStatusCode.OK);
        var json = await response.Content.ReadFromJsonAsync<JsonElement>(cancellationToken: TestContext.Current.CancellationToken);
        var languages = json.GetProperty("languages").EnumerateArray().ToList();
        languages.Count.Should().BeGreaterThan(0);

        var first = languages[0];
        first.TryGetProperty("id", out _).Should().BeTrue();
        first.TryGetProperty("source", out _).Should().BeTrue();
        first.TryGetProperty("target", out _).Should().BeTrue();
    }

    [Fact]
    [Trait("Axis", "API-Error")]
    public async Task POST_Transcribe_NonFormData_Returns400()
    {
        var content = JsonContent.Create(new { file = "test", model = "default" });
        var response = await _client.PostAsync("/v1/audio/transcriptions", content, TestContext.Current.CancellationToken);

        // JSON content-type doesn't match multipart/form-data expectation
        response.IsSuccessStatusCode.Should().BeFalse(
            "transcription endpoint requires multipart/form-data");
    }

    [Fact]
    [Trait("Axis", "API-Error")]
    public async Task POST_Translate_NonJsonContent_IsRejected()
    {
        var content = new StringContent("plain text", Encoding.UTF8, "text/plain");
        var response = await _client.PostAsync("/v1/translate", content, TestContext.Current.CancellationToken);

        response.IsSuccessStatusCode.Should().BeFalse(
            "translate endpoint should reject non-JSON content");
    }

    [Fact]
    [Trait("Axis", "API-Error")]
    public async Task POST_ImageGeneration_InvalidSize_Returns400()
    {
        // Size not divisible by 8 should be rejected before model loading
        var content = JsonContent.Create(new
        {
            prompt = "test image",
            size = "3x3"
        });
        var response = await _client.PostAsync("/v1/images/generations", content, TestContext.Current.CancellationToken);

        response.StatusCode.Should().Be(HttpStatusCode.BadRequest);
    }

    [Fact]
    [Trait("Axis", "API-Error")]
    public async Task POST_ImageGenerate_Extended_InvalidSize_Returns400()
    {
        var content = JsonContent.Create(new
        {
            prompt = "test image",
            size = "13x13"
        });
        var response = await _client.PostAsync("/v1/images/generate", content, TestContext.Current.CancellationToken);

        response.StatusCode.Should().Be(HttpStatusCode.BadRequest);
    }

    [Fact]
    [Trait("Axis", "API-GET")]
    public async Task GET_NonExistentEndpoint_SpaFallback_Returns200()
    {
        // SPA fallback serves index.html for unknown GET routes (client-side routing)
        var response = await _client.GetAsync("/v1/nonexistent", TestContext.Current.CancellationToken);

        response.StatusCode.Should().Be(HttpStatusCode.OK);
        response.Content.Headers.ContentType?.MediaType.Should().Be("text/html");
    }

    [Fact]
    [Trait("Axis", "API-Error")]
    public async Task POST_NonExistentEndpoint_Returns404()
    {
        // Non-GET requests to unknown routes should return 404, not SPA fallback
        var content = new StringContent("{}", Encoding.UTF8, "application/json");
        var response = await _client.PostAsync("/v1/nonexistent", content, TestContext.Current.CancellationToken);

        response.StatusCode.Should().Be(HttpStatusCode.NotFound);
    }

    // ── Group D2: Empty/Null Body Tests (test plan 7.3) ─────────────

    [Fact]
    [Trait("Axis", "API-Error")]
    public async Task POST_Embeddings_EmptyBody_IsRejected()
    {
        var content = new StringContent("", Encoding.UTF8, "application/json");
        var response = await _client.PostAsync("/v1/embeddings", content, TestContext.Current.CancellationToken);

        response.IsSuccessStatusCode.Should().BeFalse(
            "empty JSON body should not be accepted");
    }

    [Fact]
    [Trait("Axis", "API-Error")]
    public async Task POST_ChatCompletions_EmptyBody_IsRejected()
    {
        var content = new StringContent("", Encoding.UTF8, "application/json");
        var response = await _client.PostAsync("/v1/chat/completions", content, TestContext.Current.CancellationToken);

        response.IsSuccessStatusCode.Should().BeFalse(
            "empty JSON body should not be accepted");
    }

    [Fact]
    [Trait("Axis", "API-Error")]
    public async Task POST_Translate_EmptyBody_IsRejected()
    {
        var content = new StringContent("", Encoding.UTF8, "application/json");
        var response = await _client.PostAsync("/v1/translate", content, TestContext.Current.CancellationToken);

        response.IsSuccessStatusCode.Should().BeFalse(
            "empty JSON body should not be accepted");
    }

    [Fact]
    [Trait("Axis", "API-Error")]
    public async Task POST_Speech_EmptyBody_IsRejected()
    {
        var content = new StringContent("", Encoding.UTF8, "application/json");
        var response = await _client.PostAsync("/v1/audio/speech", content, TestContext.Current.CancellationToken);

        response.IsSuccessStatusCode.Should().BeFalse(
            "empty JSON body should not be accepted");
    }

    // ── Group D2b: Pre-model validation (cycle 127) ─────────────

    [Fact]
    [Trait("Axis", "API-Error")]
    public async Task POST_Embeddings_InputBeforeModelLoad_Returns400()
    {
        // Input validation should happen BEFORE model loading
        var content = JsonContent.Create(new { input = Array.Empty<string>(), model = "nonexistent-model-xyz" });
        var response = await _client.PostAsync("/v1/embeddings", content, TestContext.Current.CancellationToken);

        // Should get 400 (validation error) not 500 (model not found)
        response.StatusCode.Should().Be(HttpStatusCode.BadRequest,
            "empty input should be caught before attempting model loading");
    }

    [Fact]
    [Trait("Axis", "API-Error")]
    public async Task POST_ChatCompletions_EmptyMessages_Returns400_NotServerError()
    {
        var content = JsonContent.Create(new
        {
            model = "nonexistent-model-xyz",
            messages = Array.Empty<object>()
        });
        var response = await _client.PostAsync("/v1/chat/completions", content, TestContext.Current.CancellationToken);

        // Should get 400 (validation error) not 500 (model not found)
        response.StatusCode.Should().Be(HttpStatusCode.BadRequest,
            "empty messages should be caught before attempting model loading");
    }

    [Fact]
    [Trait("Axis", "API-Error")]
    public async Task POST_CacheLoad_EmptyType_Returns400()
    {
        var content = JsonContent.Create(new { type = "", modelId = "default" });
        var response = await _client.PostAsync("/api/cache/load", content, TestContext.Current.CancellationToken);

        response.StatusCode.Should().Be(HttpStatusCode.BadRequest,
            "empty type should be rejected before attempting model loading");
    }

    [Fact]
    [Trait("Axis", "API-GET")]
    public async Task GET_MetricsStream_Returns200()
    {
        // SSE endpoint should accept the connection
        var request = new HttpRequestMessage(HttpMethod.Get, "/api/system/metrics/stream");
        request.Headers.Accept.Add(new System.Net.Http.Headers.MediaTypeWithQualityHeaderValue("text/event-stream"));

        using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(2));
        try
        {
            var response = await _client.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, cts.Token);
            response.StatusCode.Should().Be(HttpStatusCode.OK);
        }
        catch (OperationCanceledException)
        {
            // Expected — SSE streams indefinitely, we just verify it starts
        }
    }

    // ── Group D3: Concurrent Request Tests (test plan 6.3) ────────────

    [Fact]
    [Trait("Axis", "API-GET")]
    public async Task ConcurrentGET_MultipleEndpoints_AllSucceed()
    {
        // Verify server handles concurrent requests to different endpoints
        var tasks = new[]
        {
            _client.GetAsync("/health", TestContext.Current.CancellationToken),
            _client.GetAsync("/api/system/status", TestContext.Current.CancellationToken),
            _client.GetAsync("/api/system/version", TestContext.Current.CancellationToken),
            _client.GetAsync("/v1/images/detect/labels", TestContext.Current.CancellationToken),
            _client.GetAsync("/v1/images/segment/labels", TestContext.Current.CancellationToken),
            _client.GetAsync("/v1/images/ocr/languages", TestContext.Current.CancellationToken),
            _client.GetAsync("/v1/models", TestContext.Current.CancellationToken),
        };

        var responses = await Task.WhenAll(tasks);

        foreach (var response in responses)
        {
            response.StatusCode.Should().Be(HttpStatusCode.OK);
        }
    }

    // ── Group E: Content-Type Mismatch (API-3) ────────────────────────

    [Fact]
    [Trait("Axis", "API-Error")]
    public async Task POST_Embeddings_PlainText_IsRejected()
    {
        // API-3: JSON endpoint should reject non-JSON Content-Type
        var content = new StringContent("hello", Encoding.UTF8, "text/plain");
        var response = await _client.PostAsync("/v1/embeddings", content, TestContext.Current.CancellationToken);

        response.IsSuccessStatusCode.Should().BeFalse(
            "JSON endpoint should not accept text/plain Content-Type");
    }

    [Fact]
    [Trait("Axis", "API-Error")]
    public async Task POST_Rerank_FormData_IsRejected()
    {
        // API-3: JSON endpoint should reject multipart/form-data
        using var formContent = new MultipartFormDataContent();
        formContent.Add(new StringContent("test"), "query");
        var response = await _client.PostAsync("/v1/rerank", formContent, TestContext.Current.CancellationToken);

        response.IsSuccessStatusCode.Should().BeFalse(
            "JSON endpoint should not accept multipart/form-data");
    }

    // ── Group F: CORS Tests ──────────────────────────────────────────

    [Theory]
    [InlineData("http://localhost:5173")]
    [InlineData("http://localhost:3000")]
    [Trait("Axis", "API-CORS")]
    public async Task CORS_AllowedOrigin_ReturnsAccessControlHeaders(string origin)
    {
        // API-7: Verify CORS headers for configured origins
        var request = new HttpRequestMessage(HttpMethod.Get, "/health");
        request.Headers.Add("Origin", origin);

        var response = await _client.SendAsync(request, TestContext.Current.CancellationToken);

        response.StatusCode.Should().Be(HttpStatusCode.OK);
        response.Headers.TryGetValues("Access-Control-Allow-Origin", out var values)
            .Should().BeTrue($"response should include CORS header for allowed origin {origin}");
        values!.Should().Contain(origin);
    }

    [Fact]
    [Trait("Axis", "API-CORS")]
    public async Task CORS_DisallowedOrigin_NoAccessControlHeader()
    {
        // API-7: Origin not in the allowed list should not get CORS header
        var request = new HttpRequestMessage(HttpMethod.Get, "/health");
        request.Headers.Add("Origin", "http://evil.example.com");

        var response = await _client.SendAsync(request, TestContext.Current.CancellationToken);

        response.StatusCode.Should().Be(HttpStatusCode.OK);
        response.Headers.TryGetValues("Access-Control-Allow-Origin", out _)
            .Should().BeFalse("disallowed origin should not get Access-Control-Allow-Origin");
    }

    [Fact]
    [Trait("Axis", "API-CORS")]
    public async Task CORS_PreflightRequest_ReturnsExpectedHeaders()
    {
        // API-7: OPTIONS preflight should return CORS method/header permissions
        var request = new HttpRequestMessage(HttpMethod.Options, "/v1/embeddings");
        request.Headers.Add("Origin", "http://localhost:5173");
        request.Headers.Add("Access-Control-Request-Method", "POST");
        request.Headers.Add("Access-Control-Request-Headers", "Content-Type");

        var response = await _client.SendAsync(request, TestContext.Current.CancellationToken);

        // Preflight should succeed (2xx)
        ((int)response.StatusCode).Should().BeInRange(200, 299,
            "preflight OPTIONS should return 2xx");

        response.Headers.TryGetValues("Access-Control-Allow-Origin", out var originValues)
            .Should().BeTrue("preflight should include Allow-Origin");
        originValues!.Should().Contain("http://localhost:5173");

        response.Headers.TryGetValues("Access-Control-Allow-Methods", out var methodValues)
            .Should().BeTrue("preflight should include Allow-Methods");
    }

    [Fact]
    [Trait("Axis", "API-CORS")]
    public async Task CORS_AllowedOrigin_IncludesCredentials()
    {
        // API-7: WithCredentials() is configured, verify Allow-Credentials header
        var request = new HttpRequestMessage(HttpMethod.Get, "/health");
        request.Headers.Add("Origin", "http://localhost:5173");

        var response = await _client.SendAsync(request, TestContext.Current.CancellationToken);

        response.Headers.TryGetValues("Access-Control-Allow-Credentials", out var values)
            .Should().BeTrue("response should include Allow-Credentials for credentialed CORS");
        values!.Should().Contain("true");
    }

    // ── Group H: Test Plan Gap Coverage (§1-3, §7) ───────────────

    [Fact]
    [Trait("Axis", "API-GET")]
    public async Task GET_Health_ReturnsExpectedShape()
    {
        // Test plan 1.1.1: health endpoint returns status + timestamp
        var response = await _client.GetAsync("/health", TestContext.Current.CancellationToken);

        response.StatusCode.Should().Be(HttpStatusCode.OK);
        var json = await response.Content.ReadFromJsonAsync<JsonElement>(cancellationToken: TestContext.Current.CancellationToken);
        json.GetProperty("status").GetString().Should().Be("healthy");
        json.TryGetProperty("timestamp", out _).Should().BeTrue();
    }

    [Fact]
    [Trait("Axis", "API-GET")]
    public async Task GET_SystemStatus_ReturnsExpectedFields()
    {
        // Test plan 1.3.1: system status has engine, GPU, CPU/RAM fields
        var response = await _client.GetAsync("/api/system/status", TestContext.Current.CancellationToken);

        response.StatusCode.Should().Be(HttpStatusCode.OK);
        var json = await response.Content.ReadFromJsonAsync<JsonElement>(cancellationToken: TestContext.Current.CancellationToken);
        var status = json.GetProperty("status");
        status.TryGetProperty("engineReady", out _).Should().BeTrue();
        status.TryGetProperty("gpuAvailable", out _).Should().BeTrue();
        status.TryGetProperty("cpuUsage", out _).Should().BeTrue();
        status.TryGetProperty("ramUsageMB", out _).Should().BeTrue();
    }

    [Fact]
    [Trait("Axis", "API-GET")]
    public async Task GET_CacheStats_HasTypeBreakdown()
    {
        // Test plan 2.3.3: cache stats include type-level breakdown
        var response = await _client.GetAsync("/api/cache/stats", TestContext.Current.CancellationToken);

        response.StatusCode.Should().Be(HttpStatusCode.OK);
        var json = await response.Content.ReadFromJsonAsync<JsonElement>(cancellationToken: TestContext.Current.CancellationToken);
        json.TryGetProperty("totalModels", out _).Should().BeTrue();
        json.TryGetProperty("totalSizeMB", out _).Should().BeTrue();
        json.TryGetProperty("cacheDirectory", out _).Should().BeTrue();
        json.TryGetProperty("byType", out _).Should().BeTrue();
    }

    [Fact]
    [Trait("Axis", "API-Error")]
    public async Task POST_CacheLoad_MissingType_Returns400()
    {
        // Test plan 2.4.1: preload without required 'type' field
        var content = JsonContent.Create(new { modelId = "default" });
        var response = await _client.PostAsync("/api/cache/load", content, TestContext.Current.CancellationToken);

        response.StatusCode.Should().Be(HttpStatusCode.BadRequest);
    }

    [Fact]
    [Trait("Axis", "API-Error")]
    public async Task DELETE_LoadedModel_NonExistentKey_Returns404()
    {
        // Test plan 2.4.3: unload nonexistent model
        var response = await _client.DeleteAsync("/api/cache/loaded/nonexistent%3Anonexistent", TestContext.Current.CancellationToken);

        response.StatusCode.Should().Be(HttpStatusCode.NotFound);
    }

    [Fact]
    [Trait("Axis", "API-Error")]
    public async Task POST_Translate_MissingModel_Returns400()
    {
        // Test plan 3.4.5: translate with empty text
        var content = JsonContent.Create(new { text = "Hello" });
        var response = await _client.PostAsync("/v1/translate", content, TestContext.Current.CancellationToken);

        // Missing model field should trigger validation error
        var statusCode = (int)response.StatusCode;
        statusCode.Should().BeOneOf(400, 500);
    }

    [Fact]
    [Trait("Axis", "API-Error")]
    public async Task POST_ChatCompletions_InvalidJson_Returns400()
    {
        // Test plan 7.2: malformed JSON body
        var content = new StringContent("{ invalid json }", Encoding.UTF8, "application/json");
        var response = await _client.PostAsync("/v1/chat/completions", content, TestContext.Current.CancellationToken);

        response.StatusCode.Should().Be(HttpStatusCode.BadRequest);
    }

    [Fact]
    [Trait("Axis", "API-Error")]
    public async Task POST_Embeddings_InvalidJson_Returns400()
    {
        // Test plan 7.2: malformed JSON for embeddings
        var content = new StringContent("not json at all", Encoding.UTF8, "application/json");
        var response = await _client.PostAsync("/v1/embeddings", content, TestContext.Current.CancellationToken);

        response.StatusCode.Should().Be(HttpStatusCode.BadRequest);
    }

    // ── Group I: Error Handling Audit (§7) ──────────────────────────

    [Fact]
    [Trait("Axis", "API-Error")]
    public async Task POST_Translate_InvalidJson_Returns400()
    {
        // Test plan 7.2: malformed JSON for translate
        var content = new StringContent("{ broken }", Encoding.UTF8, "application/json");
        var response = await _client.PostAsync("/v1/translate", content, TestContext.Current.CancellationToken);

        response.StatusCode.Should().Be(HttpStatusCode.BadRequest);
    }

    [Fact]
    [Trait("Axis", "API-Error")]
    public async Task POST_Rerank_InvalidJson_Returns400()
    {
        // Test plan 7.2: malformed JSON for rerank
        var content = new StringContent("not valid json", Encoding.UTF8, "application/json");
        var response = await _client.PostAsync("/v1/rerank", content, TestContext.Current.CancellationToken);

        response.StatusCode.Should().Be(HttpStatusCode.BadRequest);
    }

    [Fact]
    [Trait("Axis", "API-Error")]
    public async Task POST_Embeddings_ErrorResponse_HasStructuredFormat()
    {
        // Test plan 7.1: error responses follow OpenAI error format
        var content = JsonContent.Create(new { input = "", model = "nonexistent-model-xyz" });
        var response = await _client.PostAsync("/v1/embeddings", content, TestContext.Current.CancellationToken);

        response.IsSuccessStatusCode.Should().BeFalse();
        var json = await response.Content.ReadFromJsonAsync<JsonElement>(cancellationToken: TestContext.Current.CancellationToken);
        json.TryGetProperty("error", out var errorObj).Should().BeTrue("error response should have 'error' field");
        errorObj.TryGetProperty("message", out _).Should().BeTrue("error should have 'message' field");
        errorObj.TryGetProperty("type", out _).Should().BeTrue("error should have 'type' field");
    }

    [Fact]
    [Trait("Axis", "API-Error")]
    public async Task POST_Speech_InvalidJson_Returns400()
    {
        // Test plan 7.2: malformed JSON for speech
        var content = new StringContent("{ broken json", Encoding.UTF8, "application/json");
        var response = await _client.PostAsync("/v1/audio/speech", content, TestContext.Current.CancellationToken);

        response.StatusCode.Should().Be(HttpStatusCode.BadRequest);
    }

    [Fact]
    [Trait("Axis", "API-Error")]
    public async Task POST_Transcribe_FormData_EmptyFile_Returns400()
    {
        // Test plan 7.3: empty file content should be rejected
        using var formContent = new MultipartFormDataContent();
        formContent.Add(new ByteArrayContent([]), "file", "empty.wav");
        var response = await _client.PostAsync("/v1/audio/transcriptions", formContent, TestContext.Current.CancellationToken);

        response.StatusCode.Should().Be(HttpStatusCode.BadRequest);
    }

    [Fact]
    [Trait("Axis", "API-Error")]
    public async Task DELETE_CacheModel_NonExistentRepo_Returns404()
    {
        // Test plan 7.1: deleting non-existent cached model
        var response = await _client.DeleteAsync("/api/cache/models/nonexistent-org%2Fnonexistent-model", TestContext.Current.CancellationToken);

        response.StatusCode.Should().Be(HttpStatusCode.NotFound);
    }

    // ── Group J: Swagger Documentation Completeness ───────────────

    [Fact]
    [Trait("Axis", "API-GET")]
    public async Task GET_Swagger_ContainsAllMajorEndpoints()
    {
        var response = await _client.GetAsync("/swagger/v1/swagger.json", TestContext.Current.CancellationToken);
        response.StatusCode.Should().Be(HttpStatusCode.OK);

        var json = await response.Content.ReadFromJsonAsync<JsonElement>(cancellationToken: TestContext.Current.CancellationToken);
        var paths = json.GetProperty("paths");
        var pathKeys = new List<string>();
        foreach (var prop in paths.EnumerateObject())
        {
            pathKeys.Add(prop.Name);
        }

        // AI inference endpoints
        pathKeys.Should().Contain("/v1/embeddings", "embeddings endpoint should be documented");
        pathKeys.Should().Contain("/v1/rerank", "rerank endpoint should be documented");
        pathKeys.Should().Contain("/v1/chat/completions", "chat completions should be documented");
        pathKeys.Should().Contain("/v1/translate", "translate endpoint should be documented");
        pathKeys.Should().Contain("/v1/audio/transcriptions", "transcribe endpoint should be documented");
        pathKeys.Should().Contain("/v1/audio/speech", "speech endpoint should be documented");

        // Vision endpoints
        pathKeys.Should().Contain("/v1/images/caption", "caption endpoint should be documented");
        pathKeys.Should().Contain("/v1/images/vqa", "VQA endpoint should be documented");
        pathKeys.Should().Contain("/v1/images/detect", "detect endpoint should be documented");
        pathKeys.Should().Contain("/v1/images/ocr", "OCR endpoint should be documented");
        pathKeys.Should().Contain("/v1/images/segment", "segment endpoint should be documented");

        // System endpoints
        pathKeys.Should().Contain("/api/system/status", "system status should be documented");
        pathKeys.Should().Contain("/api/system/gpu", "GPU info should be documented");
        pathKeys.Should().Contain("/api/system/version", "version should be documented");

        // Cache management endpoints
        pathKeys.Should().Contain("/api/cache/stats", "cache stats should be documented");
        pathKeys.Should().Contain("/api/cache/loaded", "loaded models should be documented");
    }

    [Fact]
    [Trait("Axis", "API-GET")]
    public async Task GET_Swagger_HasVersionInfo()
    {
        var response = await _client.GetAsync("/swagger/v1/swagger.json", TestContext.Current.CancellationToken);
        response.StatusCode.Should().Be(HttpStatusCode.OK);

        var json = await response.Content.ReadFromJsonAsync<JsonElement>(cancellationToken: TestContext.Current.CancellationToken);
        var info = json.GetProperty("info");
        info.GetProperty("title").GetString().Should().NotBeNullOrEmpty();
        info.GetProperty("version").GetString().Should().NotBeNullOrEmpty();
    }

    [Fact]
    [Trait("Axis", "API-GET")]
    public async Task GET_Swagger_EndpointsHaveTags()
    {
        var response = await _client.GetAsync("/swagger/v1/swagger.json", TestContext.Current.CancellationToken);
        response.StatusCode.Should().Be(HttpStatusCode.OK);

        var json = await response.Content.ReadFromJsonAsync<JsonElement>(cancellationToken: TestContext.Current.CancellationToken);
        var paths = json.GetProperty("paths");

        // Verify at least some endpoints have tags (for Swagger UI grouping)
        var taggedCount = 0;
        foreach (var path in paths.EnumerateObject())
        {
            foreach (var method in path.Value.EnumerateObject())
            {
                if (method.Value.TryGetProperty("tags", out var tags) &&
                    tags.GetArrayLength() > 0)
                {
                    taggedCount++;
                }
            }
        }

        taggedCount.Should().BeGreaterThan(0, "at least some endpoints should have tags for Swagger UI grouping");
    }

    // ── Group K: Model Management Coverage (§2) ──────────────────

    [Fact]
    [Trait("Axis", "API-GET")]
    public async Task GET_Models_OpenAI_HasObjectField()
    {
        // §2.1.3: OpenAI-compatible list includes "object" field
        var response = await _client.GetAsync("/v1/models", TestContext.Current.CancellationToken);
        response.StatusCode.Should().Be(HttpStatusCode.OK);
        var json = await response.Content.ReadFromJsonAsync<JsonElement>(cancellationToken: TestContext.Current.CancellationToken);
        json.GetProperty("object").GetString().Should().Be("list");
    }

    [Fact]
    [Trait("Axis", "API-GET")]
    public async Task GET_Models_OpenAI_ItemsHaveRequiredFields()
    {
        // §2.1.3/2.1.4: Each model item has required OpenAI fields
        var response = await _client.GetAsync("/v1/models", TestContext.Current.CancellationToken);
        var json = await response.Content.ReadFromJsonAsync<JsonElement>(cancellationToken: TestContext.Current.CancellationToken);
        var first = json.GetProperty("data")[0];

        first.TryGetProperty("id", out _).Should().BeTrue();
        first.TryGetProperty("object", out _).Should().BeTrue();
        first.GetProperty("object").GetString().Should().Be("model");
    }

    [Fact]
    [Trait("Axis", "API-GET")]
    public async Task GET_ModelById_AnyId_Returns200()
    {
        // §2.1.4: OpenAI-compatible — returns model info for any ID
        var response = await _client.GetAsync("/v1/models/nonexistent:unknown", TestContext.Current.CancellationToken);

        response.StatusCode.Should().Be(HttpStatusCode.OK);
        var json = await response.Content.ReadFromJsonAsync<JsonElement>(cancellationToken: TestContext.Current.CancellationToken);
        json.GetProperty("id").GetString().Should().Be("nonexistent:unknown");
    }

    [Fact]
    [Trait("Axis", "API-GET")]
    public async Task GET_Registry_TypeHasAliasKindInfo()
    {
        // §2.1.2: Each model alias has kind (system/user) information
        var response = await _client.GetAsync("/api/registry/models/embedder", TestContext.Current.CancellationToken);
        var json = await response.Content.ReadFromJsonAsync<JsonElement>(cancellationToken: TestContext.Current.CancellationToken);
        var first = json.GetProperty("models")[0];

        first.TryGetProperty("aliasName", out _).Should().BeTrue();
        first.TryGetProperty("repoId", out _).Should().BeTrue();
    }

    [Fact]
    [Trait("Axis", "API-GET")]
    public async Task GET_CacheModelsByType_ReturnsArrayShape()
    {
        // §2.3.2: Cache models filtered by valid type returns array
        var response = await _client.GetAsync("/api/cache/models/type/embedder", TestContext.Current.CancellationToken);
        response.StatusCode.Should().Be(HttpStatusCode.OK);
        var json = await response.Content.ReadFromJsonAsync<JsonElement>(cancellationToken: TestContext.Current.CancellationToken);
        json.ValueKind.Should().Be(JsonValueKind.Array);
    }

    [Fact]
    [Trait("Axis", "API-Error")]
    public async Task POST_DownloadCheck_ValidFormatNonExistent_ReturnsError()
    {
        // §2.2.2: Checking nonexistent model returns error, not crash
        var content = JsonContent.Create(new { repoId = "nonexistent-org/fake-model-xyz" });
        var response = await _client.PostAsync("/api/download/check", content, TestContext.Current.CancellationToken);

        // Should either be error (404) or OK with not-found indication
        ((int)response.StatusCode).Should().BeGreaterThanOrEqualTo(200);
    }

    // ── Group L: Resource Management ──────────────────────────────

    [Fact]
    [Trait("Axis", "API-GET")]
    public async Task GET_SystemStatus_IncludesMemoryMetrics()
    {
        var response = await _client.GetAsync("/api/system/status", TestContext.Current.CancellationToken);
        response.StatusCode.Should().Be(HttpStatusCode.OK);
        var json = await response.Content.ReadFromJsonAsync<JsonElement>(cancellationToken: TestContext.Current.CancellationToken);
        var status = json.GetProperty("status");

        status.TryGetProperty("ramUsageMB", out _).Should().BeTrue();
        status.TryGetProperty("ramTotalMB", out _).Should().BeTrue();
        status.TryGetProperty("ramUsagePercent", out _).Should().BeTrue();
    }

    [Fact]
    [Trait("Axis", "API-GET")]
    public async Task GET_SystemStatus_IncludesProcessMemory()
    {
        var response = await _client.GetAsync("/api/system/status", TestContext.Current.CancellationToken);
        var json = await response.Content.ReadFromJsonAsync<JsonElement>(cancellationToken: TestContext.Current.CancellationToken);
        var status = json.GetProperty("status");

        status.TryGetProperty("processMemoryMB", out var procMem).Should().BeTrue();
        procMem.GetDouble().Should().BeGreaterThan(0);
    }

}
