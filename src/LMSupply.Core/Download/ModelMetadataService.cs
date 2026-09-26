using System.Diagnostics;
using System.Net;
using System.Net.Http.Headers;
using System.Text.Json;
using System.Text.Json.Serialization;
using LMSupply.Json;

namespace LMSupply.Download;

/// <summary>
/// Service for fetching and caching model metadata from HuggingFace Hub.
/// </summary>
public sealed class ModelMetadataService : IDisposable
{
    private readonly HttpClient _httpClient;
    private readonly string? _cacheDir;
    private bool _disposed;

    private const string ApiBaseUrl = "https://huggingface.co/api/models";
    private const string MetadataFileName = ".metadata.json";

    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        TypeInfoResolver = CoreJsonContext.Default,
        PropertyNameCaseInsensitive = true,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull
    };

    /// <summary>
    /// Initializes a new ModelMetadataService.
    /// </summary>
    /// <param name="cacheDir">Cache directory for storing metadata. If null, metadata won't be cached locally.</param>
    /// <param name="hfToken">Optional HuggingFace API token for private repositories.</param>
    public ModelMetadataService(string? cacheDir = null, string? hfToken = null)
    {
        _cacheDir = cacheDir;

        var handler = new HttpClientHandler
        {
            AutomaticDecompression = DecompressionMethods.All
        };

        _httpClient = new HttpClient(handler)
        {
            Timeout = TimeSpan.FromSeconds(30)
        };

        _httpClient.DefaultRequestHeaders.UserAgent.Add(
            new ProductInfoHeaderValue("LMSupply", "1.0"));

        var token = hfToken ?? Environment.GetEnvironmentVariable("HF_TOKEN");
        if (!string.IsNullOrEmpty(token))
        {
            _httpClient.DefaultRequestHeaders.Authorization =
                new AuthenticationHeaderValue("Bearer", token);
        }
    }

    /// <summary>
    /// Fetches model metadata from HuggingFace Hub API.
    /// </summary>
    /// <param name="repoId">The repository ID (e.g., "BAAI/bge-small-en-v1.5").</param>
    /// <param name="useCache">Whether to use cached metadata if available.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Model metadata, or null if not available.</returns>
    public async Task<ModelMetadata?> GetMetadataAsync(
        string repoId,
        bool useCache = true,
        CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(repoId);

        // Try cache first
        if (useCache && _cacheDir is not null)
        {
            var cached = TryLoadFromCache(repoId);
            if (cached is not null)
                return cached;
        }

        // Fetch from API
        try
        {
            var metadata = await FetchFromApiAsync(repoId, cancellationToken);
            if (metadata is not null && _cacheDir is not null)
            {
                await SaveToCacheAsync(repoId, metadata, cancellationToken);
            }
            return metadata;
        }
        catch (HttpRequestException)
        {
            // API failure is non-fatal, return null
            return null;
        }
        catch (TaskCanceledException) when (!cancellationToken.IsCancellationRequested)
        {
            // Timeout is non-fatal
            return null;
        }
    }

    /// <summary>
    /// Loads cached metadata for a model without making API calls.
    /// </summary>
    /// <param name="repoId">The repository ID.</param>
    /// <returns>Cached metadata, or null if not available.</returns>
    public ModelMetadata? GetCachedMetadata(string repoId)
    {
        if (_cacheDir is null)
            return null;

        return TryLoadFromCache(repoId);
    }

    /// <summary>
    /// Refreshes metadata for a model by fetching from API and updating cache.
    /// </summary>
    public async Task<ModelMetadata?> RefreshMetadataAsync(
        string repoId,
        CancellationToken cancellationToken = default)
    {
        return await GetMetadataAsync(repoId, useCache: false, cancellationToken);
    }

    private async Task<ModelMetadata?> FetchFromApiAsync(
        string repoId,
        CancellationToken cancellationToken)
    {
        var url = $"{ApiBaseUrl}/{repoId}";

        var response = await _httpClient.GetAsync(url, cancellationToken);

        if (response.StatusCode == HttpStatusCode.NotFound)
            return null;

        if (response.StatusCode == HttpStatusCode.Unauthorized)
            return null; // Private repo without token

        response.EnsureSuccessStatusCode();

        var json = await response.Content.ReadAsStringAsync(cancellationToken);
        var apiResponse = JsonSerializer.Deserialize<HfApiResponse>(json, JsonOptions);

        if (apiResponse is null)
            return null;

        return MapToMetadata(apiResponse);
    }

    private static ModelMetadata MapToMetadata(HfApiResponse api)
    {
        // Extract license from tags if not directly available
        var license = api.Tags?
            .FirstOrDefault(t => t.StartsWith("license:", StringComparison.OrdinalIgnoreCase))
            ?.Replace("license:", "", StringComparison.OrdinalIgnoreCase);

        // Parse lastModified
        DateTime? lastModified = null;
        if (!string.IsNullOrEmpty(api.LastModified) &&
            DateTime.TryParse(api.LastModified, out var parsed))
        {
            lastModified = parsed;
        }

        return new ModelMetadata
        {
            Description = api.CardData?.Description ?? api.Description,
            License = license,
            Author = api.Author,
            Downloads = api.Downloads,
            Likes = api.Likes,
            Tags = api.Tags ?? [],
            PipelineTag = api.PipelineTag,
            LibraryName = api.LibraryName,
            LastModifiedOnHub = lastModified,
            FetchedAt = DateTime.UtcNow,
            IsGated = api.Gated ?? false,
            IsPrivate = api.Private ?? false
        };
    }

    private string GetMetadataFilePath(string repoId)
    {
        var sanitizedRepoId = repoId.Replace("/", "--");
        return Path.Combine(_cacheDir!, $"models--{sanitizedRepoId}", MetadataFileName);
    }

    private ModelMetadata? TryLoadFromCache(string repoId)
    {
        var filePath = GetMetadataFilePath(repoId);
        if (!File.Exists(filePath))
            return null;

        try
        {
            var json = File.ReadAllText(filePath);
            return JsonSerializer.Deserialize<ModelMetadata>(json, JsonOptions);
        }
        catch (Exception ex)
        {
            Trace.TraceInformation($"[ModelMetadataService] Cache load failed: {ex.Message}");
            return null;
        }
    }

    private async Task SaveToCacheAsync(
        string repoId,
        ModelMetadata metadata,
        CancellationToken cancellationToken)
    {
        var filePath = GetMetadataFilePath(repoId);
        var directory = Path.GetDirectoryName(filePath);

        if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
            Directory.CreateDirectory(directory);

        try
        {
            var json = JsonSerializer.Serialize(metadata, JsonOptions);
            await File.WriteAllTextAsync(filePath, json, cancellationToken);
        }
        catch (Exception ex)
        {
            Trace.TraceInformation($"[ModelMetadataService] Cache write failed: {ex.Message}");
        }
    }

    public void Dispose()
    {
        if (_disposed) return;
        _httpClient.Dispose();
        _disposed = true;
    }

    /// <summary>
    /// Internal class for deserializing HuggingFace API response.
    /// </summary>
    internal sealed class HfApiResponse
    {
        [JsonPropertyName("id")]
        public string? Id { get; set; }

        [JsonPropertyName("author")]
        public string? Author { get; set; }

        [JsonPropertyName("downloads")]
        public long Downloads { get; set; }

        [JsonPropertyName("likes")]
        public long Likes { get; set; }

        [JsonPropertyName("tags")]
        public List<string>? Tags { get; set; }

        [JsonPropertyName("pipeline_tag")]
        public string? PipelineTag { get; set; }

        [JsonPropertyName("library_name")]
        public string? LibraryName { get; set; }

        [JsonPropertyName("lastModified")]
        public string? LastModified { get; set; }

        [JsonPropertyName("gated")]
        public bool? Gated { get; set; }

        [JsonPropertyName("private")]
        public bool? Private { get; set; }

        [JsonPropertyName("description")]
        public string? Description { get; set; }

        [JsonPropertyName("cardData")]
        public CardData? CardData { get; set; }
    }

    internal sealed class CardData
    {
        [JsonPropertyName("description")]
        public string? Description { get; set; }
    }
}
