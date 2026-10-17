using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.Json.Serialization;
using LMSupply.Json;

namespace LMSupply.Llama.Server;

public sealed partial class LlamaServerClient
{
    /// <summary>
    /// The <c>architecture.output_modalities</c> the server reports for its model on <c>GET /v1/models</c> — what the
    /// model can produce (<c>"text"</c>; <c>"decisions"</c> for a native decision model). Null when the server does not
    /// report an <c>architecture</c> (older builds), so the caller keeps its legacy behavior.
    /// </summary>
    internal Task<IReadOnlyList<string>?> GetOutputModalitiesAsync(CancellationToken cancellationToken = default) =>
        GuardAsync(() => GetOutputModalitiesCoreAsync(cancellationToken));

    /// <summary>
    /// Sends a typed decision request to <c>POST /v1/systemone</c> and returns the response body. The request is the
    /// caller's (state and questions); a refusal (400 for an invalid request, 501 for a model that is not a decision
    /// model) throws <see cref="LMSupply.Exceptions.InferenceBackendException"/> with the server's body.
    /// </summary>
    internal Task<JsonObject> DecideAsync(JsonObject request, CancellationToken cancellationToken = default) =>
        GuardAsync(() => DecideCoreAsync(request, cancellationToken));

    private async Task<IReadOnlyList<string>?> GetOutputModalitiesCoreAsync(CancellationToken cancellationToken)
    {
        using var response = await SendWithTimeoutAsync(
            new HttpRequestMessage(HttpMethod.Get, $"{_baseUrl}/v1/models"),
            HttpCompletionOption.ResponseContentRead,
            cancellationToken);

        await EnsureSuccessOrThrowContextExceptionAsync(response, _maxContextLength, cancellationToken);

        var json = await response.Content.ReadAsStringAsync(cancellationToken);
        var models = JsonSerializer.Deserialize(json, JsonOptions.TypeInfo<ModelsResponse>());
        return models?.Data?.FirstOrDefault()?.Architecture?.OutputModalities;
    }

    private async Task<JsonObject> DecideCoreAsync(JsonObject request, CancellationToken cancellationToken)
    {
        using var content = new StringContent(request.ToJsonString(), Encoding.UTF8, "application/json");
        using var response = await PostWithTimeoutAsync($"{_baseUrl}/v1/systemone", content, cancellationToken);

        await EnsureSuccessOrThrowContextExceptionAsync(response, _maxContextLength, cancellationToken);

        var json = await response.Content.ReadAsStringAsync(cancellationToken);
        return JsonNode.Parse(json) as JsonObject
            ?? throw new LMSupply.Exceptions.InferenceBackendException(response.StatusCode, json);
    }
}

#region Models Response Models

internal sealed class ModelsResponse
{
    public List<ModelsEntry>? Data { get; set; }
}

internal sealed class ModelsEntry
{
    public string? Id { get; set; }

    public ModelArchitecture? Architecture { get; set; }
}

internal sealed class ModelArchitecture
{
    [JsonPropertyName("input_modalities")]
    public List<string>? InputModalities { get; set; }

    [JsonPropertyName("output_modalities")]
    public List<string>? OutputModalities { get; set; }
}

#endregion
