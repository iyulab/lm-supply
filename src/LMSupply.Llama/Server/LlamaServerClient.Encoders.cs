using System.Text;
using System.Text.Json;
using LMSupply.Json;

namespace LMSupply.Llama.Server;

public sealed partial class LlamaServerClient
{
    /// <summary>
    /// Generates embeddings for a single text input.
    /// Requires server started with --embedding flag.
    /// </summary>
    public async Task<float[]> GenerateEmbeddingAsync(
        string input,
        CancellationToken cancellationToken = default)
    {
        var result = await GenerateEmbeddingsBatchAsync([input], cancellationToken);
        return result[0];
    }

    /// <summary>
    /// Generates embeddings for multiple text inputs in batch.
    /// Requires server started with --embedding flag.
    /// </summary>
    private async Task<float[][]> GenerateEmbeddingsBatchCoreAsync(
        IReadOnlyList<string> inputs,
        CancellationToken cancellationToken = default)
    {
        var request = new EmbeddingRequest
        {
            Input = inputs.Count == 1 ? inputs[0] : inputs
        };

        var json = JsonSerializer.Serialize(request, JsonOptions.TypeInfoOf(request));
        using var content = new StringContent(json, Encoding.UTF8, "application/json");

        using var response = await PostWithTimeoutAsync(
            $"{_baseUrl}/v1/embeddings",
            content,
            cancellationToken);

        await EnsureSuccessOrThrowContextExceptionAsync(response, _maxContextLength, cancellationToken);

        var responseJson = await response.Content.ReadAsStringAsync(cancellationToken);
        var embeddingResponse = JsonSerializer.Deserialize(responseJson, JsonOptions.TypeInfo<EmbeddingResponse>());

        if (embeddingResponse?.Data == null || embeddingResponse.Data.Count == 0)
        {
            throw new InvalidOperationException("No embeddings returned from server");
        }

        // Sort by index to ensure correct order
        return embeddingResponse.Data
            .OrderBy(d => d.Index)
            .Select(d => d.Embedding)
            .ToArray();
    }

    /// <summary>
    /// Reranks documents by relevance to a query.
    /// Requires server started with --embedding and --pooling rank flags.
    /// </summary>
    private async Task<IReadOnlyList<RerankResult>> RerankCoreAsync(
        string query,
        IReadOnlyList<string> documents,
        int topN = 10,
        CancellationToken cancellationToken = default)
    {
        var request = new RerankRequest
        {
            Query = query,
            Documents = documents.ToList(),
            TopN = Math.Min(topN, documents.Count)
        };

        var json = JsonSerializer.Serialize(request, JsonOptions.TypeInfoOf(request));
        using var content = new StringContent(json, Encoding.UTF8, "application/json");

        using var response = await PostWithTimeoutAsync(
            $"{_baseUrl}/v1/rerank",
            content,
            cancellationToken);

        await EnsureSuccessOrThrowContextExceptionAsync(response, _maxContextLength, cancellationToken);

        var responseJson = await response.Content.ReadAsStringAsync(cancellationToken);
        var rerankResponse = JsonSerializer.Deserialize(responseJson, JsonOptions.TypeInfo<RerankResponse>());

        return rerankResponse?.Results ?? [];
    }
}
