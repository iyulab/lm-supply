using LMSupply.Console.Host.Infrastructure;
using LMSupply.Console.Host.Models.OpenAI;
using LMSupply.Console.Host.Services;

namespace LMSupply.Console.Host.Endpoints;

public static class EmbedEndpoints
{
    public static void MapEmbedEndpoints(this WebApplication app)
    {
        var group = app.MapGroup("/v1")
            .WithTags("Embeddings");

        // POST /v1/embeddings - OpenAI compatible
        group.MapPost("/embeddings", async (EmbeddingRequest request, ModelManagerService manager, CancellationToken ct) =>
        {
            try
            {
                var inputs = ApiHelper.ParseInput(request.Input);
                if (inputs.Count == 0)
                {
                    return ApiHelper.Error("'input' field is required");
                }

                if (inputs.Any(string.IsNullOrEmpty))
                {
                    return ApiHelper.Error("'input' must not contain an empty string");
                }

                await using var scope = await manager.GetEmbedderAsync(request.Model, ct);
                var embeddings = await scope.Model.EmbedAsync(inputs, ct);

                var isBase64 = string.Equals(request.EncodingFormat, "base64", StringComparison.OrdinalIgnoreCase);
                var dims = request.Dimensions;

                var data = embeddings.Select((vec, i) =>
                {
                    // Apply optional dimensions truncation
                    var finalVec = dims.HasValue && dims.Value > 0 && dims.Value < vec.Length
                        ? vec[..dims.Value]
                        : vec;

                    object embeddingValue;
                    if (isBase64)
                    {
                        // Encode raw float bytes as base64 (same as OpenAI: little-endian IEEE 754)
                        var bytes = System.Runtime.InteropServices.MemoryMarshal.Cast<float, byte>(finalVec).ToArray();
                        embeddingValue = Convert.ToBase64String(bytes);
                    }
                    else
                    {
                        embeddingValue = finalVec;
                    }

                    return new EmbeddingData { Index = i, Embedding = embeddingValue };
                }).ToList();

                // Token count: rough estimate (no CountTokens on IEmbeddingModel)
                var totalTokens = inputs.Sum(t => t.Length / 4);

                return Results.Ok(new EmbeddingResponse
                {
                    Data = data,
                    Model = scope.Model.ModelId,
                    Usage = new EmbeddingUsage
                    {
                        PromptTokens = totalTokens,
                        TotalTokens = totalTokens
                    }
                });
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
        .WithName("CreateEmbedding")
        .WithSummary("Create embeddings for text (OpenAI compatible)")
        .WithDescription("Creates embeddings for the given text input. Compatible with OpenAI's embedding API.")
        .Produces<EmbeddingResponse>()
        .Produces<ErrorResponse>(400)
        .Produces<ErrorResponse>(404)
        .Produces<ErrorResponse>(500);
    }
}
