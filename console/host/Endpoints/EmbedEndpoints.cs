using LMSupply.Console.Host.Infrastructure;
using LMSupply.Console.Host.Models.OpenAI;
using LMSupply.Console.Host.Services;

namespace LMSupply.Console.Host.Endpoints;

public static class EmbedEndpoints
{
    public static void MapEmbedEndpoints(this WebApplication app)
    {
        var group = app.MapGroup("/v1")
            .WithTags("Embeddings")
            .WithOpenApi();

        // POST /v1/embeddings - OpenAI compatible
        group.MapPost("/embeddings", async (EmbeddingRequest request, ModelManagerService manager, CancellationToken ct) =>
        {
            try
            {
                var embedder = await manager.GetEmbedderAsync(request.Model, ct);
                var inputs = ApiHelper.ParseInput(request.Input);

                var embeddings = await embedder.EmbedAsync(inputs, ct);

                var data = embeddings.Select((e, i) => new EmbeddingData
                {
                    Index = i,
                    Embedding = e
                }).ToList();

                return Results.Ok(new EmbeddingResponse
                {
                    Data = data,
                    Model = embedder.ModelId,
                    Usage = new EmbeddingUsage
                    {
                        PromptTokens = inputs.Sum(t => t.Length / 4), // rough estimate
                        TotalTokens = inputs.Sum(t => t.Length / 4)
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
        .Produces<ErrorResponse>(500);
    }
}
