using LMSupply.Console.Host.Infrastructure;
using LMSupply.Console.Host.Models.OpenAI;
using LMSupply.Console.Host.Services;
using RerankRequest = LMSupply.Console.Host.Models.OpenAI.RerankRequest;

namespace LMSupply.Console.Host.Endpoints;

public static class RerankEndpoints
{
    public static void MapRerankEndpoints(this WebApplication app)
    {
        var group = app.MapGroup("/v1")
            .WithTags("Rerank")
            .WithOpenApi();

        // POST /v1/rerank - Cohere API compatible
        group.MapPost("/rerank", async (RerankRequest request, ModelManagerService manager, CancellationToken ct) =>
        {
            try
            {
                if (string.IsNullOrWhiteSpace(request.Query))
                {
                    return ApiHelper.Error("'query' field is required");
                }

                if (request.Documents == null || request.Documents.Count == 0)
                {
                    return ApiHelper.Error("'documents' field is required and must not be empty");
                }

                var reranker = await manager.GetRerankerAsync(request.Model, ct);
                var results = await reranker.RerankAsync(
                    request.Query,
                    request.Documents,
                    request.TopN,
                    ct);

                var id = ApiHelper.GenerateId("rerank");

                return Results.Ok(new RerankResponse
                {
                    Id = id,
                    Model = reranker.ModelId,
                    Results = results.Select(r => new RerankResult
                    {
                        Index = r.OriginalIndex,
                        RelevanceScore = r.Score,
                        Document = request.ReturnDocuments
                            ? new RerankDocument { Text = r.Document }
                            : null
                    }).ToList()
                });
            }
            catch (Exception ex)
            {
                return ApiHelper.InternalError(ex);
            }
        })
        .WithName("Rerank")
        .WithSummary("Rerank documents by relevance (Cohere API compatible)")
        .WithDescription("Ranks documents based on their relevance to a query. Compatible with Cohere's rerank API.")
        .Produces<RerankResponse>()
        .Produces<ErrorResponse>(400)
        .Produces<ErrorResponse>(500);
    }
}
