using LMSupply.Console.Host.Infrastructure;
using LMSupply.Console.Host.Models.OpenAI;
using LMSupply.Console.Host.Services;
using TranslateRequest = LMSupply.Console.Host.Models.OpenAI.TranslateRequest;

namespace LMSupply.Console.Host.Endpoints;

public static class TranslateEndpoints
{
    public static void MapTranslateEndpoints(this WebApplication app)
    {
        var group = app.MapGroup("/v1")
            .WithTags("Translate")
            .WithOpenApi();

        // POST /v1/translate
        group.MapPost("/translate", async (TranslateRequest request, ModelManagerService manager, CancellationToken ct) =>
        {
            try
            {
                var inputs = ApiHelper.ParseInput(request.Input);

                if (inputs.Count == 0)
                {
                    return ApiHelper.Error("'input' field is required");
                }

                var translator = await manager.GetTranslatorAsync(request.Model, ct);

                var id = ApiHelper.GenerateId("translate");

                if (inputs.Count == 1)
                {
                    var result = await translator.TranslateAsync(inputs[0], ct);
                    return Results.Ok(new TranslateResponse
                    {
                        Id = id,
                        Model = translator.ModelId,
                        Translations =
                        [
                            new TranslationResult
                            {
                                Index = 0,
                                SourceText = result.SourceText,
                                TranslatedText = result.TranslatedText,
                                SourceLanguage = result.SourceLanguage,
                                TargetLanguage = result.TargetLanguage
                            }
                        ]
                    });
                }

                // Batch translation
                var results = await translator.TranslateBatchAsync(inputs, ct);
                return Results.Ok(new TranslateResponse
                {
                    Id = id,
                    Model = translator.ModelId,
                    Translations = results.Select((r, i) => new TranslationResult
                    {
                        Index = i,
                        SourceText = r.SourceText,
                        TranslatedText = r.TranslatedText,
                        SourceLanguage = r.SourceLanguage,
                        TargetLanguage = r.TargetLanguage
                    }).ToList()
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
        .WithName("Translate")
        .WithSummary("Translate text between languages")
        .WithDescription("Translates text from one language to another using neural machine translation.")
        .Produces<TranslateResponse>()
        .Produces<ErrorResponse>(400)
        .Produces<ErrorResponse>(500);

        // GET /v1/translate/languages - List available translation directions
        group.MapGet("/translate/languages", () =>
        {
            var models = LMSupply.Translator.LocalTranslator.GetAllModels();
            return Results.Ok(new
            {
                languages = models.Select(m => new
                {
                    id = m.Id,
                    alias = m.Alias,
                    source = m.SourceLanguage,
                    target = m.TargetLanguage
                })
            });
        })
        .WithName("ListTranslateLanguages")
        .WithSummary("List available translation directions")
        .WithDescription("Returns all supported translation language pairs and their aliases.");
    }
}
