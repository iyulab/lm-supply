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
            ;

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

                await using var scope = await manager.GetTranslatorAsync(request.Model, ct);

                var id = ApiHelper.GenerateId("translate");

                if (inputs.Count == 1)
                {
                    var result = await scope.Model.TranslateAsync(inputs[0], ct);
                    return Results.Ok(new TranslateResponse
                    {
                        Id = id,
                        Model = scope.Model.ModelId,
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
                var results = await scope.Model.TranslateBatchAsync(inputs, ct);
                return Results.Ok(new TranslateResponse
                {
                    Id = id,
                    Model = scope.Model.ModelId,
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

        // POST /v2/translate — DeepL API v2 compatible
        app.MapPost("/v2/translate", async (DeepLTranslateRequest request, ModelManagerService manager, CancellationToken ct) =>
        {
            try
            {
                // Parse text: string or string[]
                var texts = new List<string>();
                if (request.Text.ValueKind == System.Text.Json.JsonValueKind.String)
                {
                    texts.Add(request.Text.GetString()!);
                }
                else if (request.Text.ValueKind == System.Text.Json.JsonValueKind.Array)
                {
                    foreach (var el in request.Text.EnumerateArray())
                        if (el.ValueKind == System.Text.Json.JsonValueKind.String)
                            texts.Add(el.GetString()!);
                }

                if (texts.Count == 0)
                    return ApiHelper.Error("'text' field is required");

                // Normalize language codes (DeepL uppercase "EN" → "ko", "KO" etc.)
                var targetLang = request.TargetLang.ToLowerInvariant();
                var sourceLang = request.SourceLang?.ToLowerInvariant();

                // The model is chosen by language pair: the one that translates into the target language (out of the source
                // language, when given). A target no model translates into resolves to no alias, so the answer is model_not_found.
                var candidates = LMSupply.Translator.LocalTranslator.GetAllModels()
                    .Where(m => string.Equals(m.TargetLanguage, targetLang, StringComparison.OrdinalIgnoreCase)
                        && (sourceLang is null || string.Equals(m.SourceLanguage, sourceLang, StringComparison.OrdinalIgnoreCase)))
                    .ToList();
                if (candidates.Count > 1)
                {
                    return ApiHelper.Error(
                        $"'source_lang' is required: more than one model translates into '{targetLang}' ({string.Join(", ", candidates.Select(m => m.SourceLanguage))})");
                }

                var modelAlias = candidates.Count == 1 ? candidates[0].AliasName : $"default-{targetLang}";

                await using var scope = await manager.GetTranslatorAsync(modelAlias, ct);

                var translations = new List<DeepLTranslation>();
                foreach (var text in texts)
                {
                    var result = await scope.Model.TranslateAsync(text, ct);
                    translations.Add(new DeepLTranslation
                    {
                        DetectedSourceLanguage = (result.SourceLanguage ?? sourceLang ?? "auto").ToUpperInvariant(),
                        Text = result.TranslatedText
                    });
                }

                return Results.Ok(new DeepLTranslateResponse { Translations = translations });
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
        .WithName("DeepLTranslate")
        .WithSummary("Translate text (DeepL API v2 compatible)")
        .WithDescription("DeepL-compatible translation. Uses ISO 639-1 language codes (case-insensitive).")
        .WithTags("Translate")
        .Produces<DeepLTranslateResponse>()
        .Produces<ErrorResponse>(400)
        .Produces<ErrorResponse>(404);

        // GET /v1/translate/languages - List available translation directions
        group.MapGet("/translate/languages", () =>
        {
            var models = LMSupply.Translator.LocalTranslator.GetAllModels();
            return Results.Ok(new
            {
                languages = models.Select(m => new
                {
                    id = m.Id,
                    alias = m.AliasName,
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
