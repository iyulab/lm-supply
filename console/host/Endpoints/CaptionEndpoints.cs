using LMSupply.Console.Host.Infrastructure;
using LMSupply.Console.Host.Models.OpenAI;
using LMSupply.Console.Host.Services;

namespace LMSupply.Console.Host.Endpoints;

public static class CaptionEndpoints
{
    public static void MapCaptionEndpoints(this WebApplication app)
    {
        var group = app.MapGroup("/v1/images")
            .WithTags("Vision")
            .WithOpenApi();

        // POST /v1/images/caption - Image captioning
        group.MapPost("/caption", async (HttpRequest request, ModelManagerService manager, CancellationToken ct) =>
        {
            try
            {
                if (!request.HasFormContentType)
                {
                    return ApiHelper.Error("Form data expected with 'file' field");
                }

                var form = await request.ReadFormAsync(ct);
                var file = form.Files.GetFile("file");

                if (file == null || file.Length == 0)
                {
                    return ApiHelper.Error("Image file is required in 'file' field");
                }

                var model = form["model"].FirstOrDefault() ?? "default";

                var captioner = await manager.GetCaptionerAsync(model, ct);

                using var stream = file.OpenReadStream();
                using var memoryStream = new MemoryStream();
                await stream.CopyToAsync(memoryStream, ct);

                var result = await captioner.CaptionAsync(memoryStream.ToArray(), ct);

                return Results.Ok(new CaptionResponse
                {
                    Id = ApiHelper.GenerateId("caption"),
                    Model = captioner.ModelId,
                    Caption = result.Caption,
                    Confidence = result.Confidence,
                    Alternatives = result.AlternativeCaptions
                });
            }
            catch (Exception ex)
            {
                return ApiHelper.InternalError(ex);
            }
        })
        .DisableAntiforgery()
        .WithName("CaptionImage")
        .WithSummary("Generate a caption for an image")
        .WithDescription("Generates a text description of the image content.")
        .Accepts<IFormFile>("multipart/form-data")
        .Produces<CaptionResponse>()
        .Produces<ErrorResponse>(400)
        .Produces<ErrorResponse>(500);

        // POST /v1/images/vqa - Visual Question Answering
        group.MapPost("/vqa", async (HttpRequest request, ModelManagerService manager, CancellationToken ct) =>
        {
            try
            {
                if (!request.HasFormContentType)
                {
                    return ApiHelper.Error("Form data expected with 'file' and 'question' fields");
                }

                var form = await request.ReadFormAsync(ct);
                var file = form.Files.GetFile("file");

                if (file == null || file.Length == 0)
                {
                    return ApiHelper.Error("Image file is required in 'file' field");
                }

                var question = form["question"].FirstOrDefault();
                if (string.IsNullOrWhiteSpace(question))
                {
                    return ApiHelper.Error("'question' field is required");
                }

                var model = form["model"].FirstOrDefault() ?? "default";

                var captioner = await manager.GetCaptionerAsync(model, ct);

                using var stream = file.OpenReadStream();
                using var memoryStream = new MemoryStream();
                await stream.CopyToAsync(memoryStream, ct);
                memoryStream.Position = 0;

                var result = await captioner.AnswerAsync(memoryStream, question, ct);

                return Results.Ok(new VqaResponse
                {
                    Id = ApiHelper.GenerateId("vqa"),
                    Model = captioner.ModelId,
                    Question = question,
                    Answer = result.Answer,
                    Confidence = result.Confidence
                });
            }
            catch (Exception ex)
            {
                return ApiHelper.InternalError(ex);
            }
        })
        .DisableAntiforgery()
        .WithName("VisualQA")
        .WithSummary("Answer a question about an image")
        .WithDescription("Answers natural language questions about image content using visual question answering.")
        .Accepts<IFormFile>("multipart/form-data")
        .Produces<VqaResponse>()
        .Produces<ErrorResponse>(400)
        .Produces<ErrorResponse>(500);
    }
}
