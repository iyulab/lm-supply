using LMSupply.Console.Host.Infrastructure;
using LMSupply.Console.Host.Models.OpenAI;
using LMSupply.Console.Host.Services;
using LMSupply.Vision;

namespace LMSupply.Console.Host.Endpoints;

public static class OcrEndpoints
{
    public static void MapOcrEndpoints(this WebApplication app)
    {
        var group = app.MapGroup("/v1/images")
            .WithTags("Vision")
            ;

        // POST /v1/images/ocr - OCR text recognition
        group.MapPost("/ocr", async (HttpRequest request, ModelManagerService manager, CancellationToken ct) =>
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

                var language = form["language"].FirstOrDefault() ?? "en";

                await using var scope = await manager.GetOcrAsync(language, ct);

                using var stream = file.OpenReadStream();
                using var memoryStream = new MemoryStream();
                await stream.CopyToAsync(memoryStream, ct);

                var imageBytes = memoryStream.ToArray();
                var result = await scope.Model.RecognizeAsync(imageBytes, ct);

                // Get image dimensions for page metadata
                var imageSize = ImageLoader.Identify(imageBytes);
                int imageWidth = imageSize?.Width ?? 0;
                int imageHeight = imageSize?.Height ?? 0;

                var blocks = result.Regions.Select(r => new Infrastructure.Vision.OcrBlock
                {
                    Text = r.Text,
                    Confidence = r.Confidence,
                    BoundingBox = new Infrastructure.Vision.OcrBoundingBox
                    {
                        X = r.BoundingBox.X,
                        Y = r.BoundingBox.Y,
                        Width = r.BoundingBox.Width,
                        Height = r.BoundingBox.Height
                    },
                    Lines = [new Infrastructure.Vision.OcrLine
                    {
                        Text = r.Text,
                        Confidence = r.Confidence,
                        BoundingBox = new Infrastructure.Vision.OcrBoundingBox
                        {
                            X = r.BoundingBox.X,
                            Y = r.BoundingBox.Y,
                            Width = r.BoundingBox.Width,
                            Height = r.BoundingBox.Height
                        }
                    }]
                }).ToList();

                return Results.Ok(new OcrResponse
                {
                    Id = ApiHelper.GenerateId("ocr"),
                    Model = $"{scope.Model.DetectionModelId}+{scope.Model.RecognitionModelId}",
                    FullText = result.FullText,
                    Pages = [new Infrastructure.Vision.OcrPage { Width = imageWidth, Height = imageHeight, Blocks = blocks }],
                    DetectedLanguages = [new DetectedLanguage { LanguageCode = language, Confidence = 1.0f }]
                });
            }
            catch (Exception ex)
            {
                return ApiHelper.InternalError(ex);
            }
        })
        .DisableAntiforgery()
        .WithName("RecognizeText")
        .WithSummary("Extract text from an image (OCR)")
        .WithDescription("Performs optical character recognition to extract text from images.")
        .Accepts<IFormFile>("multipart/form-data")
        .Produces<OcrResponse>()
        .Produces<ErrorResponse>(400)
        .Produces<ErrorResponse>(404)
        .Produces<ErrorResponse>(500);

        // GET /v1/images/ocr/languages - List supported OCR languages
        group.MapGet("/ocr/languages", () =>
        {
            var languages = LMSupply.Ocr.LocalOcr.GetSupportedLanguages();
            return Results.Ok(new { languages });
        })
        .WithName("ListOcrLanguages")
        .WithSummary("List supported OCR languages")
        .WithDescription("Returns a list of language codes supported for OCR.");
    }
}
