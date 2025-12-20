using LMSupply.Console.Host.Infrastructure;
using LMSupply.Console.Host.Models.OpenAI;
using LMSupply.Console.Host.Services;
using LMSupply.Detector;

namespace LMSupply.Console.Host.Endpoints;

public static class DetectEndpoints
{
    public static void MapDetectEndpoints(this WebApplication app)
    {
        var group = app.MapGroup("/v1/images")
            .WithTags("Vision")
            .WithOpenApi();

        // POST /v1/images/detect - Object detection
        group.MapPost("/detect", async (HttpRequest request, ModelManagerService manager, CancellationToken ct) =>
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
                var thresholdStr = form["threshold"].FirstOrDefault();
                var threshold = float.TryParse(thresholdStr, out var t) ? t : 0.5f;

                var detector = await manager.GetDetectorAsync(model, ct);

                using var stream = file.OpenReadStream();
                using var memoryStream = new MemoryStream();
                await stream.CopyToAsync(memoryStream, ct);

                // Use SDK extension method for threshold filtering
                var results = await detector.DetectAsync(memoryStream.ToArray(), threshold, ct);

                return Results.Ok(new DetectionResponse
                {
                    Id = ApiHelper.GenerateId("detect"),
                    Model = detector.ModelId,
                    Objects = results.Select(r => new DetectedObject
                    {
                        Label = r.Label,
                        Confidence = r.Confidence,
                        BoundingBox = new Models.OpenAI.BoundingBox
                        {
                            X = r.Box.X1,
                            Y = r.Box.Y1,
                            Width = r.Box.Width,
                            Height = r.Box.Height
                        }
                    }).ToList()
                });
            }
            catch (Exception ex)
            {
                return ApiHelper.InternalError(ex);
            }
        })
        .DisableAntiforgery()
        .WithName("DetectObjects")
        .WithSummary("Detect objects in an image")
        .WithDescription("Detects objects in an image and returns their locations and labels using YOLO-based detection.")
        .Accepts<IFormFile>("multipart/form-data")
        .Produces<DetectionResponse>()
        .Produces<ErrorResponse>(400)
        .Produces<ErrorResponse>(500);

        // GET /v1/images/detect/labels - List COCO class labels
        group.MapGet("/detect/labels", () =>
        {
            var labels = LMSupply.Detector.LocalDetector.CocoClassLabels;
            return Results.Ok(new
            {
                labels = labels.Select((l, i) => new { id = i, name = l })
            });
        })
        .WithName("ListDetectionLabels")
        .WithSummary("List available object detection labels")
        .WithDescription("Returns the COCO class labels used for object detection.");
    }
}
