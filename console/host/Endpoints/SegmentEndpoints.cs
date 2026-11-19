using LMSupply.Console.Host.Infrastructure;
using LMSupply.Console.Host.Models.OpenAI;
using LMSupply.Console.Host.Services;

namespace LMSupply.Console.Host.Endpoints;

public static class SegmentEndpoints
{
    public static void MapSegmentEndpoints(this WebApplication app)
    {
        var group = app.MapGroup("/v1/images")
            .WithTags("Vision")
            ;

        // POST /v1/images/segment - Image segmentation
        group.MapPost("/segment", async (HttpRequest request, ModelManagerService manager, CancellationToken ct) =>
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
                var maskFormat = form["mask_format"].FirstOrDefault() ?? "none";

                await using var scope = await manager.GetSegmenterAsync(model, ct);

                using var stream = file.OpenReadStream();
                using var memoryStream = new MemoryStream();
                await stream.CopyToAsync(memoryStream, ct);

                var result = await ApiHelper.ForUploadedImage(scope.Model.SegmentAsync(memoryStream.ToArray(), ct));

                // Get top segments using SDK method
                var topSegments = result.GetTopSegments(10, scope.Model.ClassLabels);
                var segments = topSegments.Select(s =>
                {
                    SegmentMask? mask = null;
                    if (maskFormat != "none")
                    {
                        var counts = maskFormat switch
                        {
                            "rle" => Infrastructure.Vision.RleEncoder.Encode(result.ClassMap, s.ClassId),
                            "raw" => Convert.ToBase64String(result.ClassMap.Select(v => (byte)Math.Clamp(v, 0, 255)).ToArray()),
                            _ => null
                        };
                        if (counts != null)
                        {
                            mask = new SegmentMask
                            {
                                Format = maskFormat,
                                Size = [result.Height, result.Width],
                                Counts = counts
                            };
                        }
                    }

                    return new Segment
                    {
                        Id = s.ClassId,
                        Label = s.Label,
                        Score = s.CoverageRatio,
                        Mask = mask
                    };
                }).ToList();

                return Results.Ok(new SegmentationResponse
                {
                    Id = ApiHelper.GenerateId("seg"),
                    Model = scope.Model.ModelId,
                    Segments = segments
                });
            }
            catch (Exception ex)
            {
                return ApiHelper.InternalError(ex);
            }
        })
        .DisableAntiforgery()
        .WithName("SegmentImage")
        .WithSummary("Segment an image into semantic regions")
        .WithDescription("Performs semantic segmentation on an image, classifying each pixel into predefined categories.")
        .Accepts<IFormFile>("multipart/form-data")
        .Produces<SegmentationResponse>()
        .Produces<ErrorResponse>(400)
        .Produces<ErrorResponse>(404)
        .Produces<ErrorResponse>(500);

        // GET /v1/images/segment/labels - List ADE20K class labels
        group.MapGet("/segment/labels", () =>
        {
            var labels = LMSupply.Segmenter.LocalSegmenter.Ade20kClassLabels;
            return Results.Ok(new
            {
                labels = labels.Select((l, i) => new { id = i, name = l })
            });
        })
        .WithName("ListSegmentLabels")
        .WithSummary("List available segmentation labels (ADE20K)")
        .WithDescription("Returns the ADE20K dataset class labels used for semantic segmentation.");
    }
}
