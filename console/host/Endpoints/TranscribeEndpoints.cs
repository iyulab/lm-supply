using LMSupply.Console.Host.Infrastructure;
using LMSupply.Console.Host.Models.OpenAI;
using LMSupply.Console.Host.Services;

namespace LMSupply.Console.Host.Endpoints;

public static class TranscribeEndpoints
{
    public static void MapTranscribeEndpoints(this WebApplication app)
    {
        var group = app.MapGroup("/v1/audio")
            .WithTags("Audio")
            .WithOpenApi();

        // POST /v1/audio/transcriptions - OpenAI compatible
        group.MapPost("/transcriptions", async (HttpRequest request, ModelManagerService manager, CancellationToken ct) =>
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
                    return ApiHelper.Error("Audio file is required in 'file' field");
                }

                var model = form["model"].FirstOrDefault() ?? "default";
                var language = form["language"].FirstOrDefault();
                var responseFormat = form["response_format"].FirstOrDefault() ?? "json";

                var transcriber = await manager.GetTranscriberAsync(model, ct);

                using var stream = file.OpenReadStream();
                using var memoryStream = new MemoryStream();
                await stream.CopyToAsync(memoryStream, ct);

                var result = await transcriber.TranscribeAsync(memoryStream.ToArray(), cancellationToken: ct);

                // Simple JSON format (default)
                if (responseFormat == "json" || responseFormat == "text")
                {
                    if (responseFormat == "text")
                    {
                        return Results.Text(result.Text);
                    }

                    return Results.Ok(new TranscriptionResponse
                    {
                        Text = result.Text
                    });
                }

                // Verbose JSON format
                return Results.Ok(new VerboseTranscriptionResponse
                {
                    Task = "transcribe",
                    Language = result.Language ?? "unknown",
                    Duration = (float)(result.DurationSeconds ?? 0),
                    Text = result.Text,
                    Segments = result.Segments?.Select((s, i) => new TranscriptionSegment
                    {
                        Id = i,
                        Start = (float)s.Start,
                        End = (float)s.End,
                        Text = s.Text
                    }).ToList()
                });
            }
            catch (Exception ex)
            {
                return ApiHelper.InternalError(ex);
            }
        })
        .DisableAntiforgery()
        .WithName("CreateTranscription")
        .WithSummary("Transcribe audio to text (OpenAI compatible)")
        .WithDescription("Transcribes audio into text. Compatible with OpenAI's audio transcription API.")
        .Accepts<IFormFile>("multipart/form-data")
        .Produces<TranscriptionResponse>()
        .Produces<VerboseTranscriptionResponse>()
        .Produces<ErrorResponse>(400)
        .Produces<ErrorResponse>(500);
    }
}
