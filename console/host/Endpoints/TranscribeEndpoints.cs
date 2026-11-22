using LMSupply.Console.Host.Infrastructure;
using LMSupply.Console.Host.Infrastructure.Audio;
using LMSupply.Console.Host.Models.OpenAI;
using LMSupply.Console.Host.Services;
using LMSupply.Transcriber;

namespace LMSupply.Console.Host.Endpoints;

public static class TranscribeEndpoints
{
    public static void MapTranscribeEndpoints(this WebApplication app)
    {
        var group = app.MapGroup("/v1/audio").WithTags("Audio");

        group.MapPost("/transcriptions", (HttpRequest req, ModelManagerService mgr, CancellationToken ct)
            => HandleTranscriptionAsync(req, mgr, translateToEnglish: false, ct))
            .DisableAntiforgery()
            .WithName("CreateTranscription")
            .WithSummary("Transcribe audio to text (OpenAI compatible)")
            .WithDescription("Transcribes audio into text. Compatible with OpenAI's audio transcription API.")
            .Accepts<IFormFile>("multipart/form-data")
            .Produces<TranscriptionResponse>()
            .Produces<VerboseTranscriptionResponse>()
            .Produces<ErrorResponse>(400)
            .Produces<ErrorResponse>(404)
            .Produces<ErrorResponse>(500);

        group.MapPost("/translations", (HttpRequest req, ModelManagerService mgr, CancellationToken ct)
            => HandleTranscriptionAsync(req, mgr, translateToEnglish: true, ct))
            .DisableAntiforgery()
            .WithName("CreateTranslation")
            .WithSummary("Translate audio to English (OpenAI compatible)")
            .WithDescription("Transcribes and translates audio into English. Compatible with OpenAI's audio translation API.")
            .Accepts<IFormFile>("multipart/form-data")
            .Produces<TranscriptionResponse>()
            .Produces<ErrorResponse>(400)
            .Produces<ErrorResponse>(404)
            .Produces<ErrorResponse>(500);
    }

    private static async Task<IResult> HandleTranscriptionAsync(
        HttpRequest request, ModelManagerService manager,
        bool translateToEnglish, CancellationToken ct)
    {
        try
        {
            if (!request.HasFormContentType)
                return ApiHelper.Error("Form data expected with 'file' field");

            var form = await request.ReadFormAsync(ct);
            var file = form.Files.GetFile("file");
            if (file == null || file.Length == 0)
                return ApiHelper.Error("Audio file required in 'file' field");

            var modelId = form["model"].FirstOrDefault() ?? "default";
            var language = translateToEnglish ? "en" : form["language"].FirstOrDefault();
            var responseFormat = form["response_format"].FirstOrDefault() ?? "json";
            var prompt = form["prompt"].FirstOrDefault();
            var temperatureStr = form["temperature"].FirstOrDefault();
            // timestamp_granularities[] — read but word-level is best-effort
            var timestampGranularities = form["timestamp_granularities[]"].ToArray();

            await using var scope = await manager.GetTranscriberAsync(modelId, ct);

            using var ms = new MemoryStream();
            await file.OpenReadStream().CopyToAsync(ms, ct);

            var options = new TranscribeOptions
            {
                Language = language,
                Translate = translateToEnglish,
                InitialPrompt = string.IsNullOrEmpty(prompt) ? null : prompt,
                // Enable word timestamps if "word" granularity requested (best-effort)
                WordTimestamps = Array.IndexOf(timestampGranularities, "word") >= 0
            };

            if (float.TryParse(temperatureStr, System.Globalization.NumberStyles.Float,
                System.Globalization.CultureInfo.InvariantCulture, out var temperature))
            {
                options.Temperature = temperature;
            }

            var result = await ApiHelper.ForUploadedAudio(scope.Model.TranscribeAsync(ms.ToArray(), options, ct));

            var segments = result.Segments?.Select((s, i) => new Models.OpenAI.TranscriptionSegment
            {
                Id = i,
                Start = (float)s.Start,
                End = (float)s.End,
                Text = s.Text
            }).ToList();

            return responseFormat switch
            {
                "text" => Results.Text(result.Text),
                "srt" => Results.Text(TranscriptionFormatter.ToSrt(segments, result.Text), "text/plain"),
                "vtt" => Results.Text(TranscriptionFormatter.ToVtt(segments, result.Text), "text/vtt"),
                "verbose_json" => Results.Ok(new VerboseTranscriptionResponse
                {
                    Task = translateToEnglish ? "translate" : "transcribe",
                    Language = result.Language ?? language ?? "unknown",
                    Duration = (float)(result.DurationSeconds ?? 0),
                    Text = result.Text,
                    Segments = segments
                }),
                _ => Results.Ok(new TranscriptionResponse { Text = result.Text })
            };
        }
        catch (Exception ex)
        {
            return ApiHelper.InternalError(ex);
        }
    }
}
