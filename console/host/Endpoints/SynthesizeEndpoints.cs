using LMSupply.Console.Host.Infrastructure;
using LMSupply.Console.Host.Models.OpenAI;
using LMSupply.Console.Host.Services;

namespace LMSupply.Console.Host.Endpoints;

public static class SynthesizeEndpoints
{
    public static void MapSynthesizeEndpoints(this WebApplication app)
    {
        var group = app.MapGroup("/v1/audio")
            .WithTags("Audio")
            .WithOpenApi();

        // POST /v1/audio/speech - OpenAI compatible
        group.MapPost("/speech", async (SpeechRequest request, ModelManagerService manager, CancellationToken ct) =>
        {
            try
            {
                if (string.IsNullOrWhiteSpace(request.Input))
                {
                    return ApiHelper.Error("'input' field is required");
                }

                var synthesizer = await manager.GetSynthesizerAsync(request.Model, ct);
                var result = await synthesizer.SynthesizeAsync(request.Input, cancellationToken: ct);

                // Return audio file
                var contentType = request.ResponseFormat switch
                {
                    "mp3" => "audio/mpeg",
                    "opus" => "audio/opus",
                    "aac" => "audio/aac",
                    "flac" => "audio/flac",
                    "pcm" => "audio/pcm",
                    _ => "audio/wav"
                };

                // Currently only WAV is supported
                return Results.File(
                    result.ToWavBytes(),
                    contentType: "audio/wav",
                    fileDownloadName: $"speech.wav");
            }
            catch (Exception ex)
            {
                return ApiHelper.InternalError(ex);
            }
        })
        .WithName("CreateSpeech")
        .WithSummary("Generate audio from text (OpenAI compatible)")
        .WithDescription("Generates audio from text input. Compatible with OpenAI's text-to-speech API.")
        .Produces(200, contentType: "audio/wav")
        .Produces<ErrorResponse>(400)
        .Produces<ErrorResponse>(500);
    }
}
