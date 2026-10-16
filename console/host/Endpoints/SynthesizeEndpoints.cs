using LMSupply.Console.Host.Infrastructure;
using LMSupply.Console.Host.Infrastructure.Audio;
using LMSupply.Console.Host.Models.OpenAI;
using LMSupply.Console.Host.Services;
using LMSupply.Synthesizer;

namespace LMSupply.Console.Host.Endpoints;

public static class SynthesizeEndpoints
{
    private static readonly Dictionary<string, string> VoiceMap = new(StringComparer.OrdinalIgnoreCase)
    {
        ["alloy"]   = "default",
        ["echo"]    = "default",
        ["fable"]   = "default",
        ["onyx"]    = "default",
        ["nova"]    = "default",
        ["shimmer"] = "default"
    };

    public static void MapSynthesizeEndpoints(this WebApplication app)
    {
        var group = app.MapGroup("/v1/audio").WithTags("Audio");

        // POST /v1/audio/speech - OpenAI compatible
        group.MapPost("/speech", async (SpeechRequest request, HttpContext context, ModelManagerService manager, CancellationToken ct) =>
        {
            try
            {
                if (string.IsNullOrWhiteSpace(request.Input))
                    return ApiHelper.Error("'input' field is required");

                // Map OpenAI voice name to model alias; fall back to request.Model or "default"
                var modelId = VoiceMap.TryGetValue(request.Voice ?? "", out var voiceModel)
                    ? voiceModel
                    : (request.Model ?? "default");
                await using var scope = await manager.GetSynthesizerAsync(modelId, ct);

                var synthesizeOptions = new SynthesizeOptions
                {
                    Speed = Math.Clamp(request.Speed, 0.25f, 4.0f)
                };

                SynthesisResult result;
                try
                {
                    result = await scope.Model.SynthesizeAsync(request.Input, synthesizeOptions, ct);
                }
                catch (NotSupportedException ex)
                {
                    // The voice needs a phonemizer the host does not have (a voice not in English). The library's message says what the voice
                    // takes; it is the answer the caller needs, not an internal detail.
                    return ApiHelper.Error(ex.Message, type: "not_supported_error", statusCode: 501);
                }

                var format = request.ResponseFormat ?? "wav";
                var (audioBytes, contentType) = string.Equals(format, "pcm", StringComparison.OrdinalIgnoreCase)
                    ? (result.ToPcm16Bytes(), "audio/pcm")
                    : AudioConverter.Convert(result.ToWavBytes(), format);

                context.Response.ContentType = contentType;
                context.Response.Headers["Content-Disposition"] = $"attachment; filename=\"speech.{format}\"";

                const int chunkSize = 8192;
                for (int offset = 0; offset < audioBytes.Length; offset += chunkSize)
                {
                    var chunk = audioBytes.AsMemory(offset, Math.Min(chunkSize, audioBytes.Length - offset));
                    await context.Response.Body.WriteAsync(chunk, ct);
                    await context.Response.Body.FlushAsync(ct);
                }

                return Results.Empty;
            }
            catch (Exception ex)
            {
                return ApiHelper.InternalError(ex);
            }
        })
        .WithName("CreateSpeech")
        .WithSummary("Generate audio from text (OpenAI compatible)")
        .WithDescription("Supports wav, pcm, mp3 (fallback wav for unsupported formats). Uses chunked transfer encoding. " +
            "Returns 501 for a voice whose text needs a phonemizer the host does not have (the catalog's Korean and Chinese voices).")
        .Produces(200, contentType: "audio/wav")
        .Produces<ErrorResponse>(400)
        .Produces<ErrorResponse>(404)
        .Produces<ErrorResponse>(500)
        .Produces<ErrorResponse>(501);
    }
}
