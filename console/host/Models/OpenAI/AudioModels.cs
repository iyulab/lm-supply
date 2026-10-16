using System.Text.Json.Serialization;

namespace LMSupply.Console.Host.Models.OpenAI;

/// <summary>
/// OpenAI /v1/audio/transcriptions response
/// </summary>
public sealed record TranscriptionResponse
{
    /// <summary>
    /// Transcribed text
    /// </summary>
    public required string Text { get; init; }
}

/// <summary>
/// Verbose transcription response (response_format=verbose_json)
/// </summary>
public sealed record VerboseTranscriptionResponse
{
    public required string Task { get; init; }
    public required string Language { get; init; }
    public float Duration { get; init; }
    public required string Text { get; init; }
    public IReadOnlyList<TranscriptionSegment>? Segments { get; init; }
}

/// <summary>
/// Transcription segment
/// </summary>
public sealed record TranscriptionSegment
{
    public int Id { get; init; }
    public int Seek { get; init; }
    public float Start { get; init; }
    public float End { get; init; }
    public required string Text { get; init; }
    public float Temperature { get; init; }
    [JsonPropertyName("avg_logprob")]
    public float AvgLogprob { get; init; }
    [JsonPropertyName("compression_ratio")]
    public float CompressionRatio { get; init; }
    [JsonPropertyName("no_speech_prob")]
    public float NoSpeechProb { get; init; }
}

/// <summary>
/// OpenAI /v1/audio/speech request
/// </summary>
public sealed record SpeechRequest
{
    /// <summary>
    /// TTS model to use
    /// </summary>
    public string Model { get; init; } = "default";

    /// <summary>
    /// Text to convert to speech
    /// </summary>
    public required string Input { get; init; }

    /// <summary>
    /// Voice to use (alloy, echo, fable, onyx, nova, shimmer). One of these names picks the default
    /// voice; when no voice is sent, <see cref="Model"/> names the synthesizer model.
    /// </summary>
    public string? Voice { get; init; }

    /// <summary>
    /// Response format: mp3, opus, aac, flac, wav, pcm
    /// </summary>
    [JsonPropertyName("response_format")]
    public string ResponseFormat { get; init; } = "wav";

    /// <summary>
    /// Speed of speech (0.25 to 4.0)
    /// </summary>
    public float Speed { get; init; } = 1.0f;
}
