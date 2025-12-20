using System.Text.Json.Serialization;

namespace LMSupply.Console.Host.Models.OpenAI;

/// <summary>
/// Image generation request (OpenAI compatible)
/// </summary>
public sealed record ImageGenerationRequest
{
    /// <summary>
    /// The text prompt to generate an image from.
    /// </summary>
    public required string Prompt { get; init; }

    /// <summary>
    /// The model to use for image generation.
    /// </summary>
    public string? Model { get; init; }

    /// <summary>
    /// Number of images to generate (currently only 1 is supported).
    /// </summary>
    public int N { get; init; } = 1;

    /// <summary>
    /// Image size in format "WxH" (e.g., "512x512", "768x768").
    /// </summary>
    public string? Size { get; init; }

    /// <summary>
    /// The response format (url or b64_json).
    /// </summary>
    [JsonPropertyName("response_format")]
    public string ResponseFormat { get; init; } = "b64_json";

    /// <summary>
    /// Number of inference steps (2-8 for LCM).
    /// </summary>
    public int? Steps { get; init; }

    /// <summary>
    /// Guidance scale for classifier-free guidance.
    /// </summary>
    [JsonPropertyName("guidance_scale")]
    public float? GuidanceScale { get; init; }

    /// <summary>
    /// Random seed for reproducibility.
    /// </summary>
    public int? Seed { get; init; }

    /// <summary>
    /// Negative prompt for avoiding certain features.
    /// </summary>
    [JsonPropertyName("negative_prompt")]
    public string? NegativePrompt { get; init; }
}

/// <summary>
/// Image generation response (OpenAI compatible)
/// </summary>
public sealed record ImageGenerationResponse
{
    /// <summary>
    /// Unix timestamp of when the request was created.
    /// </summary>
    public long Created { get; init; }

    /// <summary>
    /// List of generated images.
    /// </summary>
    public required IReadOnlyList<GeneratedImageData> Data { get; init; }
}

/// <summary>
/// Generated image data
/// </summary>
public sealed record GeneratedImageData
{
    /// <summary>
    /// Base64-encoded image data (when response_format is b64_json).
    /// </summary>
    [JsonPropertyName("b64_json")]
    public string? B64Json { get; init; }

    /// <summary>
    /// URL of the generated image (when response_format is url).
    /// </summary>
    public string? Url { get; init; }

    /// <summary>
    /// Revised prompt (if applicable).
    /// </summary>
    [JsonPropertyName("revised_prompt")]
    public string? RevisedPrompt { get; init; }
}

/// <summary>
/// Extended image generation response with metadata
/// </summary>
public sealed record ImageGenerationExtendedResponse
{
    /// <summary>
    /// Unique identifier for this generation.
    /// </summary>
    public required string Id { get; init; }

    /// <summary>
    /// Model used for generation.
    /// </summary>
    public required string Model { get; init; }

    /// <summary>
    /// Unix timestamp of when the request was created.
    /// </summary>
    public long Created { get; init; }

    /// <summary>
    /// List of generated images.
    /// </summary>
    public required IReadOnlyList<GeneratedImageExtendedData> Data { get; init; }

    /// <summary>
    /// Generation time in milliseconds.
    /// </summary>
    [JsonPropertyName("generation_time_ms")]
    public long GenerationTimeMs { get; init; }
}

/// <summary>
/// Extended generated image data with metadata
/// </summary>
public sealed record GeneratedImageExtendedData
{
    /// <summary>
    /// Base64-encoded image data.
    /// </summary>
    [JsonPropertyName("b64_json")]
    public string? B64Json { get; init; }

    /// <summary>
    /// Image width in pixels.
    /// </summary>
    public int Width { get; init; }

    /// <summary>
    /// Image height in pixels.
    /// </summary>
    public int Height { get; init; }

    /// <summary>
    /// Seed used for generation.
    /// </summary>
    public int Seed { get; init; }

    /// <summary>
    /// Number of inference steps used.
    /// </summary>
    public int Steps { get; init; }

    /// <summary>
    /// Original prompt used.
    /// </summary>
    public string? Prompt { get; init; }
}
