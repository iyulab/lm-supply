using System.Text.Json;
using System.Text.Json.Serialization;

namespace LMSupply.Console.Host.Models.OpenAI;

/// <summary>DeepL API v2 request format</summary>
public sealed record DeepLTranslateRequest
{
    /// <summary>Text to translate — string or string array</summary>
    public required System.Text.Json.JsonElement Text { get; init; }

    [System.Text.Json.Serialization.JsonPropertyName("source_lang")]
    public string? SourceLang { get; init; }

    [System.Text.Json.Serialization.JsonPropertyName("target_lang")]
    public required string TargetLang { get; init; }

    public string? Formality { get; init; }

    [System.Text.Json.Serialization.JsonPropertyName("split_sentences")]
    public string? SplitSentences { get; init; }
}

/// <summary>DeepL API v2 response format</summary>
public sealed record DeepLTranslateResponse
{
    public required IReadOnlyList<DeepLTranslation> Translations { get; init; }
}

public sealed record DeepLTranslation
{
    [System.Text.Json.Serialization.JsonPropertyName("detected_source_language")]
    public required string DetectedSourceLanguage { get; init; }
    public required string Text { get; init; }
}

/// <summary>
/// Translation request
/// </summary>
public sealed record TranslateRequest
{
    /// <summary>
    /// Model ID (e.g., "ko-en", "ja-en", or HuggingFace repo ID)
    /// </summary>
    public string Model { get; init; } = "default";

    /// <summary>
    /// Text(s) to translate. Can be string or array of strings.
    /// </summary>
    public required JsonElement Input { get; init; }

    /// <summary>
    /// Source language code (auto-detected if not specified)
    /// </summary>
    [JsonPropertyName("source_language")]
    public string? SourceLanguage { get; init; }

    /// <summary>
    /// Target language code
    /// </summary>
    [JsonPropertyName("target_language")]
    public string? TargetLanguage { get; init; }
}

/// <summary>
/// Translation response
/// </summary>
public sealed record TranslateResponse
{
    public required string Id { get; init; }
    public required string Model { get; init; }
    public required IReadOnlyList<TranslationResult> Translations { get; init; }
}

/// <summary>
/// Individual translation result
/// </summary>
public sealed record TranslationResult
{
    public int Index { get; init; }
    [JsonPropertyName("source_text")]
    public required string SourceText { get; init; }
    [JsonPropertyName("translated_text")]
    public required string TranslatedText { get; init; }
    [JsonPropertyName("source_language")]
    public string? SourceLanguage { get; init; }
    [JsonPropertyName("target_language")]
    public string? TargetLanguage { get; init; }
}
