namespace LMSupply.Translator;

/// <summary>
/// Represents the result of a translation operation.
/// </summary>
public sealed class TranslationResult
{
    /// <summary>
    /// Gets the original source text.
    /// </summary>
    public required string SourceText { get; init; }

    /// <summary>
    /// Gets the translated text.
    /// </summary>
    public required string TranslatedText { get; init; }

    /// <summary>
    /// Gets the source language code.
    /// </summary>
    public required string SourceLanguage { get; init; }

    /// <summary>
    /// Gets the target language code.
    /// </summary>
    public required string TargetLanguage { get; init; }

    /// <summary>
    /// Gets the confidence score of the translation (0-1).
    /// </summary>
    public float? Confidence { get; init; }

    /// <summary>
    /// Gets the inference time in milliseconds.
    /// </summary>
    public double? InferenceTimeMs { get; init; }
}
