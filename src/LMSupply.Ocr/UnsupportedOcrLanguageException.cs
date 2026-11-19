namespace LMSupply.Ocr;

/// <summary>
/// The language has no recognition model. A recognizer for another script still returns text, so a language
/// without a model is refused rather than quietly read with the English recognizer; pass <c>"en"</c> to choose
/// that recognizer on purpose.
/// </summary>
public sealed class UnsupportedOcrLanguageException : ArgumentException
{
    /// <summary>
    /// Initializes the exception for a language code that no recognition model covers.
    /// </summary>
    /// <param name="languageCode">The code that was asked for, as the caller wrote it.</param>
    /// <param name="supportedLanguages">The codes that do have a model.</param>
    public UnsupportedOcrLanguageException(string languageCode, IReadOnlyList<string> supportedLanguages)
        : base(BuildMessage(languageCode, supportedLanguages))
    {
        LanguageCode = languageCode;
        SupportedLanguages = supportedLanguages;
    }

    /// <summary>The code that was asked for.</summary>
    public string LanguageCode { get; }

    /// <summary>The language codes that have a recognition model.</summary>
    public IReadOnlyList<string> SupportedLanguages { get; }

    private static string BuildMessage(string languageCode, IReadOnlyList<string> supportedLanguages)
        => $"No OCR recognition model covers the language '{languageCode}'. Supported language codes: " +
           $"{string.Join(", ", supportedLanguages)}. To read it with the English recognizer anyway, pass \"en\".";
}
