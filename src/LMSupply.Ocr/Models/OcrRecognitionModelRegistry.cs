using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;

namespace LMSupply.Ocr.Models;

/// <summary>
/// Model registry for OCR text recognition models.
/// Language codes (en, ko, zh, etc.) are registered as system aliases,
/// so they can be used directly as model identifiers.
/// </summary>
public sealed class OcrRecognitionModelRegistry : ModelRegistryBase<RecognitionModelInfo>
{
    /// <summary>
    /// Gets the default registry instance with built-in recognition models.
    /// </summary>
    public static OcrRecognitionModelRegistry Default { get; } = CreateDefault();

    /// <summary>
    /// Builds the default registry: built-in models plus the user's
    /// ~/.lmsupply/aliases.json "ocr-recognition" section (fail-soft).
    /// </summary>
    internal static OcrRecognitionModelRegistry CreateDefault()
        => AliasConfiguration.ApplyDomain(new OcrRecognitionModelRegistry(DefaultRecognitionModels.All), AliasConfiguration.Domains.OcrRecognition);

    /// <summary>
    /// Initializes a new registry with the specified system models.
    /// </summary>
    /// <param name="systemModels">Models to register as system defaults.</param>
    public OcrRecognitionModelRegistry(IEnumerable<RecognitionModelInfo> systemModels)
        : base(systemModels) { }

    /// <summary>
    /// Returns the English recognition model for the "auto" alias.
    /// </summary>
    protected override RecognitionModelInfo GetAutoModel()
    {
        Trace.TraceInformation("[OcrRecognitionModelRegistry] Auto-selecting English recognition model");
        return DefaultRecognitionModels.CrnnEnV5 with { AliasName = "auto" };
    }

    /// <summary>
    /// Creates a fallback model info for unknown model IDs (HuggingFace repos or local paths).
    /// </summary>
    protected override RecognitionModelInfo CreateFallbackModelInfo(string modelId)
    {
        Trace.TraceInformation($"[OcrRecognitionModelRegistry] Creating fallback model info for: {modelId}");

        return new RecognitionModelInfo(
            RepoId: modelId,
            AliasName: modelId,
            DisplayName: $"Custom recognition model: {modelId}",
            ModelFile: "rec.onnx",
            DictFile: "dict.txt",
            LanguageCodes: ["en"]);
    }

    /// <summary>
    /// Resolves a recognition model for a specific language code.
    /// Always returns the model with its primary alias (e.g., "crnn-korean-v5" rather than "ko").
    /// The code is read as a BCP 47 tag or a POSIX locale name: <c>_</c> and <c>-</c> separate subtags, case is
    /// ignored, a script subtag counts before the region (<c>sr-Latn-RS</c> is Latin letters, <c>sr</c> Cyrillic) and
    /// the regionless language is the last resort (<c>en-US</c> resolves via <c>en</c>) - but only when its recognizer reads the script
    /// the code names (<c>bs-Cyrl</c> is refused: <c>bs</c> is the Latin recognizer's).
    /// A language with no recognition model is refused: another script's recognizer would still return text, so it is
    /// never substituted. Pass <c>"en"</c> to read with the English recognizer on purpose.
    /// <see cref="GetSupportedLanguages"/> lists the codes that have a model.
    /// </summary>
    /// <param name="languageCode">Language code (e.g., "en", "ko", "zh-cn", "ko_KR", "sr-Latn").</param>
    /// <returns>The recognition model for the specified language.</returns>
    /// <exception cref="ArgumentException"><paramref name="languageCode"/> is null or white space.</exception>
    /// <exception cref="UnsupportedOcrLanguageException">No recognition model covers the language.</exception>
    public RecognitionModelInfo ResolveForLanguage(string languageCode)
    {
        if (TryResolveForLanguage(languageCode, out var model))
            return model;

        throw new UnsupportedOcrLanguageException(languageCode, GetSupportedLanguages().ToList());
    }

    /// <summary>
    /// Like <see cref="ResolveForLanguage"/>, but answers <see langword="false"/> instead of throwing when no
    /// recognition model covers the language.
    /// </summary>
    /// <param name="languageCode">Language code, as for <see cref="ResolveForLanguage"/>.</param>
    /// <param name="model">The recognition model with its primary alias, when the result is <see langword="true"/>.</param>
    /// <exception cref="ArgumentException"><paramref name="languageCode"/> is null or white space.</exception>
    public bool TryResolveForLanguage(string languageCode, [NotNullWhen(true)] out RecognitionModelInfo? model)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(languageCode);

        var script = OcrLanguageCode.ScriptOf(languageCode);
        foreach (var candidate in OcrLanguageCode.Candidates(languageCode))
        {
            if (!TryResolve(candidate, out var found) || found is null)
                continue;

            var primary = FindPrimaryModel(found);

            // A code that names its script ("bs-Cyrl") is not read by a language-only entry of the other script ("bs" is the
            // Latin recognizer's): the recognizer would still return text, in letters it has no dictionary for.
            if (script is not null && !OcrLanguageCode.NamesScript(candidate, script)
                && !OcrLanguageCode.ReadsScript(primary.AliasName, script))
                continue;

            model = primary;
            return true;
        }

        model = null;
        return false;
    }

    /// <summary>
    /// Finds the primary model entry (the one with the crnn-* alias) for a model
    /// that may have been resolved via a language code alias.
    /// </summary>
    private RecognitionModelInfo FindPrimaryModel(RecognitionModelInfo model)
    {
        // If the alias is already a primary alias (starts with "crnn-"), return as-is
        if (model.AliasName.StartsWith("crnn-", StringComparison.OrdinalIgnoreCase))
            return model;

        // Look up the primary model by matching on the same RepoId + Subfolder: a model is the files it names, and
        // models that live in a repository of their own have no subfolder, so the subfolder alone cannot tell them apart.
        var primary = GetAvailableModels()
            .FirstOrDefault(m =>
                m.AliasName.StartsWith("crnn-", StringComparison.OrdinalIgnoreCase) &&
                m.RepoId == model.RepoId &&
                m.Subfolder == model.Subfolder);

        return primary ?? model;
    }

    /// <summary>
    /// Gets all supported language codes across all registered recognition models.
    /// </summary>
    public IEnumerable<string> GetSupportedLanguages()
    {
        return GetAvailableModels()
            .SelectMany(m => m.LanguageCodes)
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .Order();
    }
}
