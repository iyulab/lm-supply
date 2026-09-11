using System.Diagnostics;

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
        return DefaultRecognitionModels.CrnnEnV3 with { AliasName = "auto" };
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
    /// Always returns the model with its primary alias (e.g., "crnn-korean-v3" rather than "ko").
    /// Falls back to English if the language is not found, and says so through a trace warning -
    /// the English recognizer cannot read another script, so its output would otherwise pass for a
    /// result. <see cref="GetSupportedLanguages"/> lists the codes that resolve without falling back.
    /// Supports region codes (e.g., "en-US" resolves via "en").
    /// </summary>
    /// <param name="languageCode">ISO language code (e.g., "en", "ko", "zh-cn").</param>
    /// <returns>The recognition model for the specified language.</returns>
    public RecognitionModelInfo ResolveForLanguage(string languageCode)
    {
        // Try direct lookup (handles "en", "ko", "zh", "zh-cn", "zh-tw", etc.)
        if (TryResolve(languageCode, out var model) && model is not null)
            return FindPrimaryModel(model);

        // Try just the language part (e.g., "en" from "en-US")
        var langPart = languageCode.Split('-')[0];
        if (TryResolve(langPart, out model) && model is not null)
            return FindPrimaryModel(model);

        // Fall back to English - visibly. A recognizer that cannot read the script still returns text.
        Trace.TraceWarning(
            $"[OcrRecognitionModelRegistry] No recognition model for language '{languageCode}'; using the English " +
            "recognizer instead. Text in another script will not be read correctly. " +
            "GetSupportedLanguages() lists the codes that have a model.");
        return Resolve("crnn-en-v3");
    }

    /// <summary>
    /// Finds the primary model entry (with the crnn-*-v3 alias) for a model
    /// that may have been resolved via a language code alias.
    /// </summary>
    private RecognitionModelInfo FindPrimaryModel(RecognitionModelInfo model)
    {
        // If the alias is already a primary alias (starts with "crnn-"), return as-is
        if (model.AliasName.StartsWith("crnn-", StringComparison.OrdinalIgnoreCase))
            return model;

        // Look up the primary model by matching on the same RepoId + Subfolder
        var primary = GetAvailableModels()
            .FirstOrDefault(m =>
                m.AliasName.StartsWith("crnn-", StringComparison.OrdinalIgnoreCase) &&
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
