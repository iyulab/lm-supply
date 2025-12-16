using System.Diagnostics.CodeAnalysis;

namespace LMSupply.Ocr.Models;

/// <summary>
/// Registry of known OCR detection and recognition models.
/// </summary>
public static class ModelRegistry
{
    private static readonly Dictionary<string, DetectionModelInfo> DetectionModels = new(StringComparer.OrdinalIgnoreCase);
    private static readonly Dictionary<string, RecognitionModelInfo> RecognitionModels = new(StringComparer.OrdinalIgnoreCase);
    private static readonly Dictionary<string, string> LanguageToRecognitionModel = new(StringComparer.OrdinalIgnoreCase);

    private const string DefaultRepoId = "monkt/paddleocr-onnx";

    static ModelRegistry()
    {
        // Register default PaddleOCR v3 detection model
        RegisterDetectionModel(new DetectionModelInfo(
            RepoId: DefaultRepoId,
            Alias: "dbnet-v3",
            DisplayName: "PaddleOCR v3 Detection (DBNet)",
            ModelFile: "en_PP-OCRv3_det_infer.onnx",
            InputWidth: 960,
            InputHeight: 960));

        // Register default detection alias
        RegisterDetectionAlias("default", "dbnet-v3");

        // Register PaddleOCR v3 recognition models by language
        RegisterRecognitionModel(new RecognitionModelInfo(
            RepoId: DefaultRepoId,
            Alias: "crnn-en-v3",
            DisplayName: "PaddleOCR v3 English Recognition",
            ModelFile: "en_PP-OCRv3_rec_infer.onnx",
            DictFile: "en_dict.txt",
            LanguageCodes: ["en"]));

        RegisterRecognitionModel(new RecognitionModelInfo(
            RepoId: DefaultRepoId,
            Alias: "crnn-korean-v3",
            DisplayName: "PaddleOCR v3 Korean Recognition",
            ModelFile: "korean_PP-OCRv3_rec_infer.onnx",
            DictFile: "korean_dict.txt",
            LanguageCodes: ["ko"]));

        RegisterRecognitionModel(new RecognitionModelInfo(
            RepoId: DefaultRepoId,
            Alias: "crnn-chinese-v3",
            DisplayName: "PaddleOCR v3 Chinese Recognition",
            ModelFile: "ch_PP-OCRv3_rec_infer.onnx",
            DictFile: "chinese_cht_dict.txt",
            LanguageCodes: ["zh", "zh-cn", "zh-tw"]));

        RegisterRecognitionModel(new RecognitionModelInfo(
            RepoId: DefaultRepoId,
            Alias: "crnn-japan-v3",
            DisplayName: "PaddleOCR v3 Japanese Recognition",
            ModelFile: "japan_PP-OCRv3_rec_infer.onnx",
            DictFile: "japan_dict.txt",
            LanguageCodes: ["ja"]));

        RegisterRecognitionModel(new RecognitionModelInfo(
            RepoId: DefaultRepoId,
            Alias: "crnn-latin-v3",
            DisplayName: "PaddleOCR v3 Latin Recognition",
            ModelFile: "latin_PP-OCRv3_rec_infer.onnx",
            DictFile: "latin_dict.txt",
            LanguageCodes: ["la", "es", "fr", "de", "it", "pt", "nl", "pl", "ro", "cs", "sv", "da", "no", "fi"]));

        RegisterRecognitionModel(new RecognitionModelInfo(
            RepoId: DefaultRepoId,
            Alias: "crnn-arabic-v3",
            DisplayName: "PaddleOCR v3 Arabic Recognition",
            ModelFile: "arabic_PP-OCRv3_rec_infer.onnx",
            DictFile: "arabic_dict.txt",
            LanguageCodes: ["ar"]));

        RegisterRecognitionModel(new RecognitionModelInfo(
            RepoId: DefaultRepoId,
            Alias: "crnn-cyrillic-v3",
            DisplayName: "PaddleOCR v3 Cyrillic Recognition",
            ModelFile: "cyrillic_PP-OCRv3_rec_infer.onnx",
            DictFile: "cyrillic_dict.txt",
            LanguageCodes: ["ru", "uk", "bg", "be", "sr", "mk"]));

        RegisterRecognitionModel(new RecognitionModelInfo(
            RepoId: DefaultRepoId,
            Alias: "crnn-devanagari-v3",
            DisplayName: "PaddleOCR v3 Devanagari Recognition",
            ModelFile: "devanagari_PP-OCRv3_rec_infer.onnx",
            DictFile: "devanagari_dict.txt",
            LanguageCodes: ["hi", "mr", "ne", "sa"]));

        // Register language to recognition model mappings
        foreach (var model in RecognitionModels.Values.DistinctBy(m => m.Alias))
        {
            foreach (var lang in model.LanguageCodes)
            {
                LanguageToRecognitionModel[lang] = model.Alias;
            }
        }

        // Set default recognition alias to English
        RegisterRecognitionAlias("default", "crnn-en-v3");
    }

    #region Detection Models

    /// <summary>
    /// Registers a detection model with the registry.
    /// </summary>
    public static void RegisterDetectionModel(DetectionModelInfo model)
    {
        ArgumentNullException.ThrowIfNull(model);
        DetectionModels[model.Alias] = model;
        DetectionModels[model.RepoId + "/" + model.ModelFile] = model;
    }

    /// <summary>
    /// Registers an alias for an existing detection model.
    /// </summary>
    public static void RegisterDetectionAlias(string alias, string existingAlias)
    {
        if (!DetectionModels.TryGetValue(existingAlias, out var model))
        {
            throw new ArgumentException($"Detection model '{existingAlias}' not found in registry", nameof(existingAlias));
        }
        DetectionModels[alias] = model;
    }

    /// <summary>
    /// Tries to get detection model info by alias.
    /// </summary>
    public static bool TryGetDetectionModel(string aliasOrId, [NotNullWhen(true)] out DetectionModelInfo? modelInfo)
    {
        return DetectionModels.TryGetValue(aliasOrId, out modelInfo);
    }

    /// <summary>
    /// Gets detection model info by alias.
    /// </summary>
    public static DetectionModelInfo GetDetectionModel(string aliasOrId)
    {
        if (!TryGetDetectionModel(aliasOrId, out var model))
        {
            throw new KeyNotFoundException($"Detection model '{aliasOrId}' not found. Use GetAvailableDetectionModels() to list available models.");
        }
        return model;
    }

    /// <summary>
    /// Gets a list of available detection model aliases.
    /// </summary>
    public static IEnumerable<string> GetAvailableDetectionModels()
    {
        return DetectionModels.Values
            .Select(m => m.Alias)
            .Distinct()
            .Order();
    }

    #endregion

    #region Recognition Models

    /// <summary>
    /// Registers a recognition model with the registry.
    /// </summary>
    public static void RegisterRecognitionModel(RecognitionModelInfo model)
    {
        ArgumentNullException.ThrowIfNull(model);
        RecognitionModels[model.Alias] = model;
        RecognitionModels[model.RepoId + "/" + model.ModelFile] = model;

        // Update language mappings
        foreach (var lang in model.LanguageCodes)
        {
            LanguageToRecognitionModel[lang] = model.Alias;
        }
    }

    /// <summary>
    /// Registers an alias for an existing recognition model.
    /// </summary>
    public static void RegisterRecognitionAlias(string alias, string existingAlias)
    {
        if (!RecognitionModels.TryGetValue(existingAlias, out var model))
        {
            throw new ArgumentException($"Recognition model '{existingAlias}' not found in registry", nameof(existingAlias));
        }
        RecognitionModels[alias] = model;
    }

    /// <summary>
    /// Tries to get recognition model info by alias.
    /// </summary>
    public static bool TryGetRecognitionModel(string aliasOrId, [NotNullWhen(true)] out RecognitionModelInfo? modelInfo)
    {
        return RecognitionModels.TryGetValue(aliasOrId, out modelInfo);
    }

    /// <summary>
    /// Gets recognition model info by alias.
    /// </summary>
    public static RecognitionModelInfo GetRecognitionModel(string aliasOrId)
    {
        if (!TryGetRecognitionModel(aliasOrId, out var model))
        {
            throw new KeyNotFoundException($"Recognition model '{aliasOrId}' not found. Use GetAvailableRecognitionModels() to list available models.");
        }
        return model;
    }

    /// <summary>
    /// Gets recognition model for a specific language code.
    /// Falls back to English if language not found.
    /// </summary>
    public static RecognitionModelInfo GetRecognitionModelForLanguage(string languageCode)
    {
        if (LanguageToRecognitionModel.TryGetValue(languageCode, out var alias))
        {
            return GetRecognitionModel(alias);
        }

        // Try just the language part (e.g., "en" from "en-US")
        var langPart = languageCode.Split('-')[0];
        if (LanguageToRecognitionModel.TryGetValue(langPart, out alias))
        {
            return GetRecognitionModel(alias);
        }

        // Fall back to English
        return GetRecognitionModel("crnn-en-v3");
    }

    /// <summary>
    /// Gets a list of available recognition model aliases.
    /// </summary>
    public static IEnumerable<string> GetAvailableRecognitionModels()
    {
        return RecognitionModels.Values
            .Select(m => m.Alias)
            .Distinct()
            .Order();
    }

    /// <summary>
    /// Gets a list of supported language codes.
    /// </summary>
    public static IEnumerable<string> GetSupportedLanguages()
    {
        return LanguageToRecognitionModel.Keys.Order();
    }

    #endregion

    /// <summary>
    /// Gets all registered detection models.
    /// </summary>
    public static IEnumerable<DetectionModelInfo> GetAllDetectionModels()
    {
        return DetectionModels.Values.DistinctBy(m => m.Alias);
    }

    /// <summary>
    /// Gets all registered recognition models.
    /// </summary>
    public static IEnumerable<RecognitionModelInfo> GetAllRecognitionModels()
    {
        return RecognitionModels.Values.DistinctBy(m => m.Alias);
    }
}
