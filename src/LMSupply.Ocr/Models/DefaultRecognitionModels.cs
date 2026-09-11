namespace LMSupply.Ocr.Models;

/// <summary>
/// Provides definitions for built-in OCR text recognition models.
/// Each model is registered under its primary alias, and additional entries
/// are created for each supported language code so that language codes
/// work directly as registry lookups.
/// </summary>
internal static class DefaultRecognitionModels
{
    private const string DefaultRepoId = "monkt/paddleocr-onnx";

    // ===== Primary model definitions =====

    /// <summary>
    /// PaddleOCR English Recognition (CRNN).
    /// </summary>
    public static RecognitionModelInfo CrnnEnV3 { get; } = new(
        RepoId: DefaultRepoId,
        AliasName: "crnn-en-v3",
        DisplayName: "PaddleOCR English Recognition",
        ModelFile: "rec.onnx",
        DictFile: "dict.txt",
        LanguageCodes: ["en"])
    {
        Subfolder = "languages/english"
    };

    /// <summary>
    /// PaddleOCR Korean Recognition (CRNN).
    /// </summary>
    public static RecognitionModelInfo CrnnKoreanV3 { get; } = new(
        RepoId: DefaultRepoId,
        AliasName: "crnn-korean-v3",
        DisplayName: "PaddleOCR Korean Recognition",
        ModelFile: "rec.onnx",
        DictFile: "dict.txt",
        LanguageCodes: ["ko"])
    {
        Subfolder = "languages/korean"
    };

    /// <summary>
    /// PaddleOCR Chinese/Japanese Recognition (CRNN). One model reads both: its dictionary carries
    /// hiragana and katakana alongside the CJK ideographs, and the upstream model card lists Japanese.
    /// </summary>
    public static RecognitionModelInfo CrnnChineseV3 { get; } = new(
        RepoId: DefaultRepoId,
        AliasName: "crnn-chinese-v3",
        DisplayName: "PaddleOCR Chinese/Japanese Recognition",
        ModelFile: "rec.onnx",
        DictFile: "dict.txt",
        LanguageCodes: ["zh", "zh-cn", "zh-tw", "ja"])
    {
        Subfolder = "languages/chinese"
    };

    /// <summary>
    /// PaddleOCR Latin Recognition (CRNN).
    /// Covers: la, es, fr, de, it, pt, nl, pl, ro, cs, sv, da, no, fi.
    /// </summary>
    public static RecognitionModelInfo CrnnLatinV3 { get; } = new(
        RepoId: DefaultRepoId,
        AliasName: "crnn-latin-v3",
        DisplayName: "PaddleOCR Latin Recognition",
        ModelFile: "rec.onnx",
        DictFile: "dict.txt",
        // Every code below is backed by the model's dictionary: it holds each of that language's
        // letters beyond ASCII (Turkish ş ğ ı, Hungarian ő ű, Icelandic þ ð, ...). Latvian, Welsh,
        // Afrikaans, Uzbek and Vietnamese are left out - the dictionary lacks some of their letters.
        // nb/nn/fil are the codes an OS locale reports for Norwegian and Filipino.
        LanguageCodes: ["la", "es", "fr", "de", "it", "pt", "nl", "pl", "ro", "cs", "sk", "sv", "da", "no", "nb", "nn",
            "fi", "is", "et", "lt", "hu", "hr", "bs", "sl", "sq", "ga", "tr", "id", "ms", "sw", "tl", "fil"])
    {
        Subfolder = "languages/latin"
    };

    /// <summary>
    /// PaddleOCR Arabic Recognition (CRNN).
    /// Covers: ar, ur, fa.
    /// </summary>
    public static RecognitionModelInfo CrnnArabicV3 { get; } = new(
        RepoId: DefaultRepoId,
        AliasName: "crnn-arabic-v3",
        DisplayName: "PaddleOCR Arabic Recognition",
        ModelFile: "rec.onnx",
        DictFile: "dict.txt",
        LanguageCodes: ["ar", "ur", "fa"])
    {
        Subfolder = "languages/arabic"
    };

    /// <summary>
    /// PaddleOCR Cyrillic/Slavic Recognition (CRNN).
    /// Covers: ru, uk, bg, be, sr, mk.
    /// </summary>
    public static RecognitionModelInfo CrnnCyrillicV3 { get; } = new(
        RepoId: DefaultRepoId,
        AliasName: "crnn-cyrillic-v3",
        DisplayName: "PaddleOCR Cyrillic/Slavic Recognition",
        ModelFile: "rec.onnx",
        DictFile: "dict.txt",
        LanguageCodes: ["ru", "uk", "bg", "be", "sr", "mk"])
    {
        Subfolder = "languages/eslav"
    };

    /// <summary>
    /// PaddleOCR Devanagari Recognition (CRNN).
    /// Covers: hi, mr, ne, sa.
    /// </summary>
    public static RecognitionModelInfo CrnnDevanagariV3 { get; } = new(
        RepoId: DefaultRepoId,
        AliasName: "crnn-devanagari-v3",
        DisplayName: "PaddleOCR Devanagari Recognition",
        ModelFile: "rec.onnx",
        DictFile: "dict.txt",
        LanguageCodes: ["hi", "mr", "ne", "sa"])
    {
        Subfolder = "languages/hindi"
    };

    /// <summary>
    /// PaddleOCR Greek Recognition (CRNN).
    /// </summary>
    public static RecognitionModelInfo CrnnGreekV3 { get; } = new(
        RepoId: DefaultRepoId,
        AliasName: "crnn-greek-v3",
        DisplayName: "PaddleOCR Greek Recognition",
        ModelFile: "rec.onnx",
        DictFile: "dict.txt",
        LanguageCodes: ["el"])
    {
        Subfolder = "languages/greek"
    };

    /// <summary>
    /// PaddleOCR Thai Recognition (CRNN).
    /// </summary>
    public static RecognitionModelInfo CrnnThaiV3 { get; } = new(
        RepoId: DefaultRepoId,
        AliasName: "crnn-thai-v3",
        DisplayName: "PaddleOCR Thai Recognition",
        ModelFile: "rec.onnx",
        DictFile: "dict.txt",
        LanguageCodes: ["th"])
    {
        Subfolder = "languages/thai"
    };

    /// <summary>
    /// PaddleOCR Tamil Recognition (CRNN).
    /// </summary>
    public static RecognitionModelInfo CrnnTamilV3 { get; } = new(
        RepoId: DefaultRepoId,
        AliasName: "crnn-tamil-v3",
        DisplayName: "PaddleOCR Tamil Recognition",
        ModelFile: "rec.onnx",
        DictFile: "dict.txt",
        LanguageCodes: ["ta"])
    {
        Subfolder = "languages/tamil"
    };

    /// <summary>
    /// PaddleOCR Telugu Recognition (CRNN).
    /// </summary>
    public static RecognitionModelInfo CrnnTeluguV3 { get; } = new(
        RepoId: DefaultRepoId,
        AliasName: "crnn-telugu-v3",
        DisplayName: "PaddleOCR Telugu Recognition",
        ModelFile: "rec.onnx",
        DictFile: "dict.txt",
        LanguageCodes: ["te"])
    {
        Subfolder = "languages/telugu"
    };

    /// <summary>
    /// Gets all built-in recognition models.
    /// Primary aliases come first, then language-code aliases are generated
    /// for every additional language code each model supports.
    /// The "default" alias points to the English model.
    /// </summary>
    public static IReadOnlyList<RecognitionModelInfo> All { get; } = BuildAll();

    private static List<RecognitionModelInfo> BuildAll()
    {
        // All primary models in registration order
        RecognitionModelInfo[] primaryModels =
        [
            CrnnEnV3,
            CrnnKoreanV3,
            CrnnChineseV3,
            CrnnLatinV3,
            CrnnArabicV3,
            CrnnCyrillicV3,
            CrnnDevanagariV3,
            CrnnGreekV3,
            CrnnThaiV3,
            CrnnTamilV3,
            CrnnTeluguV3,
        ];

        var result = new List<RecognitionModelInfo>();

        // Register primary aliases first (crnn-*-v3) so that the base class
        // deduplication picks these as the canonical entries in GetAvailableModels().
        foreach (var model in primaryModels)
        {
            result.Add(model);
        }

        // "default" alias points to English
        result.Add(CrnnEnV3 with { AliasName = "default" });

        // Register language code aliases (en, ko, zh, etc.)
        foreach (var model in primaryModels)
        {
            foreach (var lang in model.LanguageCodes)
            {
                if (!lang.Equals(model.AliasName, StringComparison.OrdinalIgnoreCase))
                {
                    result.Add(model with { AliasName = lang });
                }
            }
        }

        return result;
    }
}
