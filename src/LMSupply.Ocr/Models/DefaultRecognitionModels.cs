namespace LMSupply.Ocr.Models;

/// <summary>
/// Provides definitions for built-in OCR text recognition models.
/// A primary alias ends in the PP-OCR generation of the model behind it; all of them are PP-OCRv5.
/// English, Korean, Chinese/Japanese, Greek and Thai come from the conversion repository named by <c>DefaultRepoId</c> with a
/// <c>dict.txt</c>, each dictionary byte-identical to the PaddleOCR dictionary of that generation. Arabic, Cyrillic, Devanagari,
/// Latin, Tamil and Telugu come from PaddleOCR's own PP-OCRv5 repositories, whose dictionary is the <c>character_dict</c> list of
/// <c>inference.yml</c>: for Arabic, Devanagari, Tamil and Telugu the conversion repository serves only PP-OCRv3 models, and its
/// Latin and East Slavic ones carry the first PP-OCRv5 dictionaries (Latin 502 entries, now 836; East Slavic lacks the letters
/// Serbian and Macedonian add, which the Cyrillic model has).
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
    public static RecognitionModelInfo CrnnEnV5 { get; } = new(
        RepoId: DefaultRepoId,
        AliasName: "crnn-en-v5",
        DisplayName: "PP-OCRv5 English Recognition",
        ModelFile: "rec.onnx",
        DictFile: "dict.txt",
        LanguageCodes: ["en"])
    {
        Subfolder = "languages/english",
        License = "Apache-2.0"
    };

    /// <summary>
    /// PaddleOCR Korean Recognition (CRNN).
    /// </summary>
    public static RecognitionModelInfo CrnnKoreanV5 { get; } = new(
        RepoId: DefaultRepoId,
        AliasName: "crnn-korean-v5",
        DisplayName: "PP-OCRv5 Korean Recognition",
        ModelFile: "rec.onnx",
        DictFile: "dict.txt",
        LanguageCodes: ["ko"])
    {
        Subfolder = "languages/korean",
        License = "Apache-2.0"
    };

    /// <summary>
    /// PaddleOCR Chinese/Japanese Recognition (CRNN). One model reads both: its dictionary carries
    /// hiragana and katakana alongside the CJK ideographs, and the upstream model card lists Japanese.
    /// </summary>
    public static RecognitionModelInfo CrnnChineseV5 { get; } = new(
        RepoId: DefaultRepoId,
        AliasName: "crnn-chinese-v5",
        DisplayName: "PP-OCRv5 Chinese/Japanese Recognition",
        ModelFile: "rec.onnx",
        DictFile: "dict.txt",
        LanguageCodes: ["zh", "zh-cn", "zh-tw", "ja"])
    {
        Subfolder = "languages/chinese",
        License = "Apache-2.0"
    };

    /// <summary>
    /// PaddleOCR Latin Recognition (CRNN).
    /// Covers: la, es, fr, de, it, pt, nl, pl, ro, cs, sv, da, no, fi, lv, cy, af and more (see the codes below).
    /// </summary>
    public static RecognitionModelInfo CrnnLatinV5 { get; } = new(
        RepoId: "PaddlePaddle/latin_PP-OCRv5_mobile_rec_onnx",
        AliasName: "crnn-latin-v5",
        DisplayName: "PP-OCRv5 Latin Recognition",
        ModelFile: "inference.onnx",
        DictFile: "inference.yml",
        // Every code below is backed by the model's dictionary: it holds each of that language's
        // letters beyond ASCII (Turkish ş ğ ı, Hungarian ő ű, Icelandic þ ð, Latvian ā ē ģ ķ ļ ņ ...). Uzbek and
        // Vietnamese are left out - the dictionary lacks the Latin ʻ of Uzbek and the tone-marked vowels of Vietnamese.
        // nb/nn/fil are the codes an OS locale reports for Norwegian and Filipino.
        LanguageCodes: ["la", "es", "fr", "de", "it", "pt", "nl", "pl", "ro", "cs", "sk", "sv", "da", "no", "nb", "nn",
            "fi", "is", "et", "lt", "lv", "hu", "hr", "bs", "sl", "sq", "ga", "tr", "id", "ms", "sw", "tl", "fil", "cy", "af"])
    {
        License = "Apache-2.0"
    };

    /// <summary>
    /// PaddleOCR Arabic Recognition (CRNN).
    /// Covers: ar, ur, fa.
    /// </summary>
    public static RecognitionModelInfo CrnnArabicV5 { get; } = new(
        RepoId: "PaddlePaddle/arabic_PP-OCRv5_mobile_rec_onnx",
        AliasName: "crnn-arabic-v5",
        DisplayName: "PP-OCRv5 Arabic Recognition",
        ModelFile: "inference.onnx",
        DictFile: "inference.yml",
        LanguageCodes: ["ar", "ur", "fa"])
    {
        License = "Apache-2.0"
    };

    /// <summary>
    /// PaddleOCR Cyrillic/Slavic Recognition (CRNN).
    /// Covers: ru, uk, bg, be, sr, mk.
    /// </summary>
    public static RecognitionModelInfo CrnnCyrillicV5 { get; } = new(
        RepoId: "PaddlePaddle/cyrillic_PP-OCRv5_mobile_rec_onnx",
        AliasName: "crnn-cyrillic-v5",
        DisplayName: "PP-OCRv5 Cyrillic/Slavic Recognition",
        ModelFile: "inference.onnx",
        DictFile: "inference.yml",
        LanguageCodes: ["ru", "uk", "bg", "be", "sr", "mk"])
    {
        License = "Apache-2.0"
    };

    /// <summary>
    /// PaddleOCR Devanagari Recognition (CRNN).
    /// Covers: hi, mr, ne, sa.
    /// </summary>
    public static RecognitionModelInfo CrnnDevanagariV5 { get; } = new(
        RepoId: "PaddlePaddle/devanagari_PP-OCRv5_mobile_rec_onnx",
        AliasName: "crnn-devanagari-v5",
        DisplayName: "PP-OCRv5 Devanagari Recognition",
        ModelFile: "inference.onnx",
        DictFile: "inference.yml",
        LanguageCodes: ["hi", "mr", "ne", "sa"])
    {
        License = "Apache-2.0"
    };

    /// <summary>
    /// PaddleOCR Greek Recognition (CRNN).
    /// </summary>
    public static RecognitionModelInfo CrnnGreekV5 { get; } = new(
        RepoId: DefaultRepoId,
        AliasName: "crnn-greek-v5",
        DisplayName: "PP-OCRv5 Greek Recognition",
        ModelFile: "rec.onnx",
        DictFile: "dict.txt",
        LanguageCodes: ["el"])
    {
        Subfolder = "languages/greek",
        License = "Apache-2.0"
    };

    /// <summary>
    /// PaddleOCR Thai Recognition (CRNN).
    /// </summary>
    public static RecognitionModelInfo CrnnThaiV5 { get; } = new(
        RepoId: DefaultRepoId,
        AliasName: "crnn-thai-v5",
        DisplayName: "PP-OCRv5 Thai Recognition",
        ModelFile: "rec.onnx",
        DictFile: "dict.txt",
        LanguageCodes: ["th"])
    {
        Subfolder = "languages/thai",
        License = "Apache-2.0"
    };

    /// <summary>
    /// PaddleOCR Tamil Recognition (CRNN).
    /// </summary>
    public static RecognitionModelInfo CrnnTamilV5 { get; } = new(
        RepoId: "PaddlePaddle/ta_PP-OCRv5_mobile_rec_onnx",
        AliasName: "crnn-tamil-v5",
        DisplayName: "PP-OCRv5 Tamil Recognition",
        ModelFile: "inference.onnx",
        DictFile: "inference.yml",
        LanguageCodes: ["ta"])
    {
        License = "Apache-2.0"
    };

    /// <summary>
    /// PaddleOCR Telugu Recognition (CRNN).
    /// </summary>
    public static RecognitionModelInfo CrnnTeluguV5 { get; } = new(
        RepoId: "PaddlePaddle/te_PP-OCRv5_mobile_rec_onnx",
        AliasName: "crnn-telugu-v5",
        DisplayName: "PP-OCRv5 Telugu Recognition",
        ModelFile: "inference.onnx",
        DictFile: "inference.yml",
        LanguageCodes: ["te"])
    {
        License = "Apache-2.0"
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
            CrnnEnV5,
            CrnnKoreanV5,
            CrnnChineseV5,
            CrnnLatinV5,
            CrnnArabicV5,
            CrnnCyrillicV5,
            CrnnDevanagariV5,
            CrnnGreekV5,
            CrnnThaiV5,
            CrnnTamilV5,
            CrnnTeluguV5,
        ];

        var result = new List<RecognitionModelInfo>();

        // Register primary aliases first (crnn-*-v5) so that the base class
        // deduplication picks these as the canonical entries in GetAvailableModels().
        foreach (var model in primaryModels)
        {
            result.Add(model);
        }

        // "default" alias points to English
        result.Add(CrnnEnV5 with { AliasName = "default" });

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
