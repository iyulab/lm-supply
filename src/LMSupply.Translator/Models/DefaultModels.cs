namespace LMSupply.Translator.Models;

/// <summary>
/// Default translation model configurations.
/// All models use Apache-2.0 or CC-BY-4.0 compatible licenses for commercial use.
/// Models use onnx-community repos with auto-discovery enabled for ONNX files.
/// </summary>
public static class DefaultModels
{
    /// <summary>
    /// OPUS-MT Korean to English - Default model.
    /// Apache 2.0 license.
    /// </summary>
    public static TranslatorModelInfo OpusMtKoEn { get; } = new()
    {
        Id = "onnx-community/opus-mt-ko-en",
        AliasName = "default",
        DisplayName = "OPUS-MT Ko-En",
        Architecture = "MarianMT",
        SourceLanguage = "ko",
        TargetLanguage = "en",
        ParametersM = 74f,
        SizeBytes = 445_854_852,
        BleuScore = 35.5f,
        MaxLength = 512,
        VocabSize = 65000,
        // UseAutoDiscovery = true (default), encoder/decoder files auto-discovered
        TokenizerFile = "source.spm",
        Description = "Korean to English translation using OPUS-MT.",
        License = "Apache-2.0"
    };

    /// <summary>
    /// OPUS-MT Korean to English - Alias for ko-en direction.
    /// </summary>
    public static TranslatorModelInfo OpusMtKoEnAlias { get; } = new()
    {
        Id = "onnx-community/opus-mt-ko-en",
        AliasName = "ko-en",
        DisplayName = "OPUS-MT Ko-En",
        Architecture = "MarianMT",
        SourceLanguage = "ko",
        TargetLanguage = "en",
        ParametersM = 74f,
        SizeBytes = 445_854_852,
        BleuScore = 35.5f,
        MaxLength = 512,
        VocabSize = 65000,
        // UseAutoDiscovery = true (default), encoder/decoder files auto-discovered
        TokenizerFile = "source.spm",
        Description = "Korean to English translation using OPUS-MT.",
        License = "Apache-2.0"
    };

    /// <summary>
    /// OPUS-MT Japanese to English.
    /// Apache 2.0 license.
    /// </summary>
    public static TranslatorModelInfo OpusMtJaEn { get; } = new()
    {
        Id = "onnx-community/opus-mt-ja-en",
        AliasName = "ja-en",
        DisplayName = "OPUS-MT Ja-En",
        Architecture = "MarianMT",
        SourceLanguage = "ja",
        TargetLanguage = "en",
        ParametersM = 74f,
        SizeBytes = 428_286_352,
        BleuScore = 32.0f,
        MaxLength = 512,
        VocabSize = 65000,
        // UseAutoDiscovery = true (default), encoder/decoder files auto-discovered
        TokenizerFile = "source.spm",
        Description = "Japanese to English translation using OPUS-MT.",
        License = "Apache-2.0"
    };

    /// <summary>
    /// OPUS-MT Chinese to English.
    /// Apache 2.0 license.
    /// </summary>
    public static TranslatorModelInfo OpusMtZhEn { get; } = new()
    {
        Id = "onnx-community/opus-mt-zh-en",
        AliasName = "zh-en",
        DisplayName = "OPUS-MT Zh-En",
        Architecture = "MarianMT",
        SourceLanguage = "zh",
        TargetLanguage = "en",
        ParametersM = 74f,
        SizeBytes = 445_854_852,
        BleuScore = 30.5f,
        MaxLength = 512,
        VocabSize = 65000,
        // UseAutoDiscovery = true (default), encoder/decoder files auto-discovered
        TokenizerFile = "source.spm",
        Description = "Chinese to English translation using OPUS-MT.",
        License = "Apache-2.0"
    };

    /// <summary>
    /// Gets all default models.
    /// </summary>
    public static IReadOnlyList<TranslatorModelInfo> All { get; } =
    [
        OpusMtKoEn,
        OpusMtKoEnAlias,
        OpusMtJaEn,
        OpusMtZhEn
    ];
}
