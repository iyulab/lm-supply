namespace LMSupply.Translator.Models;

/// <summary>
/// Default translation model configurations.
/// All models use Apache-2.0 or CC-BY-4.0 compatible licenses for commercial use.
/// </summary>
public static class DefaultModels
{
    /// <summary>
    /// OPUS-MT Korean to English - Default model.
    /// Apache 2.0 license.
    /// </summary>
    public static TranslatorModelInfo OpusMtKoEn { get; } = new()
    {
        Id = "Helsinki-NLP/opus-mt-ko-en",
        Alias = "default",
        DisplayName = "OPUS-MT Ko-En",
        Architecture = "MarianMT",
        SourceLanguage = "ko",
        TargetLanguage = "en",
        ParametersM = 74f,
        SizeBytes = 300_000_000,
        BleuScore = 35.5f,
        MaxLength = 512,
        VocabSize = 65000,
        EncoderFile = "encoder_model.onnx",
        DecoderFile = "decoder_model.onnx",
        TokenizerFile = "source.spm",
        Description = "Korean to English translation using OPUS-MT.",
        License = "Apache-2.0"
    };

    /// <summary>
    /// OPUS-MT Korean to English - Alias for ko-en direction.
    /// </summary>
    public static TranslatorModelInfo OpusMtKoEnAlias { get; } = new()
    {
        Id = "Helsinki-NLP/opus-mt-ko-en",
        Alias = "ko-en",
        DisplayName = "OPUS-MT Ko-En",
        Architecture = "MarianMT",
        SourceLanguage = "ko",
        TargetLanguage = "en",
        ParametersM = 74f,
        SizeBytes = 300_000_000,
        BleuScore = 35.5f,
        MaxLength = 512,
        VocabSize = 65000,
        EncoderFile = "encoder_model.onnx",
        DecoderFile = "decoder_model.onnx",
        TokenizerFile = "source.spm",
        Description = "Korean to English translation using OPUS-MT.",
        License = "Apache-2.0"
    };

    /// <summary>
    /// OPUS-MT English to Korean.
    /// Apache 2.0 license.
    /// </summary>
    public static TranslatorModelInfo OpusMtEnKo { get; } = new()
    {
        Id = "Helsinki-NLP/opus-mt-en-ko",
        Alias = "en-ko",
        DisplayName = "OPUS-MT En-Ko",
        Architecture = "MarianMT",
        SourceLanguage = "en",
        TargetLanguage = "ko",
        ParametersM = 74f,
        SizeBytes = 300_000_000,
        BleuScore = 28.0f,
        MaxLength = 512,
        VocabSize = 65000,
        EncoderFile = "encoder_model.onnx",
        DecoderFile = "decoder_model.onnx",
        TokenizerFile = "source.spm",
        Description = "English to Korean translation using OPUS-MT.",
        License = "Apache-2.0"
    };

    /// <summary>
    /// OPUS-MT Japanese to English.
    /// Apache 2.0 license.
    /// </summary>
    public static TranslatorModelInfo OpusMtJaEn { get; } = new()
    {
        Id = "Helsinki-NLP/opus-mt-ja-en",
        Alias = "ja-en",
        DisplayName = "OPUS-MT Ja-En",
        Architecture = "MarianMT",
        SourceLanguage = "ja",
        TargetLanguage = "en",
        ParametersM = 74f,
        SizeBytes = 300_000_000,
        BleuScore = 32.0f,
        MaxLength = 512,
        VocabSize = 65000,
        EncoderFile = "encoder_model.onnx",
        DecoderFile = "decoder_model.onnx",
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
        Id = "Helsinki-NLP/opus-mt-zh-en",
        Alias = "zh-en",
        DisplayName = "OPUS-MT Zh-En",
        Architecture = "MarianMT",
        SourceLanguage = "zh",
        TargetLanguage = "en",
        ParametersM = 74f,
        SizeBytes = 300_000_000,
        BleuScore = 30.5f,
        MaxLength = 512,
        VocabSize = 65000,
        EncoderFile = "encoder_model.onnx",
        DecoderFile = "decoder_model.onnx",
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
        OpusMtEnKo,
        OpusMtJaEn,
        OpusMtZhEn
    ];
}
