namespace LMSupply.Transcriber.Models;

/// <summary>
/// Default transcription model configurations.
/// All models use MIT license (OpenAI Whisper).
/// </summary>
/// <remarks>
/// <para>Alias naming convention:</para>
/// <list type="bullet">
///   <item><term>Standard tiers</term><description>fast, default, quality, large, turbo, medium, distil — multilingual or English</description></item>
///   <item><term>Language name</term><description>english — standalone language alias for backward compatibility</description></item>
///   <item><term>Language-specific tier</term><description>{tier}-{lang} using ISO 639-1 codes — e.g., large-ko, fast-ja</description></item>
/// </list>
/// </remarks>
public static class DefaultModels
{
    /// <summary>
    /// Whisper Tiny - Ultra-fast, smallest model.
    /// MIT license, 39M params, ~150MB.
    /// </summary>
    public static TranscriberModelInfo WhisperTiny { get; } = new()
    {
        Id = "onnx-community/whisper-tiny",
        AliasName = "fast",
        DisplayName = "Whisper Tiny",
        Architecture = "Whisper",
        ParametersM = 39f,
        SizeBytes = 150_000_000,
        WerLibriSpeech = 7.6f,
        MaxDurationSeconds = 30,
        SampleRate = 16000,
        NumMelBins = 80,
        HiddenSize = 384,
        IsMultilingual = true,
        Description = "Whisper Tiny for ultra-fast transcription. Best for real-time applications.",
        License = "MIT",
        DecoderFile = "decoder_model_merged.onnx"
    };

    /// <summary>
    /// Whisper Base - Default balanced model.
    /// MIT license, 74M params, ~290MB.
    /// </summary>
    public static TranscriberModelInfo WhisperBase { get; } = new()
    {
        Id = "onnx-community/whisper-base",
        AliasName = "default",
        DisplayName = "Whisper Base",
        Architecture = "Whisper",
        ParametersM = 74f,
        SizeBytes = 290_000_000,
        WerLibriSpeech = 5.0f,
        MaxDurationSeconds = 30,
        SampleRate = 16000,
        NumMelBins = 80,
        HiddenSize = 512,
        IsMultilingual = true,
        Description = "Whisper Base for balanced speed and accuracy.",
        License = "MIT",
        DecoderFile = "decoder_model_merged.onnx"
    };

    /// <summary>
    /// Whisper Small - Quality model.
    /// MIT license, 244M params, ~970MB.
    /// </summary>
    public static TranscriberModelInfo WhisperSmall { get; } = new()
    {
        Id = "onnx-community/whisper-small",
        AliasName = "quality",
        DisplayName = "Whisper Small",
        Architecture = "Whisper",
        ParametersM = 244f,
        SizeBytes = 970_000_000,
        WerLibriSpeech = 3.4f,
        MaxDurationSeconds = 30,
        SampleRate = 16000,
        NumMelBins = 80,
        HiddenSize = 768,
        IsMultilingual = true,
        Description = "Whisper Small for higher accuracy transcription.",
        License = "MIT",
        DecoderFile = "decoder_model_merged.onnx"
    };

    /// <summary>
    /// Whisper Medium (English) with timestamps - High quality model.
    /// MIT license, 769M params, ~3GB.
    /// Note: onnx-community/whisper-medium is gated; using English-only timestamped variant.
    /// </summary>
    public static TranscriberModelInfo WhisperMedium { get; } = new()
    {
        Id = "onnx-community/whisper-medium.en_timestamped",
        AliasName = "medium",
        DisplayName = "Whisper Medium (English)",
        Architecture = "Whisper",
        ParametersM = 769f,
        SizeBytes = 3_000_000_000,
        WerLibriSpeech = 2.9f,
        MaxDurationSeconds = 30,
        SampleRate = 16000,
        NumMelBins = 80,
        HiddenSize = 1024,
        IsMultilingual = false,
        SupportedLanguages = ["en"],
        Description = "Whisper Medium for high quality English transcription with word-level timestamps.",
        License = "MIT",
        DecoderFile = "decoder_model_merged.onnx"
    };

    /// <summary>
    /// Whisper Large V3 - Full 32-decoder-layer multilingual model.
    /// MIT license, 1550M params, ~6.4GB. Highest quality, slowest inference.
    /// Note: Uses whisper-large-v3-ONNX (public), not whisper-large-v3 (gated).
    /// </summary>
    public static TranscriberModelInfo WhisperLargeV3 { get; } = new()
    {
        Id = "onnx-community/whisper-large-v3-ONNX",
        AliasName = "large",
        DisplayName = "Whisper Large V3",
        Architecture = "Whisper",
        ParametersM = 1550f,
        SizeBytes = 6_400_000_000,
        WerLibriSpeech = 2.5f,
        MaxDurationSeconds = 30,
        SampleRate = 16000,
        NumMelBins = 128, // Large V3 uses 128 mel bins
        HiddenSize = 1280,
        IsMultilingual = true,
        Description = "Whisper Large V3 with full 32 decoder layers for highest quality multilingual transcription.",
        License = "MIT",
        DecoderFile = "decoder_model_merged.onnx"
    };

    /// <summary>
    /// Whisper Large V3 Turbo - Fast high-quality model.
    /// MIT license, 809M params, ~3GB. 8x faster than Large V3.
    /// Released 2024-10, best for production workloads.
    /// </summary>
    public static TranscriberModelInfo WhisperLargeV3Turbo { get; } = new()
    {
        Id = "onnx-community/whisper-large-v3-turbo",
        AliasName = "turbo",
        DisplayName = "Whisper Large V3 Turbo",
        Architecture = "Whisper",
        ParametersM = 809f,
        SizeBytes = 3_200_000_000,
        WerLibriSpeech = 2.7f,
        MaxDurationSeconds = 30,
        SampleRate = 16000,
        NumMelBins = 128,
        HiddenSize = 1280,
        IsMultilingual = true,
        Description = "Whisper Large V3 Turbo - 8x faster than V3 with near-equal quality.",
        License = "MIT",
        DecoderFile = "decoder_model_merged.onnx"
    };

    /// <summary>
    /// Distil-Whisper Large V3 - Distilled fast model.
    /// MIT license, 756M params, ~3GB. 6x faster than Large V3.
    /// Best balance of speed and quality for English.
    /// </summary>
    public static TranscriberModelInfo DistilWhisperLargeV3 { get; } = new()
    {
        Id = "distil-whisper/distil-large-v3",
        AliasName = "distil",
        DisplayName = "Distil-Whisper Large V3",
        Architecture = "Whisper",
        ParametersM = 756f,
        SizeBytes = 3_000_000_000,
        WerLibriSpeech = 2.8f,
        MaxDurationSeconds = 30,
        SampleRate = 16000,
        NumMelBins = 128,
        HiddenSize = 1280,
        IsMultilingual = false,
        SupportedLanguages = ["en"],
        Description = "Distil-Whisper Large V3 - 6x faster distilled model for English.",
        License = "MIT",
        DecoderFile = "decoder_model_merged.onnx"
    };

    /// <summary>
    /// Whisper Large V3 Turbo Korean - Korean-optimized large model.
    /// MIT license, 809M params, ~3GB. Fine-tuned for Korean ASR.
    /// Based on royshilkrot/whisper-large-v3-turbo-korean-ggml.
    /// </summary>
    public static TranscriberModelInfo WhisperLargeV3TurboKorean { get; } = new()
    {
        Id = "onnx-community/whisper-large-v3-turbo-korean-ggml-ONNX",
        AliasName = "large-ko",
        DisplayName = "Whisper Large V3 Turbo (Korean)",
        Architecture = "Whisper",
        ParametersM = 809f,
        SizeBytes = 3_200_000_000,
        MaxDurationSeconds = 30,
        SampleRate = 16000,
        NumMelBins = 128,
        HiddenSize = 1280,
        IsMultilingual = false,
        SupportedLanguages = ["ko"],
        Description = "Whisper Large V3 Turbo fine-tuned for Korean speech recognition.",
        License = "MIT",
        DecoderFile = "decoder_model_merged.onnx"
    };

    /// <summary>
    /// Whisper Base English-only - Optimized for English.
    /// MIT license, 74M params, ~290MB.
    /// </summary>
    public static TranscriberModelInfo WhisperBaseEn { get; } = new()
    {
        Id = "onnx-community/whisper-base.en",
        AliasName = "english",
        DisplayName = "Whisper Base (English)",
        Architecture = "Whisper",
        ParametersM = 74f,
        SizeBytes = 290_000_000,
        WerLibriSpeech = 4.3f,
        MaxDurationSeconds = 30,
        SampleRate = 16000,
        NumMelBins = 80,
        HiddenSize = 512,
        IsMultilingual = false,
        SupportedLanguages = ["en"],
        Description = "Whisper Base optimized for English-only transcription.",
        License = "MIT",
        DecoderFile = "decoder_model_merged.onnx"
    };

    /// <summary>
    /// NVIDIA Parakeet TDT 0.6B v3 — Conformer encoder + Token-and-Duration Transducer, 25 European languages,
    /// int8 ONNX export by istupakov (~670 MB: encoder 652 MB + decoder/joint 18 MB + 128-mel preprocessor + vocab).
    /// Not a Whisper model: no 30 s window, no autoregressive text decoder (so no runaway-repeat class of failure),
    /// no language-id output. Opt-in only — never an "auto" candidate. CC-BY-4.0 per the model card.
    /// </summary>
    public static TranscriberModelInfo ParakeetTdt06BV3 { get; } = new()
    {
        Id = "istupakov/parakeet-tdt-0.6b-v3-onnx",
        AliasName = "parakeet-tdt",
        DisplayName = "Parakeet TDT 0.6B v3 (int8)",
        Architecture = TranscriberArchitectures.ParakeetTdt,
        ParametersM = 600f,
        SizeBytes = 670_619_803,
        WerLibriSpeech = null,
        MaxDurationSeconds = 60,
        SampleRate = 16000,
        NumMelBins = 128,
        HiddenSize = 1024,
        EncoderFile = "encoder-model.int8.onnx",
        DecoderFile = "decoder_joint-model.int8.onnx",
        IsMultilingual = true,
        SupportedLanguages = ["en", "es", "fr", "de", "bg", "hr", "cs", "da", "nl", "et", "fi", "el", "hu", "it", "lv", "lt", "mt", "pl", "pt", "ro", "sk", "sl", "sv", "ru", "uk"],
        Description = "NVIDIA Parakeet TDT 0.6B v3 (ONNX int8) — fast CPU transcription for 25 European languages; transducer decoding, no translation.",
        License = "CC-BY-4.0"
    };

    /// <summary>
    /// Gets all default models.
    /// </summary>
    public static IReadOnlyList<TranscriberModelInfo> All { get; } =
    [
        WhisperTiny,
        WhisperBase,
        WhisperSmall,
        WhisperMedium,
        WhisperLargeV3,
        WhisperLargeV3Turbo,
        DistilWhisperLargeV3,
        WhisperLargeV3TurboKorean,
        WhisperBaseEn,
        ParakeetTdt06BV3
    ];
}
