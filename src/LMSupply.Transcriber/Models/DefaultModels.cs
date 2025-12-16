namespace LMSupply.Transcriber.Models;

/// <summary>
/// Default transcription model configurations.
/// All models use MIT license (OpenAI Whisper).
/// </summary>
public static class DefaultModels
{
    /// <summary>
    /// Whisper Tiny - Ultra-fast, smallest model.
    /// MIT license, 39M params, ~150MB.
    /// </summary>
    public static TranscriberModelInfo WhisperTiny { get; } = new()
    {
        Id = "onnx-community/whisper-tiny",
        Alias = "fast",
        DisplayName = "Whisper Tiny",
        Architecture = "Whisper",
        ParametersM = 39f,
        SizeBytes = 150_000_000,
        WerLibriSpeech = 7.6f,
        MaxDurationSeconds = 30,
        SampleRate = 16000,
        NumMelBins = 80,
        IsMultilingual = true,
        Description = "Whisper Tiny for ultra-fast transcription. Best for real-time applications.",
        License = "MIT"
    };

    /// <summary>
    /// Whisper Base - Default balanced model.
    /// MIT license, 74M params, ~290MB.
    /// </summary>
    public static TranscriberModelInfo WhisperBase { get; } = new()
    {
        Id = "onnx-community/whisper-base",
        Alias = "default",
        DisplayName = "Whisper Base",
        Architecture = "Whisper",
        ParametersM = 74f,
        SizeBytes = 290_000_000,
        WerLibriSpeech = 5.0f,
        MaxDurationSeconds = 30,
        SampleRate = 16000,
        NumMelBins = 80,
        IsMultilingual = true,
        Description = "Whisper Base for balanced speed and accuracy.",
        License = "MIT"
    };

    /// <summary>
    /// Whisper Small - Quality model.
    /// MIT license, 244M params, ~970MB.
    /// </summary>
    public static TranscriberModelInfo WhisperSmall { get; } = new()
    {
        Id = "onnx-community/whisper-small",
        Alias = "quality",
        DisplayName = "Whisper Small",
        Architecture = "Whisper",
        ParametersM = 244f,
        SizeBytes = 970_000_000,
        WerLibriSpeech = 3.4f,
        MaxDurationSeconds = 30,
        SampleRate = 16000,
        NumMelBins = 80,
        IsMultilingual = true,
        Description = "Whisper Small for higher accuracy transcription.",
        License = "MIT"
    };

    /// <summary>
    /// Whisper Medium - High quality model.
    /// MIT license, 769M params, ~3GB.
    /// </summary>
    public static TranscriberModelInfo WhisperMedium { get; } = new()
    {
        Id = "onnx-community/whisper-medium",
        Alias = "medium",
        DisplayName = "Whisper Medium",
        Architecture = "Whisper",
        ParametersM = 769f,
        SizeBytes = 3_000_000_000,
        WerLibriSpeech = 2.9f,
        MaxDurationSeconds = 30,
        SampleRate = 16000,
        NumMelBins = 80,
        IsMultilingual = true,
        Description = "Whisper Medium for high quality transcription.",
        License = "MIT"
    };

    /// <summary>
    /// Whisper Large V3 - Highest quality model.
    /// MIT license, 1550M params, ~6GB.
    /// </summary>
    public static TranscriberModelInfo WhisperLargeV3 { get; } = new()
    {
        Id = "onnx-community/whisper-large-v3",
        Alias = "large",
        DisplayName = "Whisper Large V3",
        Architecture = "Whisper",
        ParametersM = 1550f,
        SizeBytes = 6_000_000_000,
        WerLibriSpeech = 2.5f,
        MaxDurationSeconds = 30,
        SampleRate = 16000,
        NumMelBins = 128, // Large V3 uses 128 mel bins
        IsMultilingual = true,
        Description = "Whisper Large V3 for highest quality transcription.",
        License = "MIT"
    };

    /// <summary>
    /// Whisper Base English-only - Optimized for English.
    /// MIT license, 74M params, ~290MB.
    /// </summary>
    public static TranscriberModelInfo WhisperBaseEn { get; } = new()
    {
        Id = "onnx-community/whisper-base.en",
        Alias = "english",
        DisplayName = "Whisper Base (English)",
        Architecture = "Whisper",
        ParametersM = 74f,
        SizeBytes = 290_000_000,
        WerLibriSpeech = 4.3f,
        MaxDurationSeconds = 30,
        SampleRate = 16000,
        NumMelBins = 80,
        IsMultilingual = false,
        SupportedLanguages = ["en"],
        Description = "Whisper Base optimized for English-only transcription.",
        License = "MIT"
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
        WhisperBaseEn
    ];
}
