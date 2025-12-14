namespace LocalAI.Synthesizer.Models;

/// <summary>
/// Default TTS model configurations.
/// Uses Piper VITS models which are optimized for ONNX Runtime.
/// </summary>
public static class DefaultModels
{
    /// <summary>
    /// English US (Lessac) - High quality, default voice.
    /// Apache-2.0 license, ~64MB.
    /// </summary>
    public static SynthesizerModelInfo EnUsLessac { get; } = new()
    {
        Id = "rhasspy/piper-voices",
        Alias = "default",
        DisplayName = "English US (Lessac)",
        Architecture = "VITS",
        Language = "en-US",
        VoiceName = "en_US-lessac-medium",
        NumSpeakers = 1,
        SampleRate = 22050,
        ModelFile = "en_US-lessac-medium.onnx",
        ConfigFile = "en_US-lessac-medium.onnx.json",
        SizeBytes = 64_000_000,
        Description = "High-quality US English female voice.",
        License = "MIT"
    };

    /// <summary>
    /// English US (Ryan) - Fast, lightweight voice.
    /// Apache-2.0 license, ~16MB.
    /// </summary>
    public static SynthesizerModelInfo EnUsRyan { get; } = new()
    {
        Id = "rhasspy/piper-voices",
        Alias = "fast",
        DisplayName = "English US (Ryan)",
        Architecture = "VITS",
        Language = "en-US",
        VoiceName = "en_US-ryan-low",
        NumSpeakers = 1,
        SampleRate = 16000,
        ModelFile = "en_US-ryan-low.onnx",
        ConfigFile = "en_US-ryan-low.onnx.json",
        SizeBytes = 16_000_000,
        Description = "Fast US English male voice, optimized for speed.",
        License = "MIT"
    };

    /// <summary>
    /// English US (Amy) - High quality female voice.
    /// Apache-2.0 license, ~64MB.
    /// </summary>
    public static SynthesizerModelInfo EnUsAmy { get; } = new()
    {
        Id = "rhasspy/piper-voices",
        Alias = "quality",
        DisplayName = "English US (Amy)",
        Architecture = "VITS",
        Language = "en-US",
        VoiceName = "en_US-amy-medium",
        NumSpeakers = 1,
        SampleRate = 22050,
        ModelFile = "en_US-amy-medium.onnx",
        ConfigFile = "en_US-amy-medium.onnx.json",
        SizeBytes = 64_000_000,
        Description = "High-quality US English female voice.",
        License = "MIT"
    };

    /// <summary>
    /// English GB (Semaine) - British English voice.
    /// Apache-2.0 license, ~64MB.
    /// </summary>
    public static SynthesizerModelInfo EnGbSemaine { get; } = new()
    {
        Id = "rhasspy/piper-voices",
        Alias = "british",
        DisplayName = "English GB (Semaine)",
        Architecture = "VITS",
        Language = "en-GB",
        VoiceName = "en_GB-semaine-medium",
        NumSpeakers = 1,
        SampleRate = 22050,
        ModelFile = "en_GB-semaine-medium.onnx",
        ConfigFile = "en_GB-semaine-medium.onnx.json",
        SizeBytes = 64_000_000,
        Description = "British English female voice.",
        License = "MIT"
    };

    /// <summary>
    /// Korean (KSS) - Korean voice.
    /// Apache-2.0 license, ~64MB.
    /// </summary>
    public static SynthesizerModelInfo KoKr { get; } = new()
    {
        Id = "rhasspy/piper-voices",
        Alias = "korean",
        DisplayName = "Korean (KSS)",
        Architecture = "VITS",
        Language = "ko-KR",
        VoiceName = "ko_KR-kss-x_low",
        NumSpeakers = 1,
        SampleRate = 22050,
        ModelFile = "ko_KR-kss-x_low.onnx",
        ConfigFile = "ko_KR-kss-x_low.onnx.json",
        SizeBytes = 64_000_000,
        Description = "Korean female voice.",
        License = "MIT"
    };

    /// <summary>
    /// Japanese - Japanese voice.
    /// Apache-2.0 license, ~64MB.
    /// </summary>
    public static SynthesizerModelInfo JaJp { get; } = new()
    {
        Id = "rhasspy/piper-voices",
        Alias = "japanese",
        DisplayName = "Japanese",
        Architecture = "VITS",
        Language = "ja-JP",
        VoiceName = "ja_JP-jsut-medium",
        NumSpeakers = 1,
        SampleRate = 22050,
        ModelFile = "ja_JP-jsut-medium.onnx",
        ConfigFile = "ja_JP-jsut-medium.onnx.json",
        SizeBytes = 64_000_000,
        Description = "Japanese female voice.",
        License = "MIT"
    };

    /// <summary>
    /// Chinese (Mandarin) - Chinese voice.
    /// Apache-2.0 license, ~64MB.
    /// </summary>
    public static SynthesizerModelInfo ZhCn { get; } = new()
    {
        Id = "rhasspy/piper-voices",
        Alias = "chinese",
        DisplayName = "Chinese (Mandarin)",
        Architecture = "VITS",
        Language = "zh-CN",
        VoiceName = "zh_CN-huayan-medium",
        NumSpeakers = 1,
        SampleRate = 22050,
        ModelFile = "zh_CN-huayan-medium.onnx",
        ConfigFile = "zh_CN-huayan-medium.onnx.json",
        SizeBytes = 64_000_000,
        Description = "Mandarin Chinese female voice.",
        License = "MIT"
    };

    /// <summary>
    /// Gets all default models.
    /// </summary>
    public static IReadOnlyList<SynthesizerModelInfo> All { get; } =
    [
        EnUsLessac,
        EnUsRyan,
        EnUsAmy,
        EnGbSemaine,
        KoKr,
        JaJp,
        ZhCn
    ];
}
