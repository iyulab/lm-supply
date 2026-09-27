namespace LMSupply.Synthesizer.Models;

/// <summary>
/// Default TTS model configurations.
/// Uses Piper VITS models which are optimized for ONNX Runtime.
/// </summary>
public static class DefaultModels
{
    /// <summary>
    /// English US (LJSpeech) - the default voice: the only one here whose recordings are public domain, so an application
    /// may ship it without a license review. ~64MB.
    /// Path in repo: en/en_US/ljspeech/medium/
    /// </summary>
    public static SynthesizerModelInfo EnUsLjSpeech { get; } = new()
    {
        Id = "rhasspy/piper-voices",
        AliasName = "default",
        DisplayName = "English US (LJSpeech)",
        Architecture = "VITS",
        Language = "en-US",
        VoiceName = "en/en_US/ljspeech/medium",
        NumSpeakers = 1,
        SampleRate = 22050,
        ModelFile = "en_US-ljspeech-medium.onnx",
        ConfigFile = "en_US-ljspeech-medium.onnx.json",
        SizeBytes = 63_531_379,
        Description = "US English female voice (LJ Speech, public domain).",
        License = "Public-Domain"
    };

    /// <summary>
    /// English US (Lessac) - High quality female voice.
    /// Voice license: Blizzard 2013 Lessac research license (non-commercial), ~64MB.
    /// Path in repo: en/en_US/lessac/medium/
    /// </summary>
    public static SynthesizerModelInfo EnUsLessac { get; } = new()
    {
        Id = "rhasspy/piper-voices",
        AliasName = "lessac",
        DisplayName = "English US (Lessac)",
        Architecture = "VITS",
        Language = "en-US",
        VoiceName = "en/en_US/lessac/medium",
        NumSpeakers = 1,
        SampleRate = 22050,
        ModelFile = "en_US-lessac-medium.onnx",
        ConfigFile = "en_US-lessac-medium.onnx.json",
        SizeBytes = 63_200_000,
        Description = "High-quality US English female voice.",
        License = "Blizzard-2013-Lessac (non-commercial research)"
    };

    /// <summary>
    /// English US (Ryan) - Fast, lightweight voice.
    /// Voice license: CC BY-NC-SA 4.0 (non-commercial), ~16MB.
    /// Path in repo: en/en_US/ryan/low/
    /// </summary>
    public static SynthesizerModelInfo EnUsRyan { get; } = new()
    {
        Id = "rhasspy/piper-voices",
        AliasName = "fast",
        DisplayName = "English US (Ryan)",
        Architecture = "VITS",
        Language = "en-US",
        VoiceName = "en/en_US/ryan/low",
        NumSpeakers = 1,
        SampleRate = 16000,
        ModelFile = "en_US-ryan-low.onnx",
        ConfigFile = "en_US-ryan-low.onnx.json",
        SizeBytes = 16_000_000,
        Description = "Fast US English male voice, optimized for speed.",
        License = "CC-BY-NC-SA-4.0"
    };

    /// <summary>
    /// English US (Amy) - High quality female voice.
    /// Voice license: unspecified in its model card, ~64MB.
    /// Path in repo: en/en_US/amy/medium/
    /// </summary>
    public static SynthesizerModelInfo EnUsAmy { get; } = new()
    {
        Id = "rhasspy/piper-voices",
        AliasName = "quality",
        DisplayName = "English US (Amy)",
        Architecture = "VITS",
        Language = "en-US",
        VoiceName = "en/en_US/amy/medium",
        NumSpeakers = 1,
        SampleRate = 22050,
        ModelFile = "en_US-amy-medium.onnx",
        ConfigFile = "en_US-amy-medium.onnx.json",
        SizeBytes = 64_000_000,
        Description = "High-quality US English female voice.",
        License = "Unspecified (see the voice's MODEL_CARD)"
    };

    /// <summary>
    /// English GB (Semaine) - British English voice.
    /// Voice license: CC BY-NC-SA 4.0 (non-commercial), ~64MB.
    /// Path in repo: en/en_GB/semaine/medium/
    /// </summary>
    public static SynthesizerModelInfo EnGbSemaine { get; } = new()
    {
        Id = "rhasspy/piper-voices",
        AliasName = "british",
        DisplayName = "English GB (Semaine)",
        Architecture = "VITS",
        Language = "en-GB",
        VoiceName = "en/en_GB/semaine/medium",
        NumSpeakers = 1,
        SampleRate = 22050,
        ModelFile = "en_GB-semaine-medium.onnx",
        ConfigFile = "en_GB-semaine-medium.onnx.json",
        SizeBytes = 64_000_000,
        Description = "British English female voice.",
        License = "CC-BY-NC-SA-4.0"
    };

    /// <summary>
    /// Korean (KSS) - Korean voice (medium; there is no x_low build).
    /// Voice license: CC BY-NC-SA 4.0 (non-commercial), ~63MB.
    /// Path in repo: ko/ko_KR/kss/medium/
    /// </summary>
    public static SynthesizerModelInfo KoKr { get; } = new()
    {
        Id = "rhasspy/piper-voices",
        AliasName = "korean",
        DisplayName = "Korean (KSS)",
        Architecture = "VITS",
        Language = "ko-KR",
        VoiceName = "ko/ko_KR/kss/medium",
        NumSpeakers = 1,
        SampleRate = 22050,
        ModelFile = "ko_KR-kss-medium.onnx",
        ConfigFile = "ko_KR-kss-medium.onnx.json",
        SizeBytes = 63_221_984,
        Description = "Korean female voice.",
        License = "CC-BY-NC-SA-4.0"
    };

    /// <summary>
    /// Japanese (JSUT) - Japanese voice.
    /// Voice license: dataset license not verified (the pinned voice does not exist), ~64MB.
    /// Path in repo: ja/ja_JP/jsut/medium/
    /// </summary>
    public static SynthesizerModelInfo JaJp { get; } = new()
    {
        Id = "rhasspy/piper-voices",
        AliasName = "japanese",
        DisplayName = "Japanese",
        Architecture = "VITS",
        Language = "ja-JP",
        VoiceName = "ja/ja_JP/jsut/medium",
        NumSpeakers = 1,
        SampleRate = 22050,
        ModelFile = "ja_JP-jsut-medium.onnx",
        ConfigFile = "ja_JP-jsut-medium.onnx.json",
        SizeBytes = 64_000_000,
        Description = "Japanese female voice.",
        License = "Unknown"
    };

    /// <summary>
    /// Chinese (Mandarin) - Chinese voice.
    /// Voice license: unknown (its model card says so), ~64MB.
    /// Path in repo: zh/zh_CN/huayan/medium/
    /// </summary>
    public static SynthesizerModelInfo ZhCn { get; } = new()
    {
        Id = "rhasspy/piper-voices",
        AliasName = "chinese",
        DisplayName = "Chinese (Mandarin)",
        Architecture = "VITS",
        Language = "zh-CN",
        VoiceName = "zh/zh_CN/huayan/medium",
        NumSpeakers = 1,
        SampleRate = 22050,
        ModelFile = "zh_CN-huayan-medium.onnx",
        ConfigFile = "zh_CN-huayan-medium.onnx.json",
        SizeBytes = 64_000_000,
        Description = "Mandarin Chinese female voice.",
        License = "Unknown"
    };

    /// <summary>
    /// Gets all default models.
    /// </summary>
    public static IReadOnlyList<SynthesizerModelInfo> All { get; } =
    [
        EnUsLjSpeech,
        EnUsLessac,
        EnUsRyan,
        EnUsAmy,
        EnGbSemaine,
        KoKr,
        JaJp,
        ZhCn
    ];
}
