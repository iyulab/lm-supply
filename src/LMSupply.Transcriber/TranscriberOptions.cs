namespace LMSupply.Transcriber;

/// <summary>
/// Configuration options for the transcriber model loading.
/// </summary>
public sealed class TranscriberOptions : LMSupplyOptionsBase
{
    /// <summary>
    /// Gets or sets the model identifier.
    /// <para>Supports:</para>
    /// <list type="bullet">
    /// <item>Preset aliases: "default", "fast", "quality", "large"</item>
    /// <item>HuggingFace model IDs: "openai/whisper-base"</item>
    /// <item>Local file paths: "/path/to/model.onnx"</item>
    /// </list>
    /// <para>Default: "default" (Whisper Base)</para>
    /// </summary>
    public string ModelId { get; set; } = "default";

    /// <summary>
    /// Gets or sets whether to disable automatic model download.
    /// When true, throws an exception if the model is not found locally.
    /// <para>Default: false</para>
    /// </summary>
    public bool DisableAutoDownload { get; set; }

    /// <summary>
    /// Creates a copy of these options.
    /// </summary>
    public TranscriberOptions Clone() => new()
    {
        ModelId = ModelId,
        CacheDirectory = CacheDirectory,
        Provider = Provider,
        DisableAutoDownload = DisableAutoDownload,
        ThreadCount = ThreadCount,
        QuantizationHint = QuantizationHint
    };
}

/// <summary>
/// Options for a single transcription operation.
/// </summary>
public sealed class TranscribeOptions
{
    /// <summary>
    /// Gets or sets the language code for transcription (ISO 639-1, e.g. "en", "ko").
    /// If null, the language is identified from the first 30 s window with Whisper's
    /// language-identification step and reused for the rest of the audio; the outcome is reported
    /// as <see cref="TranscriptionResult.Language"/> / <see cref="TranscriptionResult.LanguageProbability"/>.
    /// A hint skips identification (and leaves the probability null).
    /// <para>Default: null (auto-detect)</para>
    /// </summary>
    public string? Language { get; set; }

    /// <summary>
    /// Gets or sets whether to enable translation to English.
    /// <para>Default: false</para>
    /// </summary>
    public bool Translate { get; set; }

    /// <summary>
    /// Gets or sets whether to enable timestamp token generation in the transcription.
    /// <para>
    /// <b>Important:</b> Despite the property name, this enables <b>segment-level</b> timestamps,
    /// NOT word-level timestamps. The name is retained for compatibility with Whisper's API.
    /// </para>
    /// <para>
    /// When true, the model generates timestamp tokens that create natural segment breaks
    /// based on speech patterns. Each segment will have Start and End timestamps.
    /// When false, creates a single segment per 30-second audio chunk.
    /// </para>
    /// <para>
    /// <b>Word-level timestamps:</b> True word-level timestamps (populating the Words property
    /// in TranscriptionSegment) require cross-attention alignment with Dynamic Time Warping (DTW),
    /// which is not currently implemented. For word-level timestamps, consider using
    /// <see href="https://github.com/linto-ai/whisper-timestamped">whisper-timestamped</see> or
    /// <see href="https://github.com/SYSTRAN/faster-whisper">faster-whisper</see> directly.
    /// </para>
    /// <para>Default: false</para>
    /// </summary>
    public bool WordTimestamps { get; set; }

    /// <summary>
    /// Gets or sets text the model reads as what came before the audio — names, terms, spelling and
    /// punctuation style the transcript should follow (e.g. "Attendees: Meena, Joon. Agenda: Q3 OKRs.").
    /// Placed before the start-of-transcript token behind <c>&lt;|startofprev|&gt;</c>, as Whisper does,
    /// for every 30-second window; only its last 223 tokens are used. It guides; it does not force.
    /// Needs the model's BPE merges (<c>merges.txt</c> or <c>tokenizer.json</c>), which Whisper models ship.
    /// <para>Default: null</para>
    /// </summary>
    public string? InitialPrompt { get; set; }

    /// <summary>
    /// Gets or sets the maximum number of tokens to generate for each 30-second window.
    /// The model's text context (448 tokens, prompt included) caps it, so values above what the
    /// context leaves after the prompt have no further effect. Must be at least 1.
    /// <para>Default: 448</para>
    /// </summary>
    public int MaxTokens { get; set; } = 448;

    /// <summary>
    /// Gets or sets the temperature of the first decode of each 30-second window.
    /// At 0 the decoder is greedy: it takes the most likely token at every step. Above 0 it samples
    /// from the token distribution scaled by this temperature — higher is more varied.
    /// <para>
    /// A window whose result fails the quality checks (<see cref="CompressionRatioThreshold"/>,
    /// <see cref="LogProbThreshold"/>) is decoded again at a temperature raised by
    /// <see cref="TemperatureIncrementOnFallback"/>, up to 1.0 — Whisper's temperature fallback. Must be
    /// between 0 and 1.
    /// </para>
    /// <para>Default: 0.0 (greedy)</para>
    /// </summary>
    public float Temperature { get; set; }

    /// <summary>
    /// Gets or sets how much the temperature rises each time a window is decoded again after failing
    /// the quality checks. Attempts continue while the temperature stays at or below 1.0; the last
    /// attempt's result is kept. Set to 0 to decode every window once.
    /// <para>Default: 0.2 (0.0, 0.2, 0.4, 0.6, 0.8, 1.0 — up to six attempts)</para>
    /// </summary>
    public float TemperatureIncrementOnFallback { get; set; } = 0.2f;

    /// <summary>
    /// Gets or sets the compression ratio threshold.
    /// A window whose text compresses above this ratio (highly repetitive text, a typical sign of a
    /// hallucination loop) is decoded again at a higher temperature (see
    /// <see cref="TemperatureIncrementOnFallback"/>). Segments still above it after the last attempt
    /// are dropped.
    /// <para>Default: 2.4</para>
    /// </summary>
    public float CompressionRatioThreshold { get; set; } = 2.4f;

    /// <summary>
    /// Gets or sets the average log-probability threshold. A window whose tokens average a lower
    /// log-probability than this — the decoder was unsure of what it produced — is decoded again at a
    /// higher temperature. Set to null to judge windows by <see cref="CompressionRatioThreshold"/> only.
    /// <para>Default: -1.0</para>
    /// </summary>
    public float? LogProbThreshold { get; set; } = -1.0f;

    /// <summary>
    /// Gets or sets the no-speech probability threshold.
    /// Segments with no-speech probability above this are skipped. A window above it whose average
    /// log-probability is also below <see cref="LogProbThreshold"/> is taken as silence and is not
    /// decoded again.
    /// <para>Default: 0.6</para>
    /// </summary>
    public float NoSpeechThreshold { get; set; } = 0.6f;
}
