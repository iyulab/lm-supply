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
    /// Gets or sets the initial prompt to guide the transcription style.
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
    /// Gets or sets the beam width for beam search decoding.
    /// <para>Default: 5</para>
    /// </summary>
    public int BeamWidth { get; set; } = 5;

    /// <summary>
    /// Gets or sets the temperature for sampling.
    /// Lower values make output more deterministic.
    /// <para>Default: 0.0 (greedy)</para>
    /// </summary>
    public float Temperature { get; set; }

    /// <summary>
    /// Gets or sets the compression ratio threshold.
    /// Segments whose text compresses above this ratio (highly repetitive text, a typical sign of a
    /// hallucination loop) are dropped. They are not re-decoded: there is no temperature fallback.
    /// <para>Default: 2.4</para>
    /// </summary>
    public float CompressionRatioThreshold { get; set; } = 2.4f;

    /// <summary>
    /// Gets or sets the no-speech probability threshold.
    /// Segments with no-speech probability above this are skipped.
    /// <para>Default: 0.6</para>
    /// </summary>
    public float NoSpeechThreshold { get; set; } = 0.6f;
}
