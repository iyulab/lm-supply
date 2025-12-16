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
    /// Gets or sets the number of inference threads.
    /// <para>Default: null (uses Environment.ProcessorCount)</para>
    /// </summary>
    public int? ThreadCount { get; set; }

    /// <summary>
    /// Creates a copy of these options.
    /// </summary>
    public TranscriberOptions Clone() => new()
    {
        ModelId = ModelId,
        CacheDirectory = CacheDirectory,
        Provider = Provider,
        DisableAutoDownload = DisableAutoDownload,
        ThreadCount = ThreadCount
    };
}

/// <summary>
/// Options for a single transcription operation.
/// </summary>
public sealed class TranscribeOptions
{
    /// <summary>
    /// Gets or sets the language code for transcription.
    /// If null, the model will auto-detect the language.
    /// <para>Default: null (auto-detect)</para>
    /// </summary>
    public string? Language { get; set; }

    /// <summary>
    /// Gets or sets whether to enable translation to English.
    /// <para>Default: false</para>
    /// </summary>
    public bool Translate { get; set; }

    /// <summary>
    /// Gets or sets whether to include word-level timestamps.
    /// <para>Default: false</para>
    /// </summary>
    public bool WordTimestamps { get; set; }

    /// <summary>
    /// Gets or sets the initial prompt to guide the transcription style.
    /// <para>Default: null</para>
    /// </summary>
    public string? InitialPrompt { get; set; }

    /// <summary>
    /// Gets or sets the maximum number of tokens to generate.
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
    public float Temperature { get; set; } = 0.0f;

    /// <summary>
    /// Gets or sets the compression ratio threshold.
    /// Segments above this threshold may be retried with higher temperature.
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
