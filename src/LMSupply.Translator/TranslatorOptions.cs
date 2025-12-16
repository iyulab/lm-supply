namespace LMSupply.Translator;

/// <summary>
/// Configuration options for the translator.
/// </summary>
public sealed class TranslatorOptions : LMSupplyOptionsBase
{
    /// <summary>
    /// Gets or sets the model identifier.
    /// <para>Supports:</para>
    /// <list type="bullet">
    /// <item>Preset aliases: "default", "ko-en", "en-ko", "fast"</item>
    /// <item>HuggingFace model IDs: "Helsinki-NLP/opus-mt-ko-en"</item>
    /// <item>Local file paths: "/path/to/model.onnx"</item>
    /// </list>
    /// <para>Default: "default" (Korean to English)</para>
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
    /// Gets or sets the maximum length of generated text.
    /// <para>Default: 512</para>
    /// </summary>
    public int MaxLength { get; set; } = 512;

    /// <summary>
    /// Gets or sets the beam width for beam search decoding.
    /// <para>Default: 4</para>
    /// </summary>
    public int BeamWidth { get; set; } = 4;

    /// <summary>
    /// Gets or sets whether to use greedy decoding instead of beam search.
    /// Faster but may produce lower quality translations.
    /// <para>Default: false</para>
    /// </summary>
    public bool UseGreedyDecoding { get; set; }

    /// <summary>
    /// Gets or sets the length penalty for beam search (alpha parameter).
    /// Values greater than 1.0 favor longer sequences.
    /// <para>Default: 1.0</para>
    /// </summary>
    public float LengthPenalty { get; set; } = 1.0f;

    /// <summary>
    /// Gets or sets the repetition penalty for beam search.
    /// Values greater than 1.0 penalize repeated tokens.
    /// <para>Default: 1.0</para>
    /// </summary>
    public float RepetitionPenalty { get; set; } = 1.0f;

    /// <summary>
    /// Creates a copy of these options.
    /// </summary>
    public TranslatorOptions Clone() => new()
    {
        ModelId = ModelId,
        CacheDirectory = CacheDirectory,
        Provider = Provider,
        DisableAutoDownload = DisableAutoDownload,
        ThreadCount = ThreadCount,
        MaxLength = MaxLength,
        BeamWidth = BeamWidth,
        UseGreedyDecoding = UseGreedyDecoding,
        LengthPenalty = LengthPenalty,
        RepetitionPenalty = RepetitionPenalty
    };
}
