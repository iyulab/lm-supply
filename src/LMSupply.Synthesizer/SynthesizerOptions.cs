namespace LMSupply.Synthesizer;

/// <summary>
/// Options for configuring the synthesizer model.
/// </summary>
public sealed class SynthesizerOptions : LMSupplyOptionsBase
{
    /// <summary>
    /// Gets or sets the model identifier.
    /// Can be an alias (e.g., "default", "fast"), HuggingFace model ID, or local path.
    /// <para>Default: "default"</para>
    /// </summary>
    public string ModelId { get; set; } = "default";

    /// <summary>
    /// Gets or sets whether to disable automatic model download.
    /// When true, loading uses only the local cache and throws <see cref="LMSupply.Exceptions.ModelNotFoundException"/>
    /// if a model file is not there, writing nothing to the cache.
    /// This covers model files only. The native ONNX Runtime an ONNX model runs on is provisioned separately; to keep
    /// it off the network too, see <see cref="LMSupply.Runtime.RuntimeManager.Configure"/>.
    /// <para>Default: false</para>
    /// </summary>
    public bool DisableAutoDownload { get; set; }

    /// <summary>
    /// Gets or sets the phonemizer that turns text into the voice's phonemes; when set, it is used for every voice. When
    /// unset, the package's own is used: the built-in English phonemizer for a voice of the <c>espeak</c> type in English
    /// (eSpeak NG voices <c>en</c>, <c>en-us</c>, <c>en-gb</c>, <c>en-gb-x-rp</c>), and the text itself for a voice whose
    /// phonemes are its text (<see cref="PhonemeScheme.Type"/> <c>"text"</c>). For any other voice, synthesizing text
    /// without a phonemizer throws <see cref="NotSupportedException"/>; phonemes can always be passed directly
    /// (<see cref="ISynthesizerModel.SynthesizePhonemesAsync"/>).
    /// <para>Default: null</para>
    /// </summary>
    public IPhonemizer? Phonemizer { get; set; }

    /// <summary>
    /// A copy with every option carried over — the loader keeps its own copy so a caller's later edits
    /// do not reach a running model.
    /// </summary>
    public SynthesizerOptions Clone() => new()
    {
        ModelId = ModelId,
        Provider = Provider,
        CacheDirectory = CacheDirectory,
        ThreadCount = ThreadCount,
        LogLevel = LogLevel,
        QuantizationHint = QuantizationHint,
        DisableAutoDownload = DisableAutoDownload,
        Phonemizer = Phonemizer
    };
}

/// <summary>
/// Options for a single synthesis operation.
/// </summary>
public sealed class SynthesizeOptions
{
    /// <summary>
    /// Gets or sets the speaking speed multiplier, relative to the pace the voice declares for itself
    /// (its config's <c>length_scale</c>, which is divided by this value).
    /// Values greater than 1.0 speed up, less than 1.0 slow down.
    /// <para>Default: 1.0</para>
    /// </summary>
    public float Speed { get; set; } = 1.0f;

    /// <summary>
    /// Gets or sets the speaker ID for multi-speaker models: 0 up to the voice's speaker count. A single-speaker voice
    /// ignores it.
    /// <para>Default: 0</para>
    /// </summary>
    public int SpeakerId { get; set; }

    /// <summary>
    /// Gets or sets the noise scale for variability.
    /// Higher values produce more expressive speech.
    /// <para>Default: null — the voice's own default (its config's <c>noise_scale</c>)</para>
    /// </summary>
    public float? NoiseScale { get; set; }

    /// <summary>
    /// Gets or sets the noise width for duration variability.
    /// <para>Default: null — the voice's own default (its config's <c>noise_w</c>)</para>
    /// </summary>
    public float? NoiseWidth { get; set; }

    /// <summary>
    /// Gets or sets the output audio format.
    /// <para>Default: Wav</para>
    /// </summary>
    public AudioFormat OutputFormat { get; set; } = AudioFormat.Wav;
}

/// <summary>
/// Output audio format options.
/// </summary>
public enum AudioFormat
{
    /// <summary>
    /// WAV format (uncompressed PCM).
    /// </summary>
    Wav,

    /// <summary>
    /// Raw PCM samples (16-bit signed integer).
    /// </summary>
    RawPcm16,

    /// <summary>
    /// Raw PCM samples (32-bit float).
    /// </summary>
    RawFloat32
}
