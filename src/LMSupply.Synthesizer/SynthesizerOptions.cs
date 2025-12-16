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
    /// Gets or sets the number of threads for CPU inference.
    /// If null, uses ONNX Runtime defaults.
    /// <para>Default: null</para>
    /// </summary>
    public int? ThreadCount { get; set; }

    /// <summary>
    /// Creates a deep copy of these options.
    /// </summary>
    /// <returns>A new instance with the same values.</returns>
    public SynthesizerOptions Clone() => new()
    {
        ModelId = ModelId,
        Provider = Provider,
        CacheDirectory = CacheDirectory,
        ThreadCount = ThreadCount
    };
}

/// <summary>
/// Options for a single synthesis operation.
/// </summary>
public sealed class SynthesizeOptions
{
    /// <summary>
    /// Gets or sets the speaking speed multiplier.
    /// Values greater than 1.0 speed up, less than 1.0 slow down.
    /// <para>Default: 1.0</para>
    /// </summary>
    public float Speed { get; set; } = 1.0f;

    /// <summary>
    /// Gets or sets the pitch shift in semitones.
    /// <para>Default: 0.0</para>
    /// </summary>
    public float Pitch { get; set; } = 0.0f;

    /// <summary>
    /// Gets or sets the speaker ID for multi-speaker models.
    /// <para>Default: 0</para>
    /// </summary>
    public int SpeakerId { get; set; } = 0;

    /// <summary>
    /// Gets or sets the noise scale for variability.
    /// Higher values produce more expressive speech.
    /// <para>Default: 0.667</para>
    /// </summary>
    public float NoiseScale { get; set; } = 0.667f;

    /// <summary>
    /// Gets or sets the noise width for duration variability.
    /// <para>Default: 0.8</para>
    /// </summary>
    public float NoiseWidth { get; set; } = 0.8f;

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
