namespace LMSupply.Captioner;

/// <summary>
/// Configuration options for the image captioner.
/// </summary>
public sealed class CaptionerOptions : LMSupplyOptionsBase
{
    /// <summary>
    /// Maximum number of tokens to generate in the caption.
    /// Default is 50.
    /// </summary>
    public int MaxLength { get; set; } = 50;

    /// <summary>
    /// Number of beams for beam search decoding.
    /// 1 = greedy decoding (default), higher values explore more candidates.
    /// </summary>
    public int NumBeams { get; set; } = 1;

    /// <summary>
    /// Temperature for sampling. Lower values make output more deterministic.
    /// Default is 1.0 (no temperature scaling).
    /// </summary>
    public float Temperature { get; set; } = 1.0f;

    /// <summary>
    /// Optional text the caption starts with (conditional captioning). The decoder continues from it, so the returned
    /// caption begins with the prompt — e.g. <c>"a photo of"</c> yields <c>"a photo of a dog on a couch"</c>. Null or
    /// whitespace captions freely.
    /// </summary>
    public string? Prompt { get; set; }

    /// <summary>
    /// How much the caption says. <see cref="CaptionDetail.Brief"/> (the default) is one sentence; the other levels are
    /// read by models that caption at several levels of detail (Florence-2, alias <c>quality</c>). A model that captions
    /// at one level only refuses any other value at load with <see cref="NotSupportedException"/> rather than ignoring it.
    /// A paragraph needs more tokens than the default <see cref="MaxLength"/>; raise it (about 150) for
    /// <see cref="CaptionDetail.Paragraph"/>.
    /// </summary>
    public CaptionDetail Detail { get; set; } = CaptionDetail.Brief;

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
    /// Creates a copy of these options. <see cref="LocalCaptioner"/> works on a copy, so a load never changes the
    /// instance it was given (a <c>:variant</c> qualifier no longer lands in the caller's
    /// <see cref="LMSupplyOptionsBase.QuantizationHint"/>).
    /// </summary>
    public CaptionerOptions Clone() => new()
    {
        MaxLength = MaxLength,
        NumBeams = NumBeams,
        Temperature = Temperature,
        Prompt = Prompt,
        Detail = Detail,
        DisableAutoDownload = DisableAutoDownload,
        CacheDirectory = CacheDirectory,
        Provider = Provider,
        ThreadCount = ThreadCount,
        LogLevel = LogLevel,
        QuantizationHint = QuantizationHint
    };
}

/// <summary>
/// How much a caption says.
/// </summary>
public enum CaptionDetail
{
    /// <summary>One sentence naming the main subject.</summary>
    Brief = 0,

    /// <summary>A few sentences on what is shown.</summary>
    Detailed = 1,

    /// <summary>A paragraph.</summary>
    Paragraph = 2
}
