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
    /// Gets or sets whether to disable automatic model download.
    /// When true, loading uses only the local cache and throws <see cref="LMSupply.Exceptions.ModelNotFoundException"/>
    /// if a model file is not there, writing nothing to the cache.
    /// <para>Default: false</para>
    /// </summary>
    public bool DisableAutoDownload { get; set; }
}
