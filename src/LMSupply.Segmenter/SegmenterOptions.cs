namespace LMSupply.Segmenter;

/// <summary>
/// Configuration options for the image segmenter.
/// </summary>
public sealed class SegmenterOptions : LMSupplyOptionsBase
{
    /// <summary>
    /// Gets or sets the model identifier.
    /// <para>Supports:</para>
    /// <list type="bullet">
    /// <item>Preset aliases: "default", "fast", "quality", "large"</item>
    /// <item>HuggingFace model IDs: "nvidia/segformer-b0-finetuned-ade-512-512"</item>
    /// <item>Local file paths: "/path/to/model.onnx"</item>
    /// </list>
    /// <para>Default: "default" (SegFormer-B0)</para>
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
    /// Gets or sets whether to resize output to match original image dimensions.
    /// <para>Default: true</para>
    /// </summary>
    public bool ResizeToOriginal { get; set; } = true;

    /// <summary>
    /// Creates a copy of these options.
    /// </summary>
    public SegmenterOptions Clone() => new()
    {
        ModelId = ModelId,
        CacheDirectory = CacheDirectory,
        Provider = Provider,
        DisableAutoDownload = DisableAutoDownload,
        ThreadCount = ThreadCount,
        ResizeToOriginal = ResizeToOriginal
    };
}
