namespace LMSupply.Detector;

/// <summary>
/// Configuration options for the object detector.
/// </summary>
public sealed class DetectorOptions : LMSupplyOptionsBase
{
    /// <summary>
    /// Gets or sets the model identifier.
    /// <para>Supports:</para>
    /// <list type="bullet">
    /// <item>Preset aliases: "default", "fast", "quality", "large"</item>
    /// <item>HuggingFace model IDs: "PekingU/rtdetr_r18vd"</item>
    /// <item>Local file paths: "/path/to/model.onnx"</item>
    /// </list>
    /// <para>Default: "default" (RT-DETR R18)</para>
    /// </summary>
    public string ModelId { get; set; } = "default";

    /// <summary>
    /// Gets or sets the confidence threshold for detections.
    /// Detections with confidence below this value are filtered out.
    /// <para>Default: 0.25</para>
    /// </summary>
    public float ConfidenceThreshold { get; set; } = 0.25f;

    /// <summary>
    /// Gets or sets the IoU threshold for Non-Maximum Suppression.
    /// Only applies to models that require NMS (not RT-DETR).
    /// <para>Default: 0.45</para>
    /// </summary>
    public float IouThreshold { get; set; } = 0.45f;

    /// <summary>
    /// Gets or sets the maximum number of detections to return.
    /// <para>Default: 100</para>
    /// </summary>
    public int MaxDetections { get; set; } = 100;

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
    /// Gets or sets class labels to filter. If null or empty, all classes are returned.
    /// <para>Default: null (return all classes)</para>
    /// </summary>
    public IReadOnlySet<int>? ClassFilter { get; set; }

    /// <summary>
    /// Creates a copy of these options.
    /// </summary>
    public DetectorOptions Clone() => new()
    {
        ModelId = ModelId,
        ConfidenceThreshold = ConfidenceThreshold,
        IouThreshold = IouThreshold,
        MaxDetections = MaxDetections,
        CacheDirectory = CacheDirectory,
        Provider = Provider,
        DisableAutoDownload = DisableAutoDownload,
        ThreadCount = ThreadCount,
        ClassFilter = ClassFilter is not null ? new HashSet<int>(ClassFilter) : null
    };
}
