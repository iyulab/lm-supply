using LMSupply.Hardware;

namespace LMSupply.Detector.Models;

/// <summary>
/// Metadata about a detector model.
/// </summary>
/// <remarks>
/// A record rather than a class so a derived copy can be taken with <c>with</c>. The registry's
/// auto-selection used to rebuild this by listing every property by hand, which silently drops any
/// property added later - exactly the failure that would have sent an auto-selected model through the
/// wrong decoder.
/// </remarks>
public sealed record DetectorModelInfo : IModelInfoBase, IModelMemoryInfo
{
    /// <summary>
    /// Gets or sets the model ID (HuggingFace repo ID or local path).
    /// </summary>
    public required string Id { get; init; }

    /// <summary>
    /// Gets or sets the model alias name (e.g., "default", "fast").
    /// </summary>
    public required string AliasName { get; init; }

    /// <summary>
    /// Gets or sets the display name.
    /// </summary>
    public required string DisplayName { get; init; }

    /// <summary>
    /// Gets or sets the model architecture (e.g., "RT-DETR", "YOLOv10").
    /// </summary>
    public string Architecture { get; init; } = "RT-DETR";

    /// <summary>
    /// Gets or sets the number of parameters in millions.
    /// </summary>
    public float ParametersM { get; init; }

    /// <summary>
    /// Gets or sets the model size in bytes.
    /// </summary>
    public long SizeBytes { get; init; }

    /// <summary>
    /// Gets or sets the mAP@0.5-0.95 on COCO validation set.
    /// </summary>
    public float MapCoco { get; init; }

    /// <summary>
    /// The input width this model was exported for.
    /// </summary>
    /// <remarks>
    /// Width and height are separate because they are separate in practice - the licence-plate model takes
    /// 320x240. A single square size could not describe it, and squeezing a plate into a square input is not
    /// a formatting detail: the offsets the model emits are read against priors derived from these numbers.
    /// </remarks>
    public int InputWidth { get; init; } = 640;

    /// <summary>
    /// The input height this model was exported for.
    /// </summary>
    public int InputHeight { get; init; } = 640;

    /// <summary>
    /// The vocabulary this model was trained on, in class-id order. Defaults to COCO-80, which every model
    /// shipped so far uses.
    /// </summary>
    /// <remarks>
    /// Post-processing labelled every detection from <see cref="CocoLabels"/> regardless of the model, so a
    /// single-class detector's class 0 came back as "person" — wrong, and wrong invisibly, since the string is
    /// a plausible COCO label rather than an error. Carrying the vocabulary on the model is what makes a
    /// face or plate detector expressible at all.
    /// </remarks>
    public IReadOnlyList<string> ClassLabels { get; init; } = CocoLabels.Labels;

    /// <summary>
    /// The number of classes — derived from <see cref="ClassLabels"/> rather than declared beside it, so the
    /// two cannot disagree. The previous shape allowed <c>NumClasses = 1</c> on a model still labelled from
    /// eighty names.
    /// </summary>
    public int NumClasses => ClassLabels.Count;

    /// <summary>
    /// The label for a class id, or <c>"unknown"</c> when the id is outside this model's vocabulary.
    /// </summary>
    /// <remarks>
    /// Out of range is deliberately not borrowed from COCO: a mislabelled detection that carries a plausible
    /// name passes a consumer's filters unnoticed, while "unknown" does not.
    /// </remarks>
    public string LabelFor(int classId) =>
        classId >= 0 && classId < ClassLabels.Count ? ClassLabels[classId] : "unknown";

    /// <summary>
    /// The shape of this model's raw output, which selects the decoder. Defaults to
    /// <see cref="DetectorOutputLayout.RtDetr"/>, the family every built-in alias belongs to.
    /// </summary>
    public DetectorOutputLayout OutputLayout { get; init; } = DetectorOutputLayout.RtDetr;

    /// <summary>
    /// How this model expects its input tensor built. Defaults to <see cref="DetectorInputFormat.ScaledRgb"/>.
    /// </summary>
    public DetectorInputFormat InputFormat { get; init; } = DetectorInputFormat.ScaledRgb;

    /// <summary>
    /// Whether this model needs non-maximum suppression - derived from <see cref="OutputLayout"/> rather
    /// than declared beside it, so a model cannot claim to be NMS-free while being decoded by a head that
    /// emits overlapping anchors.
    /// </summary>
    public bool RequiresNms => OutputLayout.RequiresNms();

    /// <summary>
    /// The number of keypoints this model emits, derived from <see cref="OutputLayout"/> for the same
    /// reason as <see cref="RequiresNms"/>: a count that disagreed with the decoder used to be expressible,
    /// and produced keypoints read out of a tensor that never held them.
    /// </summary>
    public int NumKeypoints => OutputLayout.KeypointCount();

    /// <summary>
    /// Gets or sets the ONNX file path relative to model directory.
    /// </summary>
    public string OnnxFile { get; init; } = "model.onnx";

    /// <summary>
    /// Gets or sets the model description.
    /// </summary>
    public string Description { get; init; } = string.Empty;

    /// <summary>
    /// Gets or sets the license identifier.
    /// </summary>
    public string License { get; init; } = "Apache-2.0";

    /// <summary>
    /// Gets model size as a human-readable string.
    /// </summary>
    public string SizeDisplay => SizeBytes switch
    {
        >= 1_000_000_000 => $"{SizeBytes / 1_000_000_000.0:F1} GB",
        >= 1_000_000 => $"{SizeBytes / 1_000_000.0:F0} MB",
        >= 1_000 => $"{SizeBytes / 1_000.0:F0} KB",
        _ => $"{SizeBytes} B"
    };

    // IModelMemoryInfo explicit implementation
    long? IModelMemoryInfo.EstimatedSizeBytes => SizeBytes;
    long IModelMemoryInfo.ParameterCount => (long)(ParametersM * 1_000_000);
    string? IModelMemoryInfo.QuantizationType => null;
}
