namespace LMSupply.Detector.Models;

/// <summary>
/// Metadata about a detector model.
/// </summary>
public sealed class DetectorModelInfo
{
    /// <summary>
    /// Gets or sets the model ID (HuggingFace repo ID or local path).
    /// </summary>
    public required string Id { get; init; }

    /// <summary>
    /// Gets or sets the model alias (e.g., "default", "fast").
    /// </summary>
    public required string Alias { get; init; }

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
    /// Gets or sets the input image size (width = height).
    /// </summary>
    public int InputSize { get; init; } = 640;

    /// <summary>
    /// Gets or sets the number of classes.
    /// </summary>
    public int NumClasses { get; init; } = 80;

    /// <summary>
    /// Gets or sets whether this model requires NMS post-processing.
    /// </summary>
    public bool RequiresNms { get; init; }

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
}
