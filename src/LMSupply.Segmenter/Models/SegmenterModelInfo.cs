namespace LMSupply.Segmenter.Models;

/// <summary>
/// Information about a segmentation model.
/// </summary>
public sealed class SegmenterModelInfo
{
    /// <summary>
    /// Gets or sets the model identifier (HuggingFace repo or local path).
    /// </summary>
    public required string Id { get; init; }

    /// <summary>
    /// Gets or sets the short alias for the model.
    /// </summary>
    public required string Alias { get; init; }

    /// <summary>
    /// Gets or sets the display name.
    /// </summary>
    public required string DisplayName { get; init; }

    /// <summary>
    /// Gets or sets the model architecture (e.g., "SegFormer", "DeepLabV3+").
    /// </summary>
    public required string Architecture { get; init; }

    /// <summary>
    /// Gets or sets the number of parameters in millions.
    /// </summary>
    public float ParametersM { get; init; }

    /// <summary>
    /// Gets or sets the approximate model size in bytes.
    /// </summary>
    public long SizeBytes { get; init; }

    /// <summary>
    /// Gets or sets the mIoU score on the benchmark dataset.
    /// </summary>
    public float MIoU { get; init; }

    /// <summary>
    /// Gets or sets the input image size.
    /// </summary>
    public int InputSize { get; init; } = 512;

    /// <summary>
    /// Gets or sets the number of classes.
    /// </summary>
    public int NumClasses { get; init; }

    /// <summary>
    /// Gets or sets the ONNX file name within the model repository.
    /// </summary>
    public string OnnxFile { get; init; } = "model.onnx";

    /// <summary>
    /// Gets or sets the encoder ONNX file name (for SAM-like models).
    /// If set, the model uses encoder/decoder split architecture.
    /// </summary>
    public string? EncoderFile { get; init; }

    /// <summary>
    /// Gets or sets the decoder ONNX file name (for SAM-like models).
    /// </summary>
    public string? DecoderFile { get; init; }

    /// <summary>
    /// Gets whether this model supports interactive segmentation (SAM-like).
    /// </summary>
    public bool IsInteractive => EncoderFile != null && DecoderFile != null;

    /// <summary>
    /// Gets or sets the dataset the model was trained on.
    /// </summary>
    public string Dataset { get; init; } = "ADE20K";

    /// <summary>
    /// Gets or sets the model description.
    /// </summary>
    public string Description { get; init; } = "";

    /// <summary>
    /// Gets or sets the license identifier.
    /// </summary>
    public string License { get; init; } = "Unknown";
}
