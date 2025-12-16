namespace LMSupply.Segmenter.Models;

/// <summary>
/// Default segmentation model configurations.
/// All models use MIT/Apache-2.0 compatible licenses for commercial use.
/// </summary>
public static class DefaultModels
{
    /// <summary>
    /// SegFormer-B0 - Default lightweight model.
    /// MIT license, 3.7M params, fast inference.
    /// </summary>
    public static SegmenterModelInfo SegFormerB0 { get; } = new()
    {
        Id = "nvidia/segformer-b0-finetuned-ade-512-512",
        Alias = "default",
        DisplayName = "SegFormer-B0",
        Architecture = "SegFormer",
        ParametersM = 3.7f,
        SizeBytes = 15_000_000,
        MIoU = 38.0f,
        InputSize = 512,
        NumClasses = 150,
        OnnxFile = "model.onnx",
        Dataset = "ADE20K",
        Description = "SegFormer-B0 for efficient semantic segmentation. Best for real-time applications.",
        License = "MIT"
    };

    /// <summary>
    /// SegFormer-B1 - Fast balanced model.
    /// MIT license, 13.7M params.
    /// </summary>
    public static SegmenterModelInfo SegFormerB1 { get; } = new()
    {
        Id = "nvidia/segformer-b1-finetuned-ade-512-512",
        Alias = "fast",
        DisplayName = "SegFormer-B1",
        Architecture = "SegFormer",
        ParametersM = 13.7f,
        SizeBytes = 55_000_000,
        MIoU = 42.2f,
        InputSize = 512,
        NumClasses = 150,
        OnnxFile = "model.onnx",
        Dataset = "ADE20K",
        Description = "SegFormer-B1 for balanced speed and accuracy.",
        License = "MIT"
    };

    /// <summary>
    /// SegFormer-B2 - Quality balanced model.
    /// MIT license, 27.4M params.
    /// </summary>
    public static SegmenterModelInfo SegFormerB2 { get; } = new()
    {
        Id = "nvidia/segformer-b2-finetuned-ade-512-512",
        Alias = "quality",
        DisplayName = "SegFormer-B2",
        Architecture = "SegFormer",
        ParametersM = 27.4f,
        SizeBytes = 110_000_000,
        MIoU = 46.5f,
        InputSize = 512,
        NumClasses = 150,
        OnnxFile = "model.onnx",
        Dataset = "ADE20K",
        Description = "SegFormer-B2 for higher accuracy segmentation.",
        License = "MIT"
    };

    /// <summary>
    /// SegFormer-B5 - Large high-accuracy model.
    /// MIT license, 84.6M params.
    /// </summary>
    public static SegmenterModelInfo SegFormerB5 { get; } = new()
    {
        Id = "nvidia/segformer-b5-finetuned-ade-640-640",
        Alias = "large",
        DisplayName = "SegFormer-B5",
        Architecture = "SegFormer",
        ParametersM = 84.6f,
        SizeBytes = 340_000_000,
        MIoU = 51.0f,
        InputSize = 640,
        NumClasses = 150,
        OnnxFile = "model.onnx",
        Dataset = "ADE20K",
        Description = "SegFormer-B5 for highest accuracy. Best for offline processing.",
        License = "MIT"
    };

    /// <summary>
    /// MobileSAM - Lightweight Segment Anything Model.
    /// Apache-2.0 license, interactive point/box prompt segmentation.
    /// </summary>
    public static SegmenterModelInfo MobileSAM { get; } = new()
    {
        Id = "ChaoningZhang/MobileSAM",
        Alias = "interactive",
        DisplayName = "MobileSAM",
        Architecture = "MobileSAM",
        ParametersM = 9.8f,
        SizeBytes = 40_000_000,
        MIoU = 0, // Not applicable for prompt-based
        InputSize = 1024,
        NumClasses = 1, // Binary segmentation
        EncoderFile = "mobile_sam_image_encoder.onnx",
        DecoderFile = "mobile_sam_mask_decoder.onnx",
        Dataset = "SA-1B",
        Description = "MobileSAM for interactive segmentation. Supports point and box prompts.",
        License = "Apache-2.0"
    };

    /// <summary>
    /// Gets all default models.
    /// </summary>
    public static IReadOnlyList<SegmenterModelInfo> All { get; } =
    [
        SegFormerB0,
        SegFormerB1,
        SegFormerB2,
        SegFormerB5,
        MobileSAM
    ];
}
