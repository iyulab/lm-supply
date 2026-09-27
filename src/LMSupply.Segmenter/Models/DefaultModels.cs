namespace LMSupply.Segmenter.Models;

/// <summary>
/// Default segmentation model configurations.
/// All models use MIT/Apache-2.0 compatible licenses for commercial use.
/// </summary>
public static class DefaultModels
{
    /// <summary>
    /// SegFormer-B0 - Default lightweight model.
    /// Apache-2.0 license, 3.7M params, fast inference.
    /// Uses optimum ONNX-exported version (nvidia original has no ONNX).
    /// </summary>
    public static SegmenterModelInfo SegFormerB0 { get; } = new()
    {
        Id = "optimum/segformer-b0-finetuned-ade-512-512",
        AliasName = "default",
        DisplayName = "SegFormer-B0",
        Architecture = "SegFormer",
        ParametersM = 3.7f,
        SizeBytes = 15_100_000,
        MIoU = 38.0f,
        InputSize = 512,
        NumClasses = 150,
        OnnxFile = "model.onnx",
        Dataset = "ADE20K",
        Description = "SegFormer-B0 for efficient semantic segmentation. Best for real-time applications.",
        License = "Apache-2.0"
    };

    /// <summary>
    /// MediaPipe Selfie Segmentation - Fast lightweight model.
    /// Apache-2.0 license, MobileNetV3 architecture.
    /// Optimized for person/background segmentation.
    /// </summary>
    public static SegmenterModelInfo MediaPipeSelfie { get; } = new()
    {
        Id = "onnx-community/mediapipe_selfie_segmentation",
        AliasName = "fast",
        DisplayName = "MediaPipe Selfie",
        Architecture = "MobileNetV3",
        ParametersM = 0.7f,
        SizeBytes = 3_000_000,
        MIoU = 0, // Not directly comparable - binary segmentation
        InputSize = 256,
        NumClasses = 2, // Background/Person
        OnnxFile = "model.onnx",
        Dataset = "Selfie",
        Description = "MediaPipe Selfie Segmentation for fast person/background segmentation.",
        License = "Apache-2.0"
    };

    /// <summary>
    /// MaskFormer ResNet50 - Quality model for ADE20K.
    /// Apache-2.0 license, 27.4M params.
    /// Full semantic segmentation with 150 classes.
    /// </summary>
    public static SegmenterModelInfo MaskFormerResNet50 { get; } = new()
    {
        Id = "onnx-community/maskformer-resnet50-ade20k-full",
        AliasName = "quality",
        DisplayName = "MaskFormer ResNet50",
        Architecture = "MaskFormer",
        ParametersM = 44f,
        SizeBytes = 178_000_000,
        MIoU = 44.5f,
        InputSize = 512,
        NumClasses = 150,
        OnnxFile = "model.onnx",
        Dataset = "ADE20K",
        Description = "MaskFormer with ResNet50 for higher accuracy segmentation.",
        License = "Apache-2.0"
    };

    /// <summary>
    /// SegFormer-B0 (large alias) - Same as default, largest available ONNX model.
    /// Note: Larger SegFormer variants (B2-B5) don't have official ONNX exports.
    /// </summary>
    public static SegmenterModelInfo SegFormerB0Large { get; } = new()
    {
        Id = "optimum/segformer-b0-finetuned-ade-512-512",
        AliasName = "large",
        DisplayName = "SegFormer-B0 (Quality)",
        Architecture = "SegFormer",
        ParametersM = 3.7f,
        SizeBytes = 15_100_000,
        MIoU = 38.0f,
        InputSize = 512,
        NumClasses = 150,
        OnnxFile = "model.onnx",
        Dataset = "ADE20K",
        Description = "SegFormer-B0 - largest ONNX-available SegFormer variant.",
        License = "Apache-2.0"
    };

    /// <summary>
    /// MobileSAM - Lightweight Segment Anything Model, interactive point/box prompt segmentation.
    /// ONNX export of the MobileSAM weights (encoder with its preprocessing included + SAM's four-token mask decoder), MIT.
    /// </summary>
    public static SegmenterModelInfo MobileSAM { get; } = new()
    {
        Id = "Acly/MobileSAM",
        AliasName = "interactive",
        DisplayName = "MobileSAM",
        Architecture = "MobileSAM",
        ParametersM = 9.8f,
        SizeBytes = 44_653_652,
        MIoU = 0, // Not applicable for prompt-based
        InputSize = 1024,
        NumClasses = 1, // Binary segmentation
        EncoderFile = "mobile_sam_image_encoder.onnx",
        DecoderFile = "sam_mask_decoder_multi.onnx",
        Dataset = "SA-1B",
        Description = "MobileSAM for interactive segmentation. Supports point and box prompts.",
        License = "MIT"
    };

    /// <summary>
    /// Gets all default models.
    /// </summary>
    public static IReadOnlyList<SegmenterModelInfo> All { get; } =
    [
        SegFormerB0,
        MediaPipeSelfie,
        MaskFormerResNet50,
        SegFormerB0Large,
        MobileSAM
    ];
}
