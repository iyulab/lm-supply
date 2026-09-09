namespace LMSupply.Detector.Models;

/// <summary>
/// Default detector model configurations.
/// Uses RT-DETR v2 ONNX models from xnorpx (Apache 2.0 compatible).
/// YOLO models are AGPL-3.0 and require Ultralytics license for commercial use.
/// </summary>
public static class DefaultModels
{
    /// <summary>
    /// RT-DETR v2 Small - Default balanced model.
    /// Apache 2.0 license, NMS-free, fast inference.
    /// </summary>
    public static DetectorModelInfo RtDetrV2S { get; } = new()
    {
        Id = "xnorpx/rt-detr2-onnx:s",
        AliasName = "default",
        DisplayName = "RT-DETR v2 Small",
        Architecture = "RT-DETR",
        ParametersM = 20f,
        SizeBytes = 80_500_000,
        MapCoco = 48.1f,
        InputSize = 640,
        RequiresNms = false,
        OnnxFile = "rt-detrv2-s.onnx",
        Description = "RT-DETR v2 Small for balanced speed and accuracy.",
        License = "Apache-2.0"
    };

    /// <summary>
    /// RT-DETR v2 Medium - Quality model.
    /// Apache 2.0 license, NMS-free, 51.9 mAP.
    /// </summary>
    public static DetectorModelInfo RtDetrV2M { get; } = new()
    {
        Id = "xnorpx/rt-detr2-onnx:m",
        AliasName = "quality",
        DisplayName = "RT-DETR v2 Medium",
        Architecture = "RT-DETR",
        ParametersM = 36f,
        SizeBytes = 133_000_000,
        MapCoco = 51.9f,
        InputSize = 640,
        RequiresNms = false,
        OnnxFile = "rt-detrv2-m.onnx",
        Description = "RT-DETR v2 Medium for higher accuracy detection.",
        License = "Apache-2.0"
    };

    /// <summary>
    /// RT-DETR v2 Large - Large model.
    /// Apache 2.0 license, NMS-free, 53.4 mAP.
    /// </summary>
    public static DetectorModelInfo RtDetrV2L { get; } = new()
    {
        Id = "xnorpx/rt-detr2-onnx:l",
        AliasName = "large",
        DisplayName = "RT-DETR v2 Large",
        Architecture = "RT-DETR",
        ParametersM = 42f,
        SizeBytes = 169_000_000,
        MapCoco = 53.4f,
        InputSize = 640,
        RequiresNms = false,
        OnnxFile = "rt-detrv2-l.onnx",
        Description = "RT-DETR v2 Large for highest accuracy detection.",
        License = "Apache-2.0"
    };

    /// <summary>
    /// RT-DETR v2 Mini-Small - Fast/lightweight model.
    /// Apache 2.0 license, NMS-free, smaller than Small variant.
    /// </summary>
    public static DetectorModelInfo RtDetrV2MS { get; } = new()
    {
        Id = "xnorpx/rt-detr2-onnx:ms",
        AliasName = "fast",
        DisplayName = "RT-DETR v2 Mini-Small",
        Architecture = "RT-DETR",
        ParametersM = 15f,
        SizeBytes = 126_000_000,
        MapCoco = 46.0f,
        InputSize = 640,
        RequiresNms = false,
        OnnxFile = "rt-detrv2-ms.onnx",
        Description = "RT-DETR v2 Mini-Small for lightweight/fast inference.",
        License = "Apache-2.0"
    };

    /// <summary>
    /// RT-DETR v2 Extra Large - Highest accuracy model.
    /// Apache 2.0 license, NMS-free, 54.3 mAP.
    /// </summary>
    public static DetectorModelInfo RtDetrV2X { get; } = new()
    {
        Id = "xnorpx/rt-detr2-onnx:x",
        AliasName = "xlarge",
        DisplayName = "RT-DETR v2 XLarge",
        Architecture = "RT-DETR",
        ParametersM = 76f,
        SizeBytes = 300_000_000,
        MapCoco = 54.3f,
        InputSize = 640,
        RequiresNms = false,
        OnnxFile = "rt-detrv2-x.onnx",
        Description = "RT-DETR v2 Extra Large for maximum accuracy.",
        License = "Apache-2.0"
    };

    // Backward compatibility aliases
    public static DetectorModelInfo RtDetrR18 => RtDetrV2S;
    public static DetectorModelInfo RtDetrR50 => RtDetrV2M;
    public static DetectorModelInfo RtDetrR101 => RtDetrV2L;
    public static DetectorModelInfo EfficientDetD0 => RtDetrV2MS;

    /// <summary>
    /// Gets all default models.
    /// </summary>
    public static IReadOnlyList<DetectorModelInfo> All { get; } =
    [
        RtDetrV2S,
        RtDetrV2M,
        RtDetrV2L,
        RtDetrV2MS,
        RtDetrV2X
    ];
}
