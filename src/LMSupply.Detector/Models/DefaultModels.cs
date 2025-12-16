namespace LMSupply.Detector.Models;

/// <summary>
/// Default detector model configurations.
/// Based on research: RT-DETR (Apache 2.0) recommended for commercial use.
/// YOLO models are AGPL-3.0 and require Ultralytics license for commercial use.
/// </summary>
public static class DefaultModels
{
    /// <summary>
    /// RT-DETR R18 - Default balanced model.
    /// Apache 2.0 license, NMS-free, 46.5 mAP.
    /// </summary>
    public static DetectorModelInfo RtDetrR18 { get; } = new()
    {
        Id = "PekingU/rtdetr_r18vd",
        Alias = "default",
        DisplayName = "RT-DETR R18",
        Architecture = "RT-DETR",
        ParametersM = 20f,
        SizeBytes = 80_000_000,
        MapCoco = 46.5f,
        InputSize = 640,
        NumClasses = 80,
        RequiresNms = false,
        OnnxFile = "model.onnx",
        Description = "RT-DETR with ResNet-18 backbone. Best balance of speed and accuracy.",
        License = "Apache-2.0"
    };

    /// <summary>
    /// RT-DETR R50 - Quality model.
    /// Apache 2.0 license, NMS-free, 53.1 mAP.
    /// </summary>
    public static DetectorModelInfo RtDetrR50 { get; } = new()
    {
        Id = "PekingU/rtdetr_r50vd",
        Alias = "quality",
        DisplayName = "RT-DETR R50",
        Architecture = "RT-DETR",
        ParametersM = 42f,
        SizeBytes = 170_000_000,
        MapCoco = 53.1f,
        InputSize = 640,
        NumClasses = 80,
        RequiresNms = false,
        OnnxFile = "model.onnx",
        Description = "RT-DETR with ResNet-50 backbone. Higher accuracy, moderate speed.",
        License = "Apache-2.0"
    };

    /// <summary>
    /// RT-DETR R101 - Large model.
    /// Apache 2.0 license, NMS-free, 54.3 mAP.
    /// </summary>
    public static DetectorModelInfo RtDetrR101 { get; } = new()
    {
        Id = "PekingU/rtdetr_r101vd",
        Alias = "large",
        DisplayName = "RT-DETR R101",
        Architecture = "RT-DETR",
        ParametersM = 76f,
        SizeBytes = 300_000_000,
        MapCoco = 54.3f,
        InputSize = 640,
        NumClasses = 80,
        RequiresNms = false,
        OnnxFile = "model.onnx",
        Description = "RT-DETR with ResNet-101 backbone. Highest accuracy.",
        License = "Apache-2.0"
    };

    /// <summary>
    /// EfficientDet-D0 - Fast/lightweight model.
    /// Apache 2.0 license, requires NMS, 33.8 mAP.
    /// </summary>
    public static DetectorModelInfo EfficientDetD0 { get; } = new()
    {
        Id = "Kalray/efficientdet-d0",
        Alias = "fast",
        DisplayName = "EfficientDet-D0",
        Architecture = "EfficientDet",
        ParametersM = 3.9f,
        SizeBytes = 15_000_000,
        MapCoco = 33.8f,
        InputSize = 512,
        NumClasses = 80,
        RequiresNms = true,
        OnnxFile = "efficientdet-d0.onnx",
        Description = "EfficientDet-D0 for lightweight/fast inference.",
        License = "Apache-2.0"
    };

    /// <summary>
    /// Gets all default models.
    /// </summary>
    public static IReadOnlyList<DetectorModelInfo> All { get; } =
    [
        RtDetrR18,
        RtDetrR50,
        RtDetrR101,
        EfficientDetD0
    ];
}
