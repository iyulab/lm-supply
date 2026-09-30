namespace LMSupply.Detector.Models;

/// <summary>
/// Default detector model configurations.
/// Uses RT-DETR v2 ONNX models from xnorpx (Apache 2.0 compatible) for the COCO aliases, OpenCV's
/// YuNet for faces (MIT) and licence plates (Apache-2.0), and Google's MoveNet (Apache-2.0) for human pose.
/// YOLO models are AGPL-3.0 and require Ultralytics license for commercial use, so no alias resolves to one.
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
        InputWidth = 640,
        InputHeight = 640,
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
        InputWidth = 640,
        InputHeight = 640,
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
        InputWidth = 640,
        InputHeight = 640,
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
        InputWidth = 640,
        InputHeight = 640,
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
        InputWidth = 640,
        InputHeight = 640,
        OnnxFile = "rt-detrv2-x.onnx",
        Description = "RT-DETR v2 Extra Large for maximum accuracy.",
        License = "Apache-2.0"
    };

    /// <summary>
    /// YuNet face detector - the "face" alias.
    /// MIT licensed, 227 KB, anchor-free with three stride branches and five landmarks per face.
    /// </summary>
    /// <remarks>
    /// COCO has no face class, and "person" is not a substitute for redaction work: blurring a whole person
    /// to hide a face destroys the frame. This is the smallest model in the registry by three orders of
    /// magnitude and runs comfortably faster than video rates on CPU.
    /// </remarks>
    public static DetectorModelInfo YuNetFace { get; } = new()
    {
        Id = "opencv/face_detection_yunet",
        AliasName = "face",
        DisplayName = "YuNet Face Detector",
        Architecture = "YuNet",
        ParametersM = 0.085f,
        SizeBytes = 232_589,
        MapCoco = 0f,
        InputWidth = 640,
        InputHeight = 640,
        ClassLabels = ["face"],
        OutputLayout = DetectorOutputLayout.YuNet,
        InputFormat = DetectorInputFormat.RawBgr,
        OnnxFile = "face_detection_yunet_2023mar.onnx",
        Description = "YuNet face detector with five facial landmarks, for redaction and anonymisation.",
        License = "MIT"
    };

    /// <summary>
    /// Licence-plate YuNet - the "plate" alias.
    /// Apache-2.0, 4.1 MB, prior-box head emitting the four corners of each plate.
    /// </summary>
    /// <remarks>
    /// Shares a name with the face model and almost nothing else: 320x240 input, prior boxes rather than an
    /// anchor-free grid, and a quadrilateral rather than an upright box. A plate photographed from an angle
    /// is not axis-aligned, so the corners are carried as keypoints alongside their upright hull.
    /// </remarks>
    public static DetectorModelInfo YuNetPlate { get; } = new()
    {
        Id = "opencv/license_plate_detection_yunet",
        AliasName = "plate",
        DisplayName = "YuNet Licence Plate Detector",
        Architecture = "LPD-YuNet",
        ParametersM = 1f,
        SizeBytes = 4_146_213,
        MapCoco = 0f,
        InputWidth = 320,
        InputHeight = 240,
        ClassLabels = ["plate"],
        OutputLayout = DetectorOutputLayout.YuNetPlate,
        InputFormat = DetectorInputFormat.RawBgr,
        OnnxFile = "license_plate_detection_lpd_yunet_2023mar.onnx",
        Description = "Licence-plate detector emitting the four corners of each plate, for redaction and anonymisation.",
        License = "Apache-2.0"
    };

    /// <summary>
    /// MoveNet SinglePose Lightning - the "pose" alias.
    /// Apache-2.0, 9.4 MB, 192x192 input, the 17 COCO keypoints of one person per frame.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Google's MoveNet weights are Apache-2.0; this file is a community ONNX conversion of them, published as
    /// Apache-2.0. The training data is COCO plus a Google-internal dataset. Verified at repository revision
    /// <c>ed0f314bb7356fd1dbf1e4f52c2d40791bf6534f</c>: <c>onnx/model.onnx</c>, 9,413,268 bytes, input
    /// <c>input</c> int32 <c>[1, 192, 192, 3]</c>, output <c>output_0</c> float <c>[1, 1, 17, 3]</c>.
    /// </para>
    /// <para>
    /// SinglePose follows the most prominent person in the frame. For more than one person use
    /// <see cref="MoveNetMultiPoseLightning"/>. Pose checkpoints trained on non-commercial datasets (AI
    /// Challenger, CrowdPose, Halpe and the "body7" mixtures that include them) are deliberately not offered.
    /// </para>
    /// <para>
    /// The parameter count is estimated from the float32 file size.
    /// </para>
    /// </remarks>
    public static DetectorModelInfo MoveNetSinglePoseLightning { get; } = new()
    {
        Id = "Xenova/movenet-singlepose-lightning",
        AliasName = "pose",
        DisplayName = "MoveNet SinglePose Lightning",
        Architecture = "MoveNet",
        ParametersM = 2.3f,
        SizeBytes = 9_413_268,
        MapCoco = 0f,
        InputWidth = 192,
        InputHeight = 192,
        ClassLabels = ["person"],
        OutputLayout = DetectorOutputLayout.MoveNetSinglePose,
        InputFormat = DetectorInputFormat.PaddedRgbInt32,
        OnnxFile = "onnx/model.onnx",
        Description = "MoveNet SinglePose Lightning: 17 COCO keypoints of one person, for posture and movement analysis.",
        License = "Apache-2.0"
    };

    /// <summary>
    /// MoveNet SinglePose Thunder - the "pose-quality" alias.
    /// Apache-2.0, 25 MB, 256x256 input, the same output as <see cref="MoveNetSinglePoseLightning"/>.
    /// </summary>
    /// <remarks>
    /// Same licence and provenance as the Lightning model. Verified at repository revision
    /// <c>38296077a99667cdad67af5096ce7eeb9b327453</c>: <c>onnx/model.onnx</c>, 25,067,197 bytes, input
    /// <c>input</c> int32 <c>[1, 256, 256, 3]</c>, output <c>output_0</c> float <c>[1, 1, 17, 3]</c>.
    /// </remarks>
    public static DetectorModelInfo MoveNetSinglePoseThunder { get; } = new()
    {
        Id = "Xenova/movenet-singlepose-thunder",
        AliasName = "pose-quality",
        DisplayName = "MoveNet SinglePose Thunder",
        Architecture = "MoveNet",
        ParametersM = 6.3f,
        SizeBytes = 25_067_197,
        MapCoco = 0f,
        InputWidth = 256,
        InputHeight = 256,
        ClassLabels = ["person"],
        OutputLayout = DetectorOutputLayout.MoveNetSinglePose,
        InputFormat = DetectorInputFormat.PaddedRgbInt32,
        OnnxFile = "onnx/model.onnx",
        Description = "MoveNet SinglePose Thunder: the more accurate single-person model, at roughly twice the cost.",
        License = "Apache-2.0"
    };

    /// <summary>
    /// MoveNet MultiPose Lightning - the "pose-multi" alias.
    /// Apache-2.0, 19 MB, up to six people per frame, each with a box and the 17 COCO keypoints.
    /// </summary>
    /// <remarks>
    /// Same licence and provenance as the Lightning model. Verified at repository revision
    /// <c>8506174df310cf154b97352afa98d1ea4b1e4b47</c>: <c>onnx/model.onnx</c>, 19,049,169 bytes, input
    /// <c>input</c> int32 <c>[1, height, width, 3]</c> (dynamic; both sides a multiple of 32), output
    /// <c>output_0</c> float <c>[1, 6, 56]</c>. Run at 256x256, the size its reference usage recommends.
    /// </remarks>
    public static DetectorModelInfo MoveNetMultiPoseLightning { get; } = new()
    {
        Id = "Xenova/movenet-multipose-lightning",
        AliasName = "pose-multi",
        DisplayName = "MoveNet MultiPose Lightning",
        Architecture = "MoveNet",
        ParametersM = 4.8f,
        SizeBytes = 19_049_169,
        MapCoco = 0f,
        InputWidth = 256,
        InputHeight = 256,
        ClassLabels = ["person"],
        OutputLayout = DetectorOutputLayout.MoveNetMultiPose,
        InputFormat = DetectorInputFormat.PaddedRgbInt32,
        OnnxFile = "onnx/model.onnx",
        Description = "MoveNet MultiPose Lightning: up to six people, each with a box and 17 COCO keypoints.",
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
        RtDetrV2X,
        YuNetFace,
        YuNetPlate,
        MoveNetSinglePoseLightning,
        MoveNetSinglePoseThunder,
        MoveNetMultiPoseLightning
    ];
}
