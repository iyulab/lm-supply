namespace LMSupply.Ocr.Models;

/// <summary>
/// Provides definitions for built-in OCR text detection models.
/// </summary>
internal static class DefaultDetectionModels
{
    private const string DefaultRepoId = "monkt/paddleocr-onnx";

    /// <summary>
    /// PP-OCRv5 mobile Detection (DBNet), from the model's own repository — the primary built-in detection model. The recognizers
    /// of this library are PP-OCRv5 models, which the v5 detector suits; on rendered pages of Korean, English, Japanese and
    /// Chinese text it reads with fewer errors than <see cref="DbNetV3"/>, and it is 4.8 MB.
    /// </summary>
    public static DetectionModelInfo DbNetV5 { get; } = new(
        RepoId: "PaddlePaddle/PP-OCRv5_mobile_det_onnx",
        AliasName: "dbnet-v5",
        DisplayName: "PP-OCRv5 Mobile Detection (DBNet)",
        ModelFile: "inference.onnx",
        InputWidth: 960,
        InputHeight: 960)
    {
        License = "Apache-2.0"
    };

    /// <summary>
    /// Same model registered under the "default" alias.
    /// </summary>
    public static DetectionModelInfo DbNetV5Default { get; } = DbNetV5 with { AliasName = "default" };

    /// <summary>
    /// PaddleOCR v3 Detection (DBNet), the built-in detection model before v5: smaller (2.3 MB) and the "fast" alias.
    /// </summary>
    public static DetectionModelInfo DbNetV3 { get; } = new(
        RepoId: DefaultRepoId,
        AliasName: "dbnet-v3",
        DisplayName: "PaddleOCR v3 Detection (DBNet)",
        ModelFile: "det.onnx",
        InputWidth: 960,
        InputHeight: 960)
    {
        Subfolder = "detection/v3",
        License = "Apache-2.0"
    };

    /// <summary>
    /// Same model registered under the "fast" alias (single-model domain).
    /// </summary>
    public static DetectionModelInfo DbNetV3Fast { get; } = DbNetV3 with { AliasName = "fast" };

    /// <summary>
    /// Gets all built-in detection models.
    /// Primary alias first so it becomes the canonical entry in GetAvailableModels().
    /// </summary>
    public static IReadOnlyList<DetectionModelInfo> All { get; } =
    [
        DbNetV5,          // dbnet-v5 (primary)
        DbNetV5Default,   // default (alias)
        DbNetV3,          // dbnet-v3
        DbNetV3Fast,      // fast (alias)
    ];
}
