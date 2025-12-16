namespace LMSupply.Ocr;

/// <summary>
/// Configuration options for the OCR engine.
/// </summary>
public sealed class OcrOptions : LMSupplyOptionsBase
{
    /// <summary>
    /// Language hint for recognition model selection.
    /// Default is "en" (English).
    /// </summary>
    public string LanguageHint { get; set; } = "en";

    /// <summary>
    /// Minimum confidence threshold for text detection.
    /// Detections below this threshold are discarded.
    /// Default is 0.5.
    /// </summary>
    public float DetectionThreshold { get; set; } = 0.5f;

    /// <summary>
    /// Minimum confidence threshold for text recognition.
    /// Results below this threshold may be marked as low confidence.
    /// Default is 0.5.
    /// </summary>
    public float RecognitionThreshold { get; set; } = 0.5f;

    /// <summary>
    /// Binarization threshold for DBNet post-processing.
    /// Default is 0.3.
    /// </summary>
    public float BinarizationThreshold { get; set; } = 0.3f;

    /// <summary>
    /// Maximum number of candidates for NMS.
    /// Default is 1000.
    /// </summary>
    public int MaxCandidates { get; set; } = 1000;

    /// <summary>
    /// Unclip ratio for polygon expansion in DBNet.
    /// Default is 1.5.
    /// </summary>
    public float UnclipRatio { get; set; } = 1.5f;

    /// <summary>
    /// Minimum area (in pixels) for detected text boxes.
    /// Smaller boxes are filtered out.
    /// Default is 10.
    /// </summary>
    public int MinBoxArea { get; set; } = 10;

    /// <summary>
    /// Whether to use polygon coordinates instead of rectangular bounding boxes.
    /// Polygons provide more accurate text region boundaries.
    /// Default is true.
    /// </summary>
    public bool UsePolygon { get; set; } = true;
}
