using System.Diagnostics;

namespace LMSupply.Ocr.Models;

/// <summary>
/// Model registry for OCR text detection models.
/// </summary>
public sealed class OcrDetectionModelRegistry : ModelRegistryBase<DetectionModelInfo>
{
    /// <summary>
    /// Gets the default registry instance with built-in detection models.
    /// </summary>
    public static OcrDetectionModelRegistry Default { get; } = CreateDefault();

    /// <summary>
    /// Builds the default registry: built-in models plus the user's
    /// ~/.lmsupply/aliases.json "ocr-detection" section (fail-soft).
    /// </summary>
    internal static OcrDetectionModelRegistry CreateDefault()
        => AliasConfiguration.ApplyDomain(new OcrDetectionModelRegistry(DefaultDetectionModels.All), AliasConfiguration.Domains.OcrDetection);

    /// <summary>
    /// Initializes a new registry with the specified system models.
    /// </summary>
    /// <param name="systemModels">Models to register as system defaults.</param>
    public OcrDetectionModelRegistry(IEnumerable<DetectionModelInfo> systemModels)
        : base(systemModels) { }

    /// <summary>
    /// Returns the default detection model (DBNet v5 mobile) for the "auto" alias.
    /// Both built-in detection models are small, so no hardware-based selection is needed.
    /// </summary>
    protected override DetectionModelInfo GetAutoModel()
    {
        Trace.TraceInformation("[OcrDetectionModelRegistry] Auto-selecting default detection model");
        return DefaultDetectionModels.DbNetV5 with { AliasName = "auto" };
    }

    /// <summary>
    /// Creates a fallback model info for unknown model IDs (HuggingFace repos or local paths).
    /// </summary>
    protected override DetectionModelInfo CreateFallbackModelInfo(string modelId)
    {
        Trace.TraceInformation($"[OcrDetectionModelRegistry] Creating fallback model info for: {modelId}");

        return new DetectionModelInfo(
            RepoId: modelId,
            AliasName: modelId,
            DisplayName: $"Custom detection model: {modelId}",
            ModelFile: "det.onnx",
            InputWidth: 960,
            InputHeight: 960);
    }
}
