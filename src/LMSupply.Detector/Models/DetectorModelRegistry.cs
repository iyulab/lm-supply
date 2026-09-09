using System.Diagnostics;
using LMSupply.Hardware;

namespace LMSupply.Detector.Models;

/// <summary>
/// Model registry for the Detector domain.
/// </summary>
public sealed class DetectorModelRegistry : ModelRegistryBase<DetectorModelInfo>
{
    /// <summary>
    /// Auto-selection candidates sorted by size descending (largest first).
    /// </summary>
    private static readonly DetectorModelInfo[] AutoCandidates =
    [
        DefaultModels.RtDetrV2L,   // 42M params, 169MB
        DefaultModels.RtDetrV2M,   // 36M params, 133MB
        DefaultModels.RtDetrV2S,   // 20M params, 80MB
        DefaultModels.RtDetrV2MS,  // 15M params, 126MB
    ];

    /// <summary>
    /// Gets the default registry instance with built-in models.
    /// </summary>
    public static DetectorModelRegistry Default { get; } = CreateDefault();

    /// <summary>
    /// Builds the default registry: built-in models plus the user's
    /// ~/.lmsupply/aliases.json "detector" section (fail-soft).
    /// </summary>
    internal static DetectorModelRegistry CreateDefault()
        => AliasConfiguration.ApplyDomain(new DetectorModelRegistry(DefaultModels.All), AliasConfiguration.Domains.Detector);

    /// <summary>
    /// Initializes a new registry with the specified system models.
    /// </summary>
    /// <param name="systemModels">Models to register as system defaults.</param>
    public DetectorModelRegistry(IEnumerable<DetectorModelInfo> systemModels)
        : base(systemModels) { }

    /// <summary>
    /// Gets the optimal model based on available VRAM.
    /// Selects the largest model that fits in available GPU memory,
    /// falling back to the smallest model if none fit.
    /// </summary>
    protected override DetectorModelInfo GetAutoModel()
    {
        var gpu = HardwareProfile.Current.GpuInfo;
        var availableVram = VramBudget.GetAvailableBytes(gpu);
        Trace.TraceInformation($"[DetectorModelRegistry] Auto-selecting model for VRAM: {availableVram / (1024 * 1024)} MB");

        DetectorModelInfo selected = AutoCandidates[^1]; // default to smallest

        foreach (var candidate in AutoCandidates)
        {
            var memInfo = (IModelMemoryInfo)candidate;
            var size = ModelMemoryEstimator.EstimateModelSizeBytes(
                memInfo.ParameterCount, memInfo.QuantizationType, memInfo.EstimatedSizeBytes);
            if (size <= availableVram)
            {
                selected = candidate;
                Trace.TraceInformation($"[DetectorModelRegistry] Selected: {candidate.Id} ({size / (1024 * 1024)} MB)");
                break;
            }
        }

        return new DetectorModelInfo
        {
            Id = selected.Id,
            AliasName = "auto",
            DisplayName = selected.DisplayName,
            Architecture = selected.Architecture,
            ParametersM = selected.ParametersM,
            SizeBytes = selected.SizeBytes,
            MapCoco = selected.MapCoco,
            InputSize = selected.InputSize,
            ClassLabels = selected.ClassLabels,
            RequiresNms = selected.RequiresNms,
            NumKeypoints = selected.NumKeypoints,
            OnnxFile = selected.OnnxFile,
            Description = selected.Description,
            License = selected.License
        };
    }

    /// <summary>
    /// Creates a fallback model info for unknown model IDs (HuggingFace repos or local paths).
    /// </summary>
    protected override DetectorModelInfo CreateFallbackModelInfo(string modelId)
    {
        Trace.TraceInformation($"[DetectorModelRegistry] Creating fallback model info for: {modelId}");

        // Check if it's a local path with an ONNX file
        if (modelId.EndsWith(".onnx", StringComparison.OrdinalIgnoreCase) ||
            Path.IsPathRooted(modelId) ||
            modelId.StartsWith("./", StringComparison.Ordinal) ||
            modelId.StartsWith("../", StringComparison.Ordinal) ||
            modelId.StartsWith(".\\", StringComparison.Ordinal) ||
            modelId.StartsWith("..\\", StringComparison.Ordinal))
        {
            return CreateLocalModelInfo(modelId);
        }

        return CreateHuggingFaceModelInfo(modelId);
    }

    private static DetectorModelInfo CreateLocalModelInfo(string path)
    {
        var fullPath = Path.GetFullPath(path);
        var directory = Path.GetDirectoryName(fullPath) ?? ".";
        var fileName = Path.GetFileName(fullPath);

        return new DetectorModelInfo
        {
            Id = fullPath,
            AliasName = "local",
            DisplayName = $"Local: {fileName}",
            Architecture = "Unknown",
            ParametersM = 0,
            SizeBytes = File.Exists(fullPath) ? new FileInfo(fullPath).Length : 0,
            MapCoco = 0,
            InputSize = 640,
            RequiresNms = false,
            OnnxFile = fileName,
            Description = $"Local model from {directory}",
            License = "Unknown"
        };
    }

    private static DetectorModelInfo CreateHuggingFaceModelInfo(string modelId)
    {
        var parts = modelId.Split('/');
        var name = parts.Length > 1 ? parts[1] : modelId;

        // Detect architecture from model ID
        var architecture = name.ToLowerInvariant() switch
        {
            var n when n.Contains("rtdetr") || n.Contains("rt-detr") => "RT-DETR",
            var n when n.Contains("yolo") => "YOLO",
            var n when n.Contains("efficientdet") => "EfficientDet",
            var n when n.Contains("detr") => "DETR",
            _ => "Unknown"
        };

        var requiresNms = architecture is not ("RT-DETR" or "DETR");
        var isPose = name.Contains("pose", StringComparison.OrdinalIgnoreCase);

        return new DetectorModelInfo
        {
            Id = modelId,
            AliasName = modelId,
            DisplayName = name,
            Architecture = architecture,
            ParametersM = 0,
            SizeBytes = 0,
            MapCoco = 0,
            InputSize = 640,
            // A pose model detects one thing; the eighty COCO names never applied to it.
            ClassLabels = isPose ? ["person"] : CocoLabels.Labels,
            RequiresNms = requiresNms,
            NumKeypoints = isPose ? 17 : 0,
            OnnxFile = "model.onnx",
            Description = $"HuggingFace model: {modelId}",
            License = "Unknown"
        };
    }
}
