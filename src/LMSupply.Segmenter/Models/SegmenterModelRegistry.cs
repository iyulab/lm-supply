using System.Diagnostics;
using LMSupply.Hardware;

namespace LMSupply.Segmenter.Models;

/// <summary>
/// Model registry for the Segmenter domain.
/// </summary>
public sealed class SegmenterModelRegistry : ModelRegistryBase<SegmenterModelInfo>
{
    /// <summary>
    /// Auto-selection candidates sorted by size descending (largest first).
    /// </summary>
    private static readonly SegmenterModelInfo[] AutoCandidates =
    [
        DefaultModels.MaskFormerResNet50,  // 44M params, 178MB
        DefaultModels.SegFormerB0,          // 3.7M params, 15MB
        DefaultModels.MediaPipeSelfie,      // 0.7M params, 3MB
    ];

    /// <summary>
    /// Gets the default registry instance with built-in models.
    /// </summary>
    public static SegmenterModelRegistry Default { get; } = CreateDefault();

    /// <summary>
    /// Builds the default registry: built-in models plus the user's
    /// ~/.lmsupply/aliases.json "segmenter" section (fail-soft).
    /// </summary>
    internal static SegmenterModelRegistry CreateDefault()
        => AliasConfiguration.ApplyDomain(new SegmenterModelRegistry(DefaultModels.All), AliasConfiguration.Domains.Segmenter);

    /// <summary>
    /// Initializes a new registry with the specified system models.
    /// </summary>
    /// <param name="systemModels">Models to register as system defaults.</param>
    public SegmenterModelRegistry(IEnumerable<SegmenterModelInfo> systemModels)
        : base(systemModels) { }

    /// <summary>
    /// Gets the optimal model based on available VRAM.
    /// Selects the largest model that fits in available GPU memory,
    /// falling back to the smallest model if none fit.
    /// </summary>
    protected override SegmenterModelInfo GetAutoModel()
    {
        var gpu = HardwareProfile.Current.GpuInfo;
        var availableVram = VramBudget.GetAvailableBytes(gpu);
        Trace.TraceInformation($"[SegmenterModelRegistry] Auto-selecting model for VRAM: {availableVram / (1024 * 1024)} MB");

        SegmenterModelInfo selected = AutoCandidates[^1]; // default to smallest

        foreach (var candidate in AutoCandidates)
        {
            var memInfo = (IModelMemoryInfo)candidate;
            var size = ModelMemoryEstimator.EstimateModelSizeBytes(
                memInfo.ParameterCount, memInfo.QuantizationType, memInfo.EstimatedSizeBytes);
            if (size <= availableVram)
            {
                selected = candidate;
                Trace.TraceInformation($"[SegmenterModelRegistry] Selected: {candidate.Id} ({size / (1024 * 1024)} MB)");
                break;
            }
        }

        return new SegmenterModelInfo
        {
            Id = selected.Id,
            AliasName = "auto",
            DisplayName = selected.DisplayName,
            Architecture = selected.Architecture,
            ParametersM = selected.ParametersM,
            SizeBytes = selected.SizeBytes,
            MIoU = selected.MIoU,
            InputSize = selected.InputSize,
            NumClasses = selected.NumClasses,
            OnnxFile = selected.OnnxFile,
            EncoderFile = selected.EncoderFile,
            DecoderFile = selected.DecoderFile,
            Dataset = selected.Dataset,
            Description = selected.Description,
            License = selected.License
        };
    }

    /// <summary>
    /// Creates a fallback model info for unknown model IDs (HuggingFace repos or local paths).
    /// </summary>
    protected override SegmenterModelInfo CreateFallbackModelInfo(string modelId)
    {
        Trace.TraceInformation($"[SegmenterModelRegistry] Creating fallback model info for: {modelId}");

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

    private static SegmenterModelInfo CreateLocalModelInfo(string path)
    {
        var fullPath = Path.GetFullPath(path);
        var directory = Path.GetDirectoryName(fullPath) ?? ".";
        var fileName = Path.GetFileName(fullPath);

        return new SegmenterModelInfo
        {
            Id = fullPath,
            AliasName = "local",
            DisplayName = $"Local: {fileName}",
            Architecture = "Unknown",
            ParametersM = 0,
            SizeBytes = File.Exists(fullPath) ? new FileInfo(fullPath).Length : 0,
            MIoU = 0,
            InputSize = 512,
            NumClasses = 150,
            OnnxFile = fileName,
            Dataset = "Unknown",
            Description = $"Local model from {directory}"
        };
    }

    private static SegmenterModelInfo CreateHuggingFaceModelInfo(string modelId)
    {
        var parts = modelId.Split('/');
        var name = parts.Length > 1 ? parts[1] : modelId;

        // Detect architecture from model ID
        var architecture = name.ToLowerInvariant() switch
        {
            var n when n.Contains("segformer") => "SegFormer",
            var n when n.Contains("deeplabv3") || n.Contains("deeplab") => "DeepLabV3+",
            var n when n.Contains("mask2former") => "Mask2Former",
            var n when n.Contains("sam") => "SAM",
            _ => "Unknown"
        };

        // Try to detect number of classes from model name
        var numClasses = name.ToLowerInvariant() switch
        {
            var n when n.Contains("ade") => 150,
            var n when n.Contains("cityscapes") => 19,
            var n when n.Contains("coco") => 171,
            _ => 150
        };

        // Detect input size from model name
        var inputSize = 512;
        if (name.Contains("640"))
            inputSize = 640;
        else if (name.Contains("1024"))
            inputSize = 1024;

        return new SegmenterModelInfo
        {
            Id = modelId,
            AliasName = modelId,
            DisplayName = name,
            Architecture = architecture,
            ParametersM = 0,
            SizeBytes = 0,
            MIoU = 0,
            InputSize = inputSize,
            NumClasses = numClasses,
            OnnxFile = "model.onnx",
            Dataset = "Unknown",
            Description = $"HuggingFace model: {modelId}"
        };
    }
}
