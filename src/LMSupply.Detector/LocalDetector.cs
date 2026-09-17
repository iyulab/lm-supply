using LMSupply.Detector.Core;
using LMSupply.Detector.Models;

namespace LMSupply.Detector;

/// <summary>
/// Main entry point for loading and using object detection models.
/// </summary>
public static class LocalDetector
{
    /// <summary>
    /// Gets the model registry for the Detector domain.
    /// Provides access to model resolution, alias management, and model enumeration.
    /// </summary>
    public static IModelRegistry<DetectorModelInfo> Registry => DetectorModelRegistry.Default;

    /// <summary>
    /// Shared pool for named model management. Supports GetOrLoadAsync / UnloadAsync by model ID.
    /// </summary>
    public static LMSupply.Pool.ModelPool<IDetectorModel, DetectorOptions> Pool { get; }
        = new(new Pool.DetectorLoader());

    /// <summary>
    /// Loads an object detection model by name or path.
    /// </summary>
    /// <param name="modelIdOrPath">
    /// Either a model alias (e.g., "default", "quality", "fast"),
    /// a HuggingFace model ID (e.g., "PekingU/rtdetr_r18vd"),
    /// or a local path to an ONNX model file.
    /// </param>
    /// <param name="options">Optional configuration options.</param>
    /// <param name="progress">Optional progress reporting for downloads.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>A loaded detector ready for inference.</returns>
    public static async Task<IDetectorModel> LoadAsync(
        string modelIdOrPath,
        DetectorOptions? options = null,
        IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        options ??= new DetectorOptions();
        // An unsupported provider is refused here, before any model resolution or download (0.67.1).
        ExecutionProviderSupport.ThrowIfUnsupported(options.Provider);

        // Parse variant qualifier (e.g., "default:fp16" → modelId="default", hint="fp16")
        var (baseId, qualifier) = LMSupplyOptionsBase.SplitQualifier(modelIdOrPath);
        options.ModelId = baseId;
        options.QuantizationHint ??= qualifier;

        var detector = new OnnxDetectorModel(options);

        // Eagerly initialize and warm up the model
        await detector.WarmupAsync(cancellationToken);

        return detector;
    }

    /// <summary>
    /// Loads the default object detection model.
    /// </summary>
    /// <param name="options">Optional configuration options.</param>
    /// <param name="progress">Optional progress reporting for downloads.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>A loaded detector ready for inference.</returns>
    public static Task<IDetectorModel> LoadAsync(
        DetectorOptions? options = null,
        IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        return LoadAsync(options?.ModelId ?? "default", options, progress, cancellationToken);
    }

    /// <summary>
    /// Gets a list of pre-configured model aliases available for use.
    /// </summary>
    /// <returns>Available model aliases.</returns>
    public static IEnumerable<string> GetAvailableModels()
    {
        return DetectorModelRegistry.Default.GetAliases().Select(a => a.Name);
    }

    /// <summary>
    /// Gets all registered model information.
    /// </summary>
    /// <returns>Collection of model information.</returns>
    public static IEnumerable<DetectorModelInfo> GetAllModels()
    {
        return DetectorModelRegistry.Default.GetAvailableModels();
    }

    /// <summary>
    /// Gets the COCO class labels.
    /// </summary>
    public static IReadOnlyList<string> CocoClassLabels => CocoLabels.Labels;
}
