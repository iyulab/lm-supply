using LMSupply.Segmenter.Core;
using LMSupply.Segmenter.Models;

namespace LMSupply.Segmenter;

/// <summary>
/// Main entry point for loading and using image segmentation models.
/// </summary>
public static class LocalSegmenter
{
    /// <summary>
    /// Loads an image segmentation model by name or path.
    /// </summary>
    /// <param name="modelIdOrPath">
    /// Either a model alias (e.g., "default", "quality", "fast"),
    /// a HuggingFace model ID (e.g., "nvidia/segformer-b0-finetuned-ade-512-512"),
    /// or a local path to an ONNX model file.
    /// </param>
    /// <param name="options">Optional configuration options.</param>
    /// <param name="progress">Optional progress reporting for downloads.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>A loaded segmenter ready for inference.</returns>
    public static async Task<ISegmenterModel> LoadAsync(
        string modelIdOrPath,
        SegmenterOptions? options = null,
        IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        options ??= new SegmenterOptions();
        options.ModelId = modelIdOrPath;

        var segmenter = new OnnxSegmenterModel(options);

        // Eagerly initialize and warm up the model
        await segmenter.WarmupAsync(cancellationToken);

        return segmenter;
    }

    /// <summary>
    /// Loads the default image segmentation model.
    /// </summary>
    /// <param name="options">Optional configuration options.</param>
    /// <param name="progress">Optional progress reporting for downloads.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>A loaded segmenter ready for inference.</returns>
    public static Task<ISegmenterModel> LoadAsync(
        SegmenterOptions? options = null,
        IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        return LoadAsync("default", options, progress, cancellationToken);
    }

    /// <summary>
    /// Gets a list of pre-configured model aliases available for use.
    /// </summary>
    /// <returns>Available model aliases.</returns>
    public static IEnumerable<string> GetAvailableModels()
    {
        return SegmenterModelRegistry.Default.GetAliases();
    }

    /// <summary>
    /// Gets all registered model information.
    /// </summary>
    /// <returns>Collection of model information.</returns>
    public static IEnumerable<SegmenterModelInfo> GetAllModels()
    {
        return SegmenterModelRegistry.Default.GetAll();
    }

    /// <summary>
    /// Gets the ADE20K class labels.
    /// </summary>
    public static IReadOnlyList<string> Ade20kClassLabels => Ade20kLabels.Labels;
}
