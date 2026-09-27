using LMSupply.Segmenter.Core;
using LMSupply.Segmenter.Interactive;
using LMSupply.Segmenter.Models;

namespace LMSupply.Segmenter;

/// <summary>
/// Main entry point for loading and using image segmentation models.
/// </summary>
public static class LocalSegmenter
{
    /// <summary>
    /// Gets the model registry for the Segmenter domain.
    /// Provides access to model resolution, alias management, and model enumeration.
    /// </summary>
    public static IModelRegistry<SegmenterModelInfo> Registry => SegmenterModelRegistry.Default;

    /// <summary>
    /// Shared pool for named model management. Supports GetOrLoadAsync / UnloadAsync by model ID.
    /// </summary>
    public static LMSupply.Pool.ModelPool<ISegmenterModel, SegmenterOptions> Pool { get; }
        = new(new Pool.SegmenterLoader());

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
        // An unsupported provider is refused here, before any model resolution or download (0.67.1).
        ExecutionProviderSupport.ThrowIfUnsupported(options.Provider);

        // Parse variant qualifier (e.g., "default:fp16" → modelId="default", hint="fp16")
        var (baseId, qualifier) = LMSupplyOptionsBase.SplitQualifier(modelIdOrPath);
        options.ModelId = baseId;
        options.QuantizationHint ??= qualifier;

        if (SegmenterModelRegistry.Default.TryResolve(options.ModelId, out var resolved) && resolved is { IsInteractive: true })
        {
            throw new ArgumentException(
                $"'{modelIdOrPath}' is a prompt-based (interactive) model; load it with LocalSegmenter.LoadInteractiveAsync.",
                nameof(modelIdOrPath));
        }

        var segmenter = new OnnxSegmenterModel(options, progress);

        // Eagerly initialize and warm up the model
        await segmenter.WarmupAsync(cancellationToken);

        return segmenter;
    }

    /// <summary>
    /// Loads a prompt-based (interactive) segmentation model — segment what a point or box points at — and warms it up.
    /// </summary>
    /// <param name="modelIdOrAlias">An interactive model alias; <c>"interactive"</c> (MobileSAM) is the one registered.</param>
    /// <param name="options">Optional configuration options (provider, cache directory, downloads).</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>A segmenter whose <see cref="IInteractiveSegmenter.CreateSessionAsync(string, CancellationToken)"/> encodes an image once for many prompts.</returns>
    /// <exception cref="ArgumentException">The id does not name an interactive model.</exception>
    public static async Task<IInteractiveSegmenter> LoadInteractiveAsync(
        string modelIdOrAlias = "interactive",
        SegmenterOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        options ??= new SegmenterOptions();
        ExecutionProviderSupport.ThrowIfUnsupported(options.Provider);

        var modelInfo = SegmenterModelRegistry.Default.Resolve(modelIdOrAlias);
        if (!modelInfo.IsInteractive)
        {
            throw new ArgumentException(
                $"'{modelIdOrAlias}' is not an interactive model; load it with LocalSegmenter.LoadAsync.", nameof(modelIdOrAlias));
        }

        options.ModelId = modelInfo.Id;
        var segmenter = new MobileSamModel(options, modelInfo);
        try
        {
            await segmenter.WarmupAsync(cancellationToken);
        }
        catch
        {
            await segmenter.DisposeAsync();
            throw;
        }

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
        return LoadAsync(options?.ModelId ?? "default", options, progress, cancellationToken);
    }

    /// <summary>
    /// Gets a list of pre-configured model aliases available for use.
    /// </summary>
    /// <returns>Available model aliases.</returns>
    public static IEnumerable<string> GetAvailableModels()
    {
        return SegmenterModelRegistry.Default.GetAliases().Select(a => a.Name);
    }

    /// <summary>
    /// Gets all registered model information.
    /// </summary>
    /// <returns>Collection of model information.</returns>
    public static IEnumerable<SegmenterModelInfo> GetAllModels()
    {
        return SegmenterModelRegistry.Default.GetAvailableModels();
    }

    /// <summary>
    /// Gets the ADE20K class labels.
    /// </summary>
    public static IReadOnlyList<string> Ade20kClassLabels => Ade20kLabels.Labels;
}
