using LMSupply.Synthesizer.Core;
using LMSupply.Synthesizer.Models;
using LMSupply.Download;
using LMSupply.Exceptions;

namespace LMSupply.Synthesizer;

/// <summary>
/// Main entry point for loading and using text-to-speech synthesis models.
/// </summary>
/// <remarks>
/// A voice is run by the contract its config declares. Text for an English or Korean voice is turned into its phonemes by
/// the package's built-in phonemizer for that language; for a voice in another language it needs a caller's
/// <see cref="SynthesizerOptions.Phonemizer"/>, and throws <see cref="NotSupportedException"/> without one. Phonemes
/// (<see cref="ISynthesizerModel.SynthesizePhonemesAsync"/>) work for every voice. See the package README.
/// </remarks>
public static class LocalSynthesizer
{
    /// <summary>
    /// Gets the model registry for the Synthesizer domain.
    /// Provides access to model resolution, alias management, and model enumeration.
    /// </summary>
    public static IModelRegistry<SynthesizerModelInfo> Registry => SynthesizerModelRegistry.Default;

    /// <summary>
    /// Shared pool for named model management. Supports GetOrLoadAsync / UnloadAsync by model ID.
    /// </summary>
    public static LMSupply.Pool.ModelPool<ISynthesizerModel, SynthesizerOptions> Pool { get; }
        = new(new Pool.SynthesizerLoader());

    /// <summary>
    /// Loads a TTS model by name or path.
    /// </summary>
    /// <param name="modelIdOrPath">
    /// Either a model alias (e.g., "default", "fast", "quality"), a catalog voice's
    /// <see cref="SynthesizerModelInfo.CatalogKey"/> (e.g., "rhasspy/piper-voices:en/en_US/ljspeech/medium"),
    /// a HuggingFace repository ID, or a local path to ONNX model files. The repository ID of the catalog's
    /// voices names no single voice and loads the first of them, the default one.
    /// </param>
    /// <param name="options">Optional configuration options.</param>
    /// <param name="progress">Optional progress reporting for downloads.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>A loaded synthesizer ready for inference.</returns>
    public static async Task<ISynthesizerModel> LoadAsync(
        string modelIdOrPath,
        SynthesizerOptions? options = null,
        IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        progress = LoadProgress.Begin(progress);
        options = options?.Clone() ?? new SynthesizerOptions();
        // An unsupported provider is refused here, before any model resolution or download (0.67.1).
        ExecutionProviderSupport.ThrowIfUnsupported(options.Provider);

        // Parse variant qualifier (e.g., "default:fp16" → modelId="default", hint="fp16")
        var (baseId, qualifier) = LMSupplyOptionsBase.SplitQualifier(modelIdOrPath);
        options.ModelId = baseId;
        options.QuantizationHint ??= qualifier;

        var synthesizer = new OnnxSynthesizerModel(options, progress);

        // Eagerly initialize and warm up the model
        await synthesizer.WarmupAsync(cancellationToken);

        return synthesizer;
    }

    /// <summary>
    /// Loads the default TTS model (English US LJSpeech).
    /// </summary>
    /// <param name="options">Optional configuration options.</param>
    /// <param name="progress">Optional progress reporting for downloads.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>A loaded synthesizer ready for inference.</returns>
    public static Task<ISynthesizerModel> LoadAsync(
        SynthesizerOptions? options = null,
        IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        progress = LoadProgress.Begin(progress);
        return LoadAsync(options?.ModelId ?? "default", options, progress, cancellationToken);
    }

    /// <summary>
    /// Bytes <see cref="LoadAsync(string, SynthesizerOptions?, IProgress{DownloadProgress}?, CancellationToken)"/> would
    /// download for the same id and options into an empty cache: the voice's model and config, at the lengths the
    /// repository lists. 0 for a voice on local disk.
    /// </summary>
    /// <remarks>
    /// Reads the repository listing (cached for the download that follows); downloads and loads nothing. The native ONNX
    /// Runtime a first load also provisions is not counted — <see cref="SynthesizerArtifacts"/> counts it. With
    /// <see cref="SynthesizerOptions.DisableAutoDownload"/> the listing comes from the cache only, as the load's does.
    /// </remarks>
    /// <exception cref="ModelNotFoundException">The repository does not exist, or (downloads disabled) it was never listed into this cache.</exception>
    /// <exception cref="ModelDownloadException">The voice's model is not in the repository.</exception>
    public static async Task<long> GetDownloadSizeBytesAsync(
        string modelIdOrPath, SynthesizerOptions? options = null, CancellationToken cancellationToken = default)
    {
        var (info, resolved) = Prepare(modelIdOrPath, options);
        return (await OnnxSynthesizerModel.PlanAsync(info, resolved, cancellationToken))?.TotalBytes ?? 0;
    }

    /// <summary>
    /// Bytes a load with the same id and options would still download now: the files of
    /// <see cref="GetDownloadSizeBytesAsync"/> the cache does not hold at the length the repository lists; 0 when cached or
    /// on local disk. A partly downloaded file counts in full. Runtimes are not counted.
    /// </summary>
    public static async Task<long> GetRemainingDownloadBytesAsync(
        string modelIdOrPath, SynthesizerOptions? options = null, CancellationToken cancellationToken = default)
    {
        var (info, resolved) = Prepare(modelIdOrPath, options);
        return await OnnxSynthesizerModel.PlanAsync(info, resolved, cancellationToken) is { } plan
            ? plan.GetRemainingBytes(OnnxSynthesizerModel.CacheDirectory(resolved))
            : 0;
    }

    /// <summary>
    /// Whether a load with the same id and options would open cached files only: the voice's model and config are in the
    /// cache at the lengths the repository lists, or the voice is on local disk. Makes no network request; never errs toward
    /// <see langword="true"/>.
    /// </summary>
    public static async Task<bool> IsModelDownloadedAsync(
        string modelIdOrPath, SynthesizerOptions? options = null, CancellationToken cancellationToken = default)
    {
        var (info, resolved) = Prepare(modelIdOrPath, options);
        // Plan from the cached listing only: the check must not reach the network.
        resolved.DisableAutoDownload = true;
        try
        {
            return await OnnxSynthesizerModel.PlanAsync(info, resolved, cancellationToken) is not { } plan
                   || plan.GetRemainingBytes(OnnxSynthesizerModel.CacheDirectory(resolved)) == 0;
        }
        catch (Exception ex) when (ex is ModelNotFoundException or ModelDownloadException)
        {
            return false;
        }
    }

    /// <summary>
    /// Downloads the voice's model and config, without loading anything, so a load with downloads disabled opens cached
    /// files only. Cached files are not fetched again; a voice on local disk downloads nothing. The load downloads through
    /// the same path.
    /// </summary>
    /// <exception cref="ModelNotFoundException">The voice is unknown, or (downloads disabled) not cached.</exception>
    public static async Task DownloadModelAsync(
        string modelIdOrPath, SynthesizerOptions? options = null, IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        var (info, resolved) = Prepare(modelIdOrPath, options);
        await OnnxSynthesizerModel.DownloadAsync(info, resolved, progress, cancellationToken);
    }

    /// <summary>
    /// Deletes what LMSupply downloaded of this voice from the cache, so the next load downloads it again. The voices of
    /// the catalog share one repository, each in a folder of its own: only this voice's files are deleted, and nothing
    /// another tool put in the shared cache. A voice on local disk is kept.
    /// </summary>
    /// <returns>Whether anything was deleted.</returns>
    /// <exception cref="IOException">A file is open, for example by a loaded synthesizer (Windows).</exception>
    public static bool DeleteModel(string modelIdOrPath, SynthesizerOptions? options = null)
    {
        var (info, resolved) = Prepare(modelIdOrPath, options);
        if (OnnxSynthesizerModel.LocalDirectory(info) is not null || File.Exists(info.Id) || !info.Id.Contains('/'))
            return false;

        return CacheManager.DeleteModelFiles(
            OnnxSynthesizerModel.CacheDirectory(resolved), info.Id, OnnxSynthesizerModel.SnapshotPaths(info));
    }

    // The registry entry a load resolves, with the options it runs under
    private static (SynthesizerModelInfo Info, SynthesizerOptions Options) Prepare(string modelIdOrPath, SynthesizerOptions? options)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(modelIdOrPath);
        options = options?.Clone() ?? new SynthesizerOptions();
        ExecutionProviderSupport.ThrowIfUnsupported(options.Provider);
        var (baseId, qualifier) = LMSupplyOptionsBase.SplitQualifier(modelIdOrPath);
        options.ModelId = baseId;
        options.QuantizationHint ??= qualifier;
        return (SynthesizerModelRegistry.Default.Resolve(baseId), options);
    }

    /// <summary>
    /// Gets a list of pre-configured model aliases available for use.
    /// </summary>
    /// <returns>Available model aliases.</returns>
    public static IEnumerable<string> GetAvailableModels()
    {
        return SynthesizerModelRegistry.Default.GetAliases().Select(a => a.Name);
    }

    /// <summary>
    /// Gets every voice of the catalog, once each.
    /// </summary>
    /// <returns>Collection of model information.</returns>
    public static IEnumerable<SynthesizerModelInfo> GetAllModels()
    {
        return SynthesizerModelRegistry.Default.GetAvailableModels();
    }
}
