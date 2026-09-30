using LMSupply.Download;
using LMSupply.Exceptions;
using LMSupply.Transcriber.Core;
using LMSupply.Transcriber.Models;

namespace LMSupply.Transcriber;

/// <summary>
/// Main entry point for loading and using speech-to-text transcription models.
/// </summary>
public static class LocalTranscriber
{
    /// <summary>
    /// Gets the model registry for the Transcriber domain.
    /// Provides access to model resolution, alias management, and model enumeration.
    /// </summary>
    public static IModelRegistry<TranscriberModelInfo> Registry => TranscriberModelRegistry.Default;

    /// <summary>
    /// Shared pool for named model management. Supports GetOrLoadAsync / UnloadAsync by model ID.
    /// </summary>
    public static LMSupply.Pool.ModelPool<ITranscriberModel, TranscriberOptions> Pool { get; }
        = new(new Pool.TranscriberLoader());

    /// <summary>
    /// Loads a transcription model by name or path.
    /// </summary>
    /// <param name="modelIdOrPath">
    /// Either a model alias (e.g., "default", "fast", "quality", "large"),
    /// a HuggingFace model ID (e.g., "openai/whisper-base"),
    /// or a local path to ONNX model files.
    /// </param>
    /// <param name="options">Optional configuration options.</param>
    /// <param name="progress">Optional progress reporting for downloads.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>A loaded transcriber ready for inference.</returns>
    public static async Task<ITranscriberModel> LoadAsync(
        string modelIdOrPath,
        TranscriberOptions? options = null,
        IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        options ??= new TranscriberOptions();
        PrepareOptions(modelIdOrPath, options);

        var transcriber = CreateModel(options, progress);
        try
        {
            // Eagerly initialize and warm up the model (the download, if any, reports through progress)
            await transcriber.WarmupAsync(cancellationToken);

            if (options.PreloadDiarization && transcriber is Diarization.IDiarizationPreload preload)
                await preload.PreloadDiarizationAsync(progress, cancellationToken);
        }
        catch
        {
            await transcriber.DisposeAsync();
            throw;
        }

        return transcriber;
    }

    // Shared by the load and the size query, so that "large:fp16" resolves the same way in both.
    private static void PrepareOptions(string modelIdOrPath, TranscriberOptions options)
    {
        // An unsupported provider is refused here, before any model resolution or download (0.67.1).
        ExecutionProviderSupport.ThrowIfUnsupported(options.Provider);

        // Parse variant qualifier (e.g., "large:fp16" → modelId="large", hint="fp16")
        var (baseId, qualifier) = LMSupplyOptionsBase.SplitQualifier(modelIdOrPath);
        options.ModelId = baseId;
        options.QuantizationHint ??= qualifier;
    }

    /// <summary>
    /// Bytes <see cref="LoadAsync(string, TranscriberOptions?, IProgress{DownloadProgress}?, CancellationToken)"/> would
    /// download for the same arguments into an empty cache — the files that load picks (the quantization follows
    /// <see cref="LMSupplyOptionsBase.QuantizationHint"/>, or without one this machine's hardware tier, exactly as the
    /// load decides), plus the speaker-diarization pair when <see cref="TranscriberOptions.PreloadDiarization"/> is set.
    /// For a consent screen or a disk budget ahead of an install step; <see cref="TranscriberModelInfo.SizeBytes"/> is the
    /// full-precision export's size and is not what a load downloads.
    /// </summary>
    /// <remarks>
    /// Reads the repository listing (one request, cached for a day and reused by the load that follows); downloads and
    /// loads nothing. The figure is the whole download whatever the cache already holds — <see cref="IsDiarizationDownloadedAsync"/>
    /// and the model cache answer what is present. A model on local disk downloads nothing, so it adds 0. With
    /// <see cref="TranscriberOptions.DisableAutoDownload"/> the listing comes from the cache only, as the load's does.
    /// </remarks>
    /// <param name="modelIdOrPath">A model alias, HuggingFace id or local path, as for <c>LoadAsync</c>; a <c>:variant</c> qualifier is honoured.</param>
    /// <param name="options">The options the load will use; not modified.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <exception cref="Exceptions.ModelNotFoundException">The repository does not exist, or (downloads disabled) was never listed into this cache.</exception>
    public static async Task<long> GetDownloadSizeBytesAsync(
        string modelIdOrPath,
        TranscriberOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        options = options?.Clone() ?? new TranscriberOptions();
        PrepareOptions(modelIdOrPath, options);

        var plan = await PlanDownloadAsync(options, cancellationToken);
        return (plan?.TotalBytes ?? 0) + (options.PreloadDiarization ? DiarizationDownloadSizeBytes : 0);
    }

    // The files a load with these (prepared) options fetches — null for a model on local disk. Shared by the size query
    // and the cache check, so that both pick the files the load picks.
    private static Task<DownloadPlan?> PlanDownloadAsync(TranscriberOptions options, CancellationToken cancellationToken)
        => Registry.TryResolve(options.ModelId, out var info) && info is not null
           && TranscriberArchitectures.IsParakeetTdt(info.Architecture)
            ? ParakeetTdtTranscriberModel.PlanDownloadAsync(options, info, cancellationToken)
            : Directory.Exists(options.ModelId)
                ? Task.FromResult<DownloadPlan?>(null)
                : OnnxTranscriberModel.PlanDownloadAsync(options, Registry.Resolve(options.ModelId), cancellationToken);

    /// <summary>
    /// Whether a load of <paramref name="modelIdOrPath"/> with <paramref name="options"/> would open cached files only —
    /// every file that load picks (the same alias, <c>:variant</c> qualifier and quantization as
    /// <see cref="LoadAsync(string, TranscriberOptions?, IProgress{DownloadProgress}?, CancellationToken)"/> and
    /// <see cref="GetDownloadSizeBytesAsync(string, TranscriberOptions?, CancellationToken)"/>) is in the cache at the length
    /// the repository lists, and none is a Git LFS pointer. Makes no network request and loads nothing.
    /// </summary>
    /// <remarks>
    /// The answer never errs toward <see langword="true"/>: a repository directory without its model files, a partial
    /// file, or a cache that never listed the repository answers <see langword="false"/>. A model on local disk answers
    /// <see langword="true"/> (the load downloads nothing). The speaker-diarization pair is not part of the answer, even
    /// with <see cref="TranscriberOptions.PreloadDiarization"/> — <see cref="IsDiarizationDownloadedAsync"/> answers for it.
    /// </remarks>
    /// <param name="modelIdOrPath">A model alias, HuggingFace id or local path, as for <c>LoadAsync</c>.</param>
    /// <param name="options">The options the load will use; not modified. <see cref="LMSupplyOptionsBase.CacheDirectory"/> is where to look.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    public static async Task<bool> IsModelDownloadedAsync(
        string modelIdOrPath,
        TranscriberOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(modelIdOrPath);
        options = options?.Clone() ?? new TranscriberOptions();
        PrepareOptions(modelIdOrPath, options);

        // Plan from the cached listing (or the download manifest) only — the check must not reach the network.
        options.DisableAutoDownload = true;

        DownloadPlan? plan;
        try
        {
            plan = await PlanDownloadAsync(options, cancellationToken);
        }
        catch (Exception ex) when (ex is ModelNotFoundException or ModelDownloadException)
        {
            // Never listed into this cache, or listed without a file (or its length) the load needs.
            return false;
        }

        if (plan is null)
            return true;

        // The files are opened from one snapshot directory: this library's own, or one another Hugging Face tool
        // wrote under the commit the revision points at.
        var cacheDir = options.CacheDirectory ?? CacheManager.GetDefaultCacheDirectory();
        return CacheManager.GetSnapshotDirectories(cacheDir, plan.RepoId, plan.Revision).Any(snapshotDir =>
            plan.Files.All(file =>
            {
                var path = Path.Combine(snapshotDir, file.Path.Replace('/', Path.DirectorySeparatorChar));
                return CacheManager.TryGetContentLength(path, out var length)
                       && length == file.SizeBytes
                       && !CacheManager.IsLfsPointerFile(path);
            }));
    }

    /// <inheritdoc cref="IsModelDownloadedAsync(string, TranscriberOptions?, CancellationToken)"/>
    /// <remarks>For <see cref="TranscriberOptions.ModelId"/> — the check that matches <see cref="LoadAsync(TranscriberOptions?, IProgress{DownloadProgress}?, CancellationToken)"/>.</remarks>
    public static Task<bool> IsModelDownloadedAsync(
        TranscriberOptions? options, CancellationToken cancellationToken = default)
        => IsModelDownloadedAsync(options?.ModelId ?? "default", options, cancellationToken);

    /// <summary>
    /// <see cref="GetDownloadSizeBytesAsync(string, TranscriberOptions?, CancellationToken)"/> for
    /// <see cref="TranscriberOptions.ModelId"/> — the size query that matches <see cref="LoadAsync(TranscriberOptions?, IProgress{DownloadProgress}?, CancellationToken)"/>.
    /// </summary>
    public static Task<long> GetDownloadSizeBytesAsync(
        TranscriberOptions? options, CancellationToken cancellationToken = default)
        => GetDownloadSizeBytesAsync(options?.ModelId ?? "default", options, cancellationToken);

    /// <summary>
    /// Picks the model family from the registry entry's <see cref="TranscriberModelInfo.Architecture"/> (or, for a local
    /// directory, from its <c>config.json</c>). Everything that is not a known Parakeet TDT export takes the Whisper path —
    /// which is what every previously supported id did.
    /// </summary>
    private static ITranscriberModel CreateModel(TranscriberOptions options, IProgress<DownloadProgress>? progress)
    {
        if (Registry.TryResolve(options.ModelId, out var info) && info is not null
            && TranscriberArchitectures.IsParakeetTdt(info.Architecture))
        {
            return new ParakeetTdtTranscriberModel(options, info, progress);
        }

        if (Directory.Exists(options.ModelId)
            && Internal.NemoConfigReader.ReadConfig(options.ModelId) is { IsTdt: true } config)
        {
            var template = DefaultModels.ParakeetTdt06BV3;
            return new ParakeetTdtTranscriberModel(options, new TranscriberModelInfo
            {
                Id = options.ModelId,
                AliasName = Path.GetFileName(options.ModelId.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar)),
                DisplayName = "Parakeet TDT (local)",
                Architecture = TranscriberArchitectures.ParakeetTdt,
                NumMelBins = config.FeaturesSize ?? template.NumMelBins,
                HiddenSize = template.HiddenSize,
                MaxDurationSeconds = template.MaxDurationSeconds,
                EncoderFile = template.EncoderFile,
                DecoderFile = template.DecoderFile,
                IsMultilingual = true,
                SupportedLanguages = template.SupportedLanguages,
                License = template.License
            }, progress);
        }

        return new OnnxTranscriberModel(options, progress);
    }

    /// <summary>
    /// Loads the default transcription model (Whisper Base).
    /// </summary>
    /// <param name="options">Optional configuration options.</param>
    /// <param name="progress">Optional progress reporting for downloads.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>A loaded transcriber ready for inference.</returns>
    public static Task<ITranscriberModel> LoadAsync(
        TranscriberOptions? options = null,
        IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        return LoadAsync(options?.ModelId ?? "default", options, progress, cancellationToken);
    }

    /// <summary>
    /// Whether the speaker-diarization models that <see cref="TranscribeOptions.Diarize"/> uses are in the cache, complete —
    /// that is, whether a diarized call (or a load with <see cref="TranscriberOptions.PreloadDiarization"/>) would open them
    /// without a download. Makes no network request and loads nothing.
    /// </summary>
    /// <param name="options">Only <see cref="LMSupplyOptionsBase.CacheDirectory"/> is read; null means the default cache.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    public static Task<bool> IsDiarizationDownloadedAsync(
        TranscriberOptions? options = null, CancellationToken cancellationToken = default)
        => Diarization.SpeakerDiarizer.IsCachedAsync(
            options?.CacheDirectory ?? CacheManager.GetDefaultCacheDirectory(), cancellationToken);

    /// <summary>
    /// Bytes the speaker-diarization models take when downloaded (pyannote segmentation-3.0 + WeSpeaker ResNet34,
    /// about 32.5 MB) — for a consent screen that lists what a load will fetch.
    /// </summary>
    public static long DiarizationDownloadSizeBytes => Diarization.SpeakerDiarizer.DownloadSizeBytes;

    /// <summary>
    /// Gets a list of pre-configured model aliases available for use.
    /// </summary>
    /// <returns>Available model aliases.</returns>
    public static IEnumerable<string> GetAvailableModels()
    {
        return TranscriberModelRegistry.Default.GetAliases().Select(a => a.Name);
    }

    /// <summary>
    /// Gets all registered model information.
    /// </summary>
    /// <returns>Collection of model information.</returns>
    public static IEnumerable<TranscriberModelInfo> GetAllModels()
    {
        return TranscriberModelRegistry.Default.GetAvailableModels();
    }
}
