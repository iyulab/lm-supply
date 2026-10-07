using LMSupply.Captioner.Inference;
using LMSupply.Captioner.Models;
using LMSupply.Core.Download;
using LMSupply.Download;
using LMSupply.Exceptions;

namespace LMSupply.Captioner;

/// <summary>
/// Main entry point for loading and using image captioning models.
/// </summary>
public static class LocalCaptioner
{
    /// <summary>
    /// Gets the model registry for the Captioner domain.
    /// Provides access to model resolution, alias management, and model enumeration.
    /// </summary>
    public static IModelRegistry<ModelInfo> Registry => CaptionerModelRegistry.Default;

    /// <summary>
    /// Shared pool for named model management. Supports GetOrLoadAsync / UnloadAsync by model ID.
    /// </summary>
    public static LMSupply.Pool.ModelPool<ICaptionerModel, CaptionerOptions> Pool { get; }
        = new(new Pool.CaptionerLoader());

    /// <summary>
    /// Loads a captioning model by name or path.
    /// </summary>
    /// <param name="modelIdOrPath">
    /// Either a model ID (e.g., "default", "quality") for auto-download, a HuggingFace repository id,
    /// or a local path to a model directory. A <c>:variant</c> qualifier (e.g. <c>"quality:fp16"</c>) picks the
    /// quantized files of a model published in several (see <see cref="ModelInfo.HasQuantizationVariants"/>).
    /// </param>
    /// <param name="options">
    /// Optional configuration options. With <see cref="CaptionerOptions.DisableAutoDownload"/> set, a model
    /// id is served from the cache only: a file that is not there throws <see cref="ModelNotFoundException"/>
    /// and nothing is downloaded or written. Not modified: the load works on a copy.
    /// </param>
    /// <param name="progress">Optional progress reporting for downloads.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>A loaded captioning model ready for inference.</returns>
    /// <exception cref="NotSupportedException">
    /// An option the model cannot honour is set: a <see cref="CaptionerOptions.Detail"/> other than
    /// <see cref="CaptionDetail.Brief"/> on a model that captions at one level only, or a
    /// <see cref="CaptionerOptions.Prompt"/> on a model whose prompt selects a task (Florence-2). Checked before
    /// anything is downloaded.
    /// </exception>
    public static async Task<ICaptionerModel> LoadAsync(
        string modelIdOrPath,
        CaptionerOptions? options = null,
        IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(modelIdOrPath);
        // A copy: the qualifier below must not land in the caller's instance (and so in its next load).
        options = options?.Clone() ?? new CaptionerOptions();
        // An unsupported provider is refused here, before any model resolution or download (0.67.1).
        ExecutionProviderSupport.ThrowIfUnsupported(options.Provider);

        // Parse variant qualifier (e.g., "default:fp16" → modelId="default", hint="fp16")
        var (baseId, qualifier) = LMSupplyOptionsBase.SplitQualifier(modelIdOrPath);
        modelIdOrPath = baseId;
        options.QuantizationHint ??= qualifier;

        ModelInfo? modelInfo = null;
        string modelDir;
        string? tokenizerDir = null; // Separate tokenizer directory for HuggingFace repos with subfolders
        var variantSuffix = "";

        // Check if it's a local directory path
        if (Directory.Exists(modelIdOrPath))
        {
            modelDir = modelIdOrPath;

            // Try to infer model info from directory contents
            if (!TryInferModelInfo(modelDir, modelDir, options, out modelInfo, out variantSuffix))
            {
                throw new ModelNotFoundException(
                    $"Could not determine model type from directory: {modelDir}",
                    modelIdOrPath);
            }

            EnsureOptionsSupported(modelInfo!, options);
        }
        // Check if it's a known model alias (only system/user aliases, not HF fallbacks)
        else if (TryResolveKnownModel(modelIdOrPath, out modelInfo))
        {
            EnsureOptionsSupported(modelInfo!, options);
            variantSuffix = SelectVariantSuffix(modelInfo!, options);

            // Download model from HuggingFace
            var cacheDir = options.CacheDirectory ?? CacheManager.GetDefaultCacheDirectory();
            using var downloader = new HuggingFaceDownloader(cacheDir, localFilesOnly: options.DisableAutoDownload);

            modelDir = await downloader.DownloadModelAsync(
                modelInfo!.RepoId,
                files: GetRequiredFiles(modelInfo, variantSuffix),
                subfolder: modelInfo.Subfolder,
                progress: progress,
                cancellationToken: cancellationToken).ConfigureAwait(false);
        }
        // Check if it's a HuggingFace repo ID
        else if (modelIdOrPath.Contains('/'))
        {
            var cacheDir = options.CacheDirectory ?? CacheManager.GetDefaultCacheDirectory();
            using var downloader = new HuggingFaceDownloader(cacheDir, localFilesOnly: options.DisableAutoDownload);

            var (downloadedDir, discovery) = await downloader.DownloadWithDiscoveryAsync(
                modelIdOrPath,
                preferences: DiscoveryPreferences(options),
                progress: progress,
                cancellationToken: cancellationToken).ConfigureAwait(false);

            // Use discovery result to get correct ONNX directory (handles subfolder structures)
            var onnxDir = discovery.GetOnnxDirectory(downloadedDir);

            // Try to infer model info using the ONNX directory
            if (!TryInferModelInfo(onnxDir, downloadedDir, options, out modelInfo, out variantSuffix))
            {
                throw new ModelNotFoundException(
                    $"Could not determine model type for: {modelIdOrPath}. " +
                    $"Discovered ONNX files: [{string.Join(", ", discovery.OnnxFiles)}]. " +
                    "Use a known model ID (e.g., 'default', 'quality') or ensure the model follows a supported format.",
                    modelIdOrPath);
            }

            EnsureOptionsSupported(modelInfo!, options);

            // ONNX files are in onnxDir, tokenizer files are in base downloadedDir
            modelDir = onnxDir;
            tokenizerDir = downloadedDir;
        }
        else
        {
            throw new ModelNotFoundException(
                $"Unknown model '{modelIdOrPath}'. Use a known model ID (e.g., 'default', 'quality'), " +
                "a HuggingFace repo ID (e.g., 'Xenova/vit-gpt2-image-captioning'), " +
                "or a local path to a model directory.",
                modelIdOrPath);
        }

        // Create the appropriate captioner based on model type
        // modelInfo is guaranteed non-null here due to control flow above
        return await CreateCaptionerAsync(modelDir, modelInfo!, variantSuffix, options, tokenizerDir).ConfigureAwait(false);
    }

    /// <summary>
    /// Bytes <see cref="LoadAsync"/> would download for the same id and options into an empty cache: the files that
    /// load picks (for a model published in quantized variants, the variant <see cref="LMSupplyOptionsBase.QuantizationHint"/>
    /// or the hardware tier selects, exactly as the load decides), at the lengths the repository lists. For a consent
    /// screen that states what a first run will fetch.
    /// </summary>
    /// <remarks>
    /// Reads the repository listing (one request, cached for a day and reused by the download that follows); downloads
    /// and loads nothing. The figure is the whole download whatever the cache already holds. A local path downloads
    /// nothing, so it is 0. The native ONNX Runtime a first load also provisions, once per host and shared by every
    /// model, is not counted. With <see cref="CaptionerOptions.DisableAutoDownload"/> the listing comes from the cache
    /// only, as the load's does.
    /// </remarks>
    /// <param name="modelIdOrPath">Anything <see cref="LoadAsync"/> accepts; a <c>:variant</c> qualifier is honoured.</param>
    /// <param name="options">The options the load will use; not modified.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <exception cref="ModelNotFoundException">The model is unknown, the repository does not exist, or (downloads disabled) it was never listed into this cache.</exception>
    /// <exception cref="ModelDownloadException">A file the load needs is not in the repository.</exception>
    public static async Task<long> GetDownloadSizeBytesAsync(
        string modelIdOrPath,
        CaptionerOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(modelIdOrPath);
        options = options?.Clone() ?? new CaptionerOptions();
        ExecutionProviderSupport.ThrowIfUnsupported(options.Provider);

        var plans = await PlanDownloadsAsync(modelIdOrPath, options, cancellationToken).ConfigureAwait(false);
        return plans.Sum(p => p.TotalBytes);
    }

    /// <summary>
    /// Bytes a load with the same id and options would still download now: the files of
    /// <see cref="GetDownloadSizeBytesAsync"/> that the cache does not hold at the length the repository lists. 0 when
    /// the model is cached or on local disk — for deciding whether to ask the user at all.
    /// </summary>
    /// <remarks>
    /// Reads the repository listing as <see cref="GetDownloadSizeBytesAsync"/> does, and the cache; downloads nothing. A
    /// partly downloaded file counts in full. Runtimes are not counted.
    /// </remarks>
    public static async Task<long> GetRemainingDownloadBytesAsync(
        string modelIdOrPath,
        CaptionerOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(modelIdOrPath);
        options = options?.Clone() ?? new CaptionerOptions();
        ExecutionProviderSupport.ThrowIfUnsupported(options.Provider);

        var plans = await PlanDownloadsAsync(modelIdOrPath, options, cancellationToken).ConfigureAwait(false);
        var cacheDir = options.CacheDirectory ?? CacheManager.GetDefaultCacheDirectory();
        return plans.Sum(p => p.GetRemainingBytes(cacheDir));
    }

    // The plans a load with these options fetches; empty for a model on local disk. Shared by the total and the
    // remaining figure, so both count the files the load picks.
    private static async Task<IReadOnlyList<DownloadPlan>> PlanDownloadsAsync(
        string modelIdOrPath, CaptionerOptions options, CancellationToken cancellationToken)
    {
        var (baseId, qualifier) = LMSupplyOptionsBase.SplitQualifier(modelIdOrPath);
        var hint = options.QuantizationHint ?? qualifier;

        if (Directory.Exists(baseId))
            return [];

        var cacheDir = options.CacheDirectory ?? CacheManager.GetDefaultCacheDirectory();
        using var downloader = new HuggingFaceDownloader(cacheDir, localFilesOnly: options.DisableAutoDownload);

        if (TryResolveKnownModel(baseId, out var modelInfo))
        {
            var plan = await downloader.PlanModelAsync(
                modelInfo!.RepoId,
                files: GetRequiredFiles(modelInfo, SelectVariantSuffix(modelInfo, options.Provider, hint)),
                subfolder: modelInfo.Subfolder,
                cancellationToken: cancellationToken).ConfigureAwait(false);
            return [plan];
        }

        if (baseId.Contains('/'))
        {
            var plan = await downloader.PlanWithDiscoveryAsync(
                baseId, DiscoveryPreferences(options.Provider, hint), cancellationToken: cancellationToken).ConfigureAwait(false);
            return [plan];
        }

        throw new ModelNotFoundException(
            $"Unknown model '{baseId}'. Use a known model ID (e.g., 'default', 'quality'), a HuggingFace repo ID, " +
            "or a local path to a model directory.",
            baseId);
    }

    /// <summary>
    /// Gets a list of pre-configured model IDs available for download.
    /// </summary>
    public static IEnumerable<string> GetAvailableModels() =>
        CaptionerModelRegistry.Default.GetAliases().Select(a => a.Name);

    /// <summary>
    /// Gets all registered model information, deduplicated by model id — aliases that point at the
    /// same model (e.g. "default" and "fast") contribute one entry. Use <see cref="GetAvailableModels"/>
    /// for the alias names themselves.
    /// </summary>
    public static IEnumerable<ModelInfo> GetAllModels() =>
        CaptionerModelRegistry.Default.GetAvailableModels();

    private static async Task<ICaptionerModel> CreateCaptionerAsync(
        string modelDir,
        ModelInfo modelInfo,
        string variantSuffix,
        CaptionerOptions options,
        string? tokenizerDir = null)
    {
        return modelInfo.Architecture switch
        {
            CaptionerArchitecture.Florence2 => await Florence2Captioner.CreateAsync(
                modelDir, OnnxFiles(modelInfo, variantSuffix), modelInfo, options, tokenizerDir).ConfigureAwait(false),
            _ => await VitGpt2Captioner.CreateAsync(modelDir, modelInfo, options, tokenizerDir).ConfigureAwait(false)
        };
    }

    /// <summary>
    /// Refuses an option the model would otherwise ignore — before anything is downloaded.
    /// </summary>
    internal static void EnsureOptionsSupported(ModelInfo modelInfo, CaptionerOptions options)
    {
        if (options.NumBeams < 1)
            throw new NotSupportedException($"CaptionerOptions.NumBeams must be at least 1; it is {options.NumBeams}.");
        if (options.Temperature is <= 0f)
            throw new NotSupportedException($"CaptionerOptions.Temperature must be positive when set; it is {options.Temperature}.");
        if (options.Temperature is not null && options.NumBeams > 1)
            throw new NotSupportedException(
                "CaptionerOptions.Temperature samples one token at a time and NumBeams searches deterministically; set one of them.");

        switch (modelInfo.Architecture)
        {
            case CaptionerArchitecture.Florence2 when !string.IsNullOrWhiteSpace(options.Prompt):
                throw new NotSupportedException(
                    $"Model '{modelInfo.AliasName}' does not continue a caption from a prompt: its prompt selects a task. " +
                    "Leave CaptionerOptions.Prompt unset and choose the level of detail with CaptionerOptions.Detail.");

            case CaptionerArchitecture.VitGpt2 when options.Detail != CaptionDetail.Brief:
                throw new NotSupportedException(
                    $"Model '{modelInfo.AliasName}' captions at one level of detail only; CaptionerOptions.Detail = " +
                    $"{options.Detail} needs a model that captions at several (alias 'quality').");
        }
    }

    /// <summary>
    /// The file-name suffix of the quantized variant a load of <paramref name="modelInfo"/> uses: empty for a model
    /// published at one precision, otherwise the variant the quantization hint or the provider's hardware tier ranks
    /// first.
    /// </summary>
    internal static string SelectVariantSuffix(ModelInfo modelInfo, CaptionerOptions options)
        => SelectVariantSuffix(modelInfo, options.Provider, options.QuantizationHint);

    private static string SelectVariantSuffix(ModelInfo modelInfo, ExecutionProvider provider, string? hint)
    {
        if (!modelInfo.HasQuantizationVariants)
            return "";

        var preferences = hint is not null
            ? ModelPreferences.ForQuantizationHint(hint)
            : ModelPreferences.ForProvider(provider);
        return SuffixFor(preferences.QuantizationPriority[0]);
    }

    private static string SuffixFor(Quantization quantization) => quantization switch
    {
        Quantization.Fp16 => "_fp16",
        Quantization.Quant8 => "_quantized",
        Quantization.Quant4 => "_q4",
        _ => ""
    };

    private static ModelPreferences DiscoveryPreferences(CaptionerOptions options)
        => DiscoveryPreferences(options.Provider, options.QuantizationHint);

    private static ModelPreferences DiscoveryPreferences(ExecutionProvider provider, string? hint)
    {
        var hwPrefs = ModelPreferences.ForProvider(provider);
        return hint is not null
            ? new ModelPreferences
            {
                PreferLowMemory = hwPrefs.PreferLowMemory,
                QuantizationPriority = ModelPreferences.ForQuantizationHint(hint).QuantizationPriority,
                PreferredProvider = provider != ExecutionProvider.Auto ? provider : hwPrefs.PreferredProvider
            }
            : hwPrefs;
    }

    /// <summary>
    /// The ONNX files a load opens, in the order the model's captioner takes them, with the variant suffix applied.
    /// For Florence-2: vision encoder, token embedding, text encoder, decoder.
    /// </summary>
    internal static IReadOnlyList<string> OnnxFiles(ModelInfo modelInfo, string variantSuffix)
    {
        static string WithSuffix(string file, string suffix)
            => suffix.Length == 0 ? file : $"{Path.GetFileNameWithoutExtension(file)}{suffix}{Path.GetExtension(file)}";

        IEnumerable<string> files = modelInfo.Architecture == CaptionerArchitecture.Florence2
            ? [modelInfo.EncoderFile, .. modelInfo.AdditionalFiles, modelInfo.DecoderFile]
            : [modelInfo.EncoderFile, modelInfo.DecoderFile, .. modelInfo.AdditionalFiles];

        return files.Select(f => WithSuffix(f, variantSuffix)).ToList();
    }

    /// <summary>
    /// Resolves only known models (system aliases, user aliases, registered IDs).
    /// Does NOT create fallback for unknown HuggingFace repos, preserving the
    /// auto-discovery download path for truly unknown models.
    /// </summary>
    private static bool TryResolveKnownModel(string modelIdOrAlias, out ModelInfo? modelInfo)
    {
        var registry = CaptionerModelRegistry.Default;

        // Check if it would resolve to a fallback (contains '/' but not registered)
        // In that case, return false so the auto-discovery path is used instead
        if (modelIdOrAlias.Contains('/'))
        {
            // Only resolve if it's a registered model ID (not a fallback)
            var knownModels = registry.GetAvailableModels();
            var match = knownModels.FirstOrDefault(m =>
                m.RepoId.Equals(modelIdOrAlias, StringComparison.OrdinalIgnoreCase));
            if (match is not null)
            {
                modelInfo = match;
                return true;
            }
            modelInfo = null;
            return false;
        }

        return registry.TryResolve(modelIdOrAlias, out modelInfo);
    }

    /// <summary>
    /// Tries to infer model info from directory contents.
    /// Handles cases where ONNX files and tokenizer files are in different directories.
    /// </summary>
    /// <param name="onnxDir">Directory containing ONNX model files.</param>
    /// <param name="baseDir">Base directory containing tokenizer/config files.</param>
    /// <param name="options">The load's options; a Florence-2 directory holding several variants is read at the one they prefer.</param>
    /// <param name="modelInfo">The inferred model info if successful.</param>
    /// <param name="variantSuffix">The variant suffix of the files found (empty for full precision).</param>
    /// <returns>True if model info was successfully inferred.</returns>
    internal static bool TryInferModelInfo(
        string onnxDir, string baseDir, CaptionerOptions options, out ModelInfo? modelInfo, out string variantSuffix)
    {
        variantSuffix = "";
        var hasVocab = File.Exists(Path.Combine(baseDir, "vocab.json")) || File.Exists(Path.Combine(onnxDir, "vocab.json"));

        // Florence-2: the four graphs, at the variant the options prefer when several are present.
        var florence = DefaultModels.Florence2Base;
        var preferred = SelectVariantSuffix(florence, options);
        foreach (var suffix in new[] { preferred, "", "_fp16", "_quantized", "_q4" }.Distinct())
        {
            if (hasVocab && OnnxFiles(florence, suffix).All(f => File.Exists(Path.Combine(onnxDir, f))))
            {
                modelInfo = florence;
                variantSuffix = suffix;
                return true;
            }
        }

        // ViT-GPT2 style model (encoder_model.onnx + decoder_model_merged.onnx). Florence-2 uses the same two names for
        // its text encoder and decoder, so a directory with any of Florence's own graphs is an incomplete Florence
        // download, not a ViT-GPT2 one — running it as ViT-GPT2 would caption with the wrong model.
        var hasFlorenceGraphs = Directory.EnumerateFiles(onnxDir, "vision_encoder*.onnx").Any()
            || Directory.EnumerateFiles(onnxDir, "embed_tokens*.onnx").Any();
        if (hasVocab && !hasFlorenceGraphs
            && File.Exists(Path.Combine(onnxDir, "encoder_model.onnx"))
            && File.Exists(Path.Combine(onnxDir, "decoder_model_merged.onnx")))
        {
            // The model this layout belongs to — not "default", which names Florence-2 since 0.104.0.
            modelInfo = DefaultModels.VitGpt2;
            return true;
        }

        modelInfo = null;
        return false;
    }

    /// <summary>
    /// Gets the list of required files for a model at the given variant.
    /// </summary>
    internal static IEnumerable<string> GetRequiredFiles(ModelInfo modelInfo, string variantSuffix)
    {
        // ONNX model files
        foreach (var file in OnnxFiles(modelInfo, variantSuffix))
            yield return file;

        // Common tokenizer and config files (these are typically in root, not subfolder)
        yield return "config.json";
        yield return "vocab.json";
        yield return "merges.txt";
        yield return "tokenizer.json";
        yield return "tokenizer_config.json";
        yield return "special_tokens_map.json";
    }
}
