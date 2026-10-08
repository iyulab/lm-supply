using LMSupply.Core.Download;
using LMSupply.Download;
using LMSupply.Embedder.Inference;
using LMSupply.Embedder.Pooling;
using LMSupply.Embedder.Utils;
using LMSupply.Hardware;
using LMSupply.Inference;
using LMSupply.Text;

namespace LMSupply.Embedder;

/// <summary>
/// Main entry point for loading and using embedding models.
/// </summary>
public static class LocalEmbedder
{
    /// <summary>
    /// Default model to use when no model is specified.
    /// BGE Small English v1.5, 33M params, MTEB top performer.
    /// </summary>
    public const string DefaultModel = "default";

    /// <summary>
    /// Gets the model registry for embedding models.
    /// Supports system aliases, user aliases, and model resolution.
    /// </summary>
    public static IModelRegistry<ModelInfo> Registry => EmbedderModelRegistry.Default;

    /// <summary>
    /// Shared pool for named model management. Supports GetOrLoadAsync / UnloadAsync by model ID.
    /// </summary>
    public static LMSupply.Pool.ModelPool<IEmbeddingModel, EmbedderOptions> Pool { get; }
        = new(new Pool.EmbedderLoader());

    /// <summary>
    /// Loads the default embedding model.
    /// </summary>
    /// <param name="options">Optional configuration options.</param>
    /// <param name="progress">Optional progress reporting for downloads.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>A loaded embedding model ready for inference.</returns>
    public static Task<IEmbeddingModel> LoadDefaultAsync(
        EmbedderOptions? options = null,
        IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        return LoadAsync(DefaultModel, options, progress, cancellationToken);
    }

    /// <summary>
    /// Loads an embedding model by name or path.
    /// </summary>
    /// <param name="modelIdOrPath">
    /// Either a catalog alias (e.g., "default", "fast", "large") or model ID for auto-download,
    /// or a local path to an ONNX/GGUF model file.
    /// GGUF models are auto-detected by "-GGUF"/"_gguf" in repo name or ".gguf" extension.
    /// </param>
    /// <param name="options">Optional configuration options.</param>
    /// <param name="progress">Optional progress reporting for downloads.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>A loaded embedding model ready for inference.</returns>
    public static async Task<IEmbeddingModel> LoadAsync(
        string modelIdOrPath,
        EmbedderOptions? options = null,
        IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        options = options?.Clone() ?? new EmbedderOptions();
        // An unsupported provider is refused here, before any model resolution or download (0.67.1).
        ExecutionProviderSupport.ThrowIfUnsupported(options.Provider);

        modelIdOrPath = PrepareLoadTarget(modelIdOrPath, options);

        // Check for GGUF format
        if (IsGgufModel(modelIdOrPath))
        {
            return await LoadGgufAsync(modelIdOrPath, options, progress, cancellationToken);
        }

        var sources = await ResolveOnnxSourcesAsync(modelIdOrPath, options, progress, cancellationToken);

        // Load tokenizer using Text.Core (auto-detects WordPiece vs SentencePiece)
        var tokenizer = await TokenizerFactory.CreateAutoSequenceAsync(sources.TokenizerDir, sources.MaxSequenceLength, cancellationToken);

        // Load inference engine (use async to ensure RuntimeManager initializes native binaries)
        var engine = await OnnxInferenceEngine.CreateAsync(sources.ModelPath, options.Provider, options, progress, cancellationToken);

        LogProviderSelection(sources.ModelId, options.Provider, engine);

        // Create pooling strategy
        var poolingStrategy = PoolingFactory.Create(sources.PoolingMode);

        // What the model info reports is what the loader did — not only what the catalog said. A model
        // without a catalog entry gets one built from its own files, so GetModelInfo() is never null and
        // the prompt prefixes it declares are applied.
        var loadedModelInfo = sources.CatalogInfo is null
            ? new ModelInfo
            {
                RepoId = sources.RepoIdForInfo,
                AliasName = sources.ModelId,
                Dimensions = engine.HiddenSize,
                MaxSequenceLength = sources.MaxSequenceLength,
                PoolingMode = sources.PoolingMode,
                DoLowerCase = options.DoLowerCase,
                QueryPrefix = sources.Prompts.Query,
                PassagePrefix = sources.Prompts.Passage,
                DefaultPrefix = sources.Prompts.Default,
                Subfolder = sources.Subfolder,
                SizeBytes = File.Exists(sources.ModelPath) ? new FileInfo(sources.ModelPath).Length : 0
            }
            : sources.CatalogInfo with
            {
                MaxSequenceLength = sources.MaxSequenceLength,
                PoolingMode = sources.PoolingMode,
                QueryPrefix = sources.Prompts.Query,
                PassagePrefix = sources.Prompts.Passage,
                DefaultPrefix = sources.Prompts.Default,
            };

        // GetVectorSpaceRevisionAsync answers the dimension from the files (catalog entry, then config.json);
        // the graph is the authority here. When the two disagree the pre-load revision differs from this one —
        // say so, so the mismatch is visible instead of a silent re-embed.
        var declaredDimensions = TryReadDeclaredDimensions(sources);
        if (declaredDimensions is { } declared && declared != engine.HiddenSize)
        {
            System.Diagnostics.Trace.TraceWarning(
                $"[LocalEmbedder.vectorspace] {sources.ModelId}: the model files declare {declared} dimensions but the graph produces {engine.HiddenSize}; " +
                "GetVectorSpaceRevisionAsync reports a different revision than the loaded model for this id.");
        }

        // The vector-space revision is derived from what was decided above — the tokenizer as built, the
        // pooling and length in effect, the prefixes the info carries, the model file actually opened —
        // so it moves exactly when those move. The canonical line is traced so two revisions can
        // be diffed by eye; the consumer sees only the hash.
        var vectorSpace = BuildVectorSpace(sources, tokenizer.Signature, options.NormalizeEmbeddings, engine.HiddenSize);
        System.Diagnostics.Trace.TraceInformation(
            $"[LocalEmbedder.vectorspace] {sources.ModelId}: {vectorSpace.Canonical} -> {vectorSpace.Revision}");

        return new EmbeddingModel(sources.ModelId, engine, tokenizer, poolingStrategy, options, loadedModelInfo, sources.ModelPath, vectorSpace);
    }

    /// <summary>
    /// The first step of every load, shared with <see cref="GetDownloadSizeBytesAsync"/> so the two resolve the same
    /// target: a <c>:variant</c> qualifier becomes the quantization hint (e.g. <c>"default:fp16"</c>), then a user alias
    /// is translated. The translation precedes format detection, so an alias that points at <c>"gguf:..."</c> enters the
    /// GGUF path.
    /// </summary>
    private static string PrepareLoadTarget(string modelIdOrPath, EmbedderOptions options)
    {
        var (baseId, qualifier) = LMSupplyOptionsBase.SplitQualifier(modelIdOrPath);
        options.QuantizationHint ??= qualifier;

        return EmbedderModelRegistry.Default.TryGetUserAliasTarget(baseId, out var userAliasTarget)
            ? userAliasTarget!
            : baseId;
    }

    /// <summary>
    /// Bytes <see cref="DownloadModelAsync"/> (and so the first <see cref="LoadAsync"/>) would download for the same id
    /// and options into an empty cache: the model files that load picks on this host, at the lengths the repository
    /// lists. For a consent screen that states what a first run will fetch.
    /// </summary>
    /// <remarks>
    /// Reads the repository listing (one request, cached for a day and reused by the download that follows); downloads
    /// and loads nothing. The figure is the whole download whatever the cache already holds;
    /// <see cref="IsModelDownloaded"/> answers what is present. A local path downloads nothing, so it is 0. Runtimes a
    /// first load also provisions (the native ONNX Runtime, or llama-server for a GGUF model), once per host and shared
    /// by every model, are not counted. With <see cref="EmbedderOptions.DisableAutoDownload"/> the listing comes from the
    /// cache only, as the load's choice does.
    /// </remarks>
    /// <param name="modelIdOrPath">Anything <see cref="LoadAsync"/> accepts.</param>
    /// <param name="options">The options the load will use; not modified.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <exception cref="ModelNotFoundException">The model is unknown, the repository does not exist or holds no model file, or (downloads disabled) it was never listed into this cache.</exception>
    /// <exception cref="ModelDownloadException">A file the load needs is not in the repository, or its listing gives no length.</exception>
    public static async Task<long> GetDownloadSizeBytesAsync(
        string modelIdOrPath,
        EmbedderOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(modelIdOrPath);
        options = options?.Clone() ?? new EmbedderOptions();
        ExecutionProviderSupport.ThrowIfUnsupported(options.Provider);

        var plan = await PlanDownloadAsync(PrepareLoadTarget(modelIdOrPath, options), options, cancellationToken);
        return plan?.TotalBytes ?? 0;
    }

    /// <summary>
    /// Bytes a <see cref="LoadAsync"/> with the same id and options would still download now: the files of
    /// <see cref="GetDownloadSizeBytesAsync"/> that the cache does not hold at the length the repository lists. 0 when the
    /// model is cached or on local disk — for deciding whether to ask the user at all.
    /// </summary>
    /// <remarks>
    /// Reads the repository listing as <see cref="GetDownloadSizeBytesAsync"/> does, and the cache; downloads nothing. A
    /// partly downloaded file counts in full. Runtimes (the native ONNX Runtime, llama-server) are not counted.
    /// </remarks>
    /// <param name="modelIdOrPath">Anything <see cref="LoadAsync"/> accepts.</param>
    /// <param name="options">The options the load will use; not modified.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <exception cref="ModelNotFoundException">As for <see cref="GetDownloadSizeBytesAsync"/>.</exception>
    /// <exception cref="ModelDownloadException">As for <see cref="GetDownloadSizeBytesAsync"/>.</exception>
    public static async Task<long> GetRemainingDownloadBytesAsync(
        string modelIdOrPath,
        EmbedderOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(modelIdOrPath);
        options = options?.Clone() ?? new EmbedderOptions();
        ExecutionProviderSupport.ThrowIfUnsupported(options.Provider);

        var plan = await PlanDownloadAsync(PrepareLoadTarget(modelIdOrPath, options), options, cancellationToken);
        return plan?.GetRemainingBytes(options.CacheDirectory ?? CacheManager.GetDefaultCacheDirectory()) ?? 0;
    }

    /// <summary>
    /// The files a load of <paramref name="target"/> (already through <see cref="PrepareLoadTarget"/>) fetches, chosen
    /// by the same branches as <see cref="LoadAsync"/>: null for a model on local disk.
    /// </summary>
    internal static async Task<DownloadPlan?> PlanDownloadAsync(
        string target,
        EmbedderOptions options,
        CancellationToken cancellationToken)
    {
        var cacheDir = options.CacheDirectory ?? CacheManager.GetDefaultCacheDirectory();

        if (IsGgufModel(target))
        {
            var repoIdOrPath = StripGgufPrefix(target);
            if (File.Exists(repoIdOrPath))
                return null;
            if (!repoIdOrPath.Contains('/'))
                throw GgufModelNotFound(target);

            using var ggufDownloader = new GgufDownloader(cacheDir, localFilesOnly: options.DisableAutoDownload);
            return await ggufDownloader.PlanAsync(repoIdOrPath, GgufPreferredQuantization, cancellationToken);
        }

        if (File.Exists(target) || target.EndsWith(".onnx", StringComparison.OrdinalIgnoreCase))
            return null;

        using var downloader = new HuggingFaceDownloader(cacheDir, localFilesOnly: options.DisableAutoDownload);

        if (EmbedderModelRegistry.Default.TryResolveCatalog(target, out var modelInfo, out var resolvedId))
        {
            return await downloader.PlanModelAsync(
                modelInfo!.RepoId, subfolder: modelInfo.Subfolder, cancellationToken: cancellationToken);
        }

        if (resolvedId.Contains('/'))
            return await downloader.PlanWithDiscoveryAsync(resolvedId, OnnxPreferences(options), cancellationToken: cancellationToken);

        throw UnknownModel(target);
    }

    /// <summary>The ONNX file preferences a repository-id load discovers with: the provider's, narrowed by a quantization hint.</summary>
    private static ModelPreferences OnnxPreferences(EmbedderOptions options)
    {
        var hwPrefs = ModelPreferences.ForProvider(options.Provider);
        return options.QuantizationHint is { } hint
            ? new ModelPreferences
            {
                PreferLowMemory = hwPrefs.PreferLowMemory,
                QuantizationPriority = ModelPreferences.ForQuantizationHint(hint).QuantizationPriority,
                PreferredProvider = options.Provider != ExecutionProvider.Auto
                    ? options.Provider : hwPrefs.PreferredProvider
            }
            : hwPrefs;
    }

    private static ModelNotFoundException UnknownModel(string modelIdOrPath) => new(
        $"Unknown model '{modelIdOrPath}'. Use a known catalog alias (e.g., 'default', 'fast'), " +
        "a HuggingFace repo ID (e.g., 'sentence-transformers/all-MiniLM-L6-v2'), " +
        "or a local path to an ONNX model file.",
        modelIdOrPath);

    /// <summary>
    /// The vector-space revision <see cref="LoadAsync"/> would report for <paramref name="modelIdOrPath"/>,
    /// computed from the cached files alone — no inference session, no download, no request (0.72.0).
    /// <c>null</c> when it cannot be known without loading: the model (or one of its files) is not in the
    /// cache, the id is unknown, the dimension is declared by neither the catalog nor the repository's
    /// <c>config.json</c>, or the model is GGUF (llama-server decides its dimension).
    /// </summary>
    /// <remarks>
    /// <para>
    /// The same decisions <see cref="LoadAsync"/> makes — model file, tokenizer as built from its files,
    /// pooling, normalization, sequence length, prefixes — go through the same code
    /// (<c>ResolveOnnxSourcesAsync</c>), so the two agree by construction for everything but the
    /// dimension, which the load takes from the ONNX graph and this method from the catalog entry or the
    /// repository's <c>config.json</c> (<c>hidden_size</c>). A model whose graph output width differs from
    /// its declared hidden size gets a different value here than after the load; the load traces a warning
    /// for it.
    /// </para>
    /// <para>
    /// This is what lets a consumer that names its vector store after the identity — before the model is
    /// loaded — use the revision without forcing a warm-up. Reads only the fields of
    /// <paramref name="options"/> that <see cref="LoadAsync"/> reads while resolving (cache directory,
    /// quantization hint, provider, sequence length, pooling, lower-casing, normalization) and never
    /// mutates the caller's instance.
    /// </para>
    /// </remarks>
    public static async Task<string?> GetVectorSpaceRevisionAsync(
        string modelIdOrPath,
        EmbedderOptions? options = null,
        CancellationToken cancellationToken = default) =>
        (await ReadVectorSpaceAsync(modelIdOrPath, options, cancellationToken).ConfigureAwait(false))?.Revision;

    /// <summary>
    /// The <see cref="IEmbeddingModel.RetrievalVectorSpaceRevision"/> a load of <paramref name="modelIdOrPath"/> would report,
    /// read from the cached files alone — the retrieval counterpart of
    /// <see cref="GetVectorSpaceRevisionAsync(string, EmbedderOptions?, CancellationToken)"/>, with the same rules for when it
    /// answers <c>null</c> and the same agreement with the load.
    /// </summary>
    public static async Task<string?> GetRetrievalVectorSpaceRevisionAsync(
        string modelIdOrPath,
        EmbedderOptions? options = null,
        CancellationToken cancellationToken = default) =>
        (await ReadVectorSpaceAsync(modelIdOrPath, options, cancellationToken).ConfigureAwait(false))?.RetrievalRevision;

    private static async Task<VectorSpaceDescriptor?> ReadVectorSpaceAsync(
        string modelIdOrPath,
        EmbedderOptions? options,
        CancellationToken cancellationToken)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(modelIdOrPath);
        var source = options ?? new EmbedderOptions();
        var offline = new EmbedderOptions
        {
            CacheDirectory = source.CacheDirectory,
            QuantizationHint = source.QuantizationHint,
            Provider = source.Provider,
            MaxSequenceLength = source.MaxSequenceLength,
            PoolingMode = source.PoolingMode,
            DoLowerCase = source.DoLowerCase,
            NormalizeEmbeddings = source.NormalizeEmbeddings,
            DisableAutoDownload = true,
        };

        var (baseId, qualifier) = LMSupplyOptionsBase.SplitQualifier(modelIdOrPath);
        offline.QuantizationHint ??= qualifier;
        if (EmbedderModelRegistry.Default.TryGetUserAliasTarget(baseId, out var userAliasTarget))
            baseId = userAliasTarget!;

        if (IsGgufModel(baseId))
            return null;

        OnnxSources sources;
        try
        {
            sources = await ResolveOnnxSourcesAsync(baseId, offline, progress: null, cancellationToken);
        }
        catch (ModelNotFoundException)
        {
            return null;
        }

        var dimensions = TryReadDeclaredDimensions(sources);
        if (dimensions is null)
            return null;

        var tokenizer = await TokenizerFactory.CreateAutoSequenceAsync(sources.TokenizerDir, sources.MaxSequenceLength, cancellationToken);
        return BuildVectorSpace(sources, tokenizer.Signature, offline.NormalizeEmbeddings, dimensions.Value);
    }

    /// <summary>
    /// Everything <see cref="LoadAsync"/> decides from the model's files and the catalog before it opens an
    /// inference session — one function, so a pre-load read and the load cannot drift apart.
    /// </summary>
    private sealed record OnnxSources(
        string ModelId,
        string ModelPath,
        string TokenizerDir,
        string? ModelRootDir,
        string? Subfolder,
        string RepoIdForInfo,
        ModelInfo? CatalogInfo,
        int MaxSequenceLength,
        PoolingMode PoolingMode,
        PromptPrefixes Prompts);

    private static async Task<OnnxSources> ResolveOnnxSourcesAsync(
        string modelIdOrPath,
        EmbedderOptions options,
        IProgress<DownloadProgress>? progress,
        CancellationToken cancellationToken)
    {
        string modelPath;
        // Directories to probe for tokenizer files (model dir first, then repo root if different).
        // The first entry must contain model.onnx; subsequent entries are fall-back search roots
        // for tokenizer assets that HuggingFace repos commonly place at the repo root rather than
        // inside the onnx/ subfolder (e.g. tokenizer.json, sentencepiece.bpe.model).
        string tokenizerPrimaryDir;
        string? tokenizerFallbackDir;
        string modelId;

        ModelInfo? loadedModelInfo = null;
        int? catalogMaxSequenceLength = null;
        PoolingMode? catalogPoolingMode = null;
        // The repository root: where a sentence-transformers model keeps 1_Pooling/ and
        // config_sentence_transformers.json (never inside the onnx/ subfolder).
        string? modelRootDir;
        string? subfolder = null;
        var repoIdForInfo = modelIdOrPath;
        // A file on disk whose catalog identity came from its own config — its declared prompts, if any, still win.
        var identityFromFiles = false;

        // Check if it's a local path
        if (File.Exists(modelIdOrPath) || modelIdOrPath.EndsWith(".onnx", StringComparison.OrdinalIgnoreCase))
        {
            modelPath = modelIdOrPath;
            modelId = Path.GetFileNameWithoutExtension(modelPath);

            tokenizerPrimaryDir = Path.GetDirectoryName(modelPath) ?? ".";
            // A copied repository keeps model.onnx in onnx/ and its root files (modules.json, 1_Pooling/,
            // config_sentence_transformers.json, often the tokenizer) one level up — read them where the repository
            // and catalog loads read them.
            modelRootDir = FindRepositoryRoot(tokenizerPrimaryDir);
            tokenizerFallbackDir = string.Equals(modelRootDir, tokenizerPrimaryDir, StringComparison.OrdinalIgnoreCase)
                ? null
                : modelRootDir;

            // A file the catalog knows is the same model as its alias: same prefixes, pooling and length. Without this an
            // E5 model loaded by path embedded queries and passages with no "query: "/"passage: " prefix, and nothing
            // said so — E5's repository does not declare them; only the catalog does.
            if (ReadDeclaredRepositoryId(tokenizerPrimaryDir, modelRootDir) is { } declaredRepoId
                && EmbedderModelRegistry.Default.TryResolveCatalog(declaredRepoId, out var pathModelInfo, out _))
            {
                loadedModelInfo = pathModelInfo;
                identityFromFiles = true;
                catalogMaxSequenceLength = pathModelInfo!.MaxSequenceLength;
                catalogPoolingMode = pathModelInfo.PoolingMode;
                options.DoLowerCase = pathModelInfo.DoLowerCase;
            }
        }
        // Check if it's a model the catalog knows. Not TryResolve: that answers any "org/repo" with a
        // fallback entry whose dimensions, pooling, length and subfolder are placeholders, and the
        // repository-id branch below — which reads the model's own files — was never reached.
        else if (EmbedderModelRegistry.Default.TryResolveCatalog(modelIdOrPath, out var modelInfo, out var resolvedId))
        {
            loadedModelInfo = modelInfo;

            // Apply model-specific defaults (the sequence length is settled below, once the model's
            // own files are on disk).
            catalogMaxSequenceLength = modelInfo!.MaxSequenceLength;
            catalogPoolingMode = modelInfo.PoolingMode;
            options.DoLowerCase = modelInfo.DoLowerCase;
            subfolder = modelInfo.Subfolder;

            // Download model
            var cacheDir = options.CacheDirectory ?? CacheManager.GetDefaultCacheDirectory();
            using var downloader = new HuggingFaceDownloader(cacheDir, localFilesOnly: options.DisableAutoDownload);

            var modelDir = await downloader.DownloadModelAsync(
                modelInfo.RepoId,
                subfolder: modelInfo.Subfolder,
                progress: progress,
                cancellationToken: cancellationToken);

            modelPath = Path.Combine(modelDir, "model.onnx");
            // modelDir is the subfolder's own directory; DownloadModelAsync puts tokenizer assets it had
            // to fetch from the repository root there too, so they live alongside model.onnx.
            tokenizerPrimaryDir = modelDir;
            tokenizerFallbackDir = null;
            modelId = modelIdOrPath;
            // The catalog is the authority for a known alias; the root is kept only for what it may add.
            modelRootDir = string.IsNullOrEmpty(modelInfo.Subfolder) ? modelDir : Path.GetDirectoryName(modelDir);
        }
        // A HuggingFace repository the catalog does not know (e.g., "BAAI/bge-small-en-v1.5"): the
        // model's own files are the only declaration there is, so they are downloaded and read.
        else if (resolvedId.Contains('/'))
        {
            var cacheDir = options.CacheDirectory ?? CacheManager.GetDefaultCacheDirectory();
            using var downloader = new HuggingFaceDownloader(cacheDir, localFilesOnly: options.DisableAutoDownload);

            // Use auto-discovery to find ONNX files and config
            var (downloadedDir, discovery) = await downloader.DownloadWithDiscoveryAsync(
                resolvedId,
                preferences: OnnxPreferences(options),
                progress: progress,
                cancellationToken: cancellationToken);

            // Find the main model ONNX file
            var mainOnnxFile = discovery.OnnxFiles.FirstOrDefault(f =>
                f.EndsWith("model.onnx", StringComparison.OrdinalIgnoreCase)) ??
                (discovery.OnnxFiles.Count > 0 ? discovery.OnnxFiles[0] : null);

            if (mainOnnxFile is null)
            {
                throw new ModelNotFoundException(
                    $"No ONNX model file found in repository '{resolvedId}'.",
                    resolvedId);
            }

            // Preserve full path including subfolder (e.g., "onnx/model.onnx")
            modelPath = Path.Combine(downloadedDir, mainOnnxFile);

            // DownloadWithDiscoveryAsync preserves directory structure, so the model may live in
            // a subfolder (e.g. onnx/) while tokenizer assets sit at the repo root.
            tokenizerPrimaryDir = Path.GetDirectoryName(modelPath)!;
            tokenizerFallbackDir = string.Equals(tokenizerPrimaryDir, downloadedDir, StringComparison.OrdinalIgnoreCase)
                ? null
                : downloadedDir;

            modelId = resolvedId.Split('/').Last();
            modelRootDir = downloadedDir;
            subfolder = discovery.Subfolder;
            repoIdForInfo = resolvedId;
        }
        else
        {
            throw UnknownModel(modelIdOrPath);
        }

        // Validate model file exists
        if (!File.Exists(modelPath))
            throw new ModelNotFoundException("Model file not found", modelPath);

        // Resolve tokenizer directory: prefer the directory that actually contains tokenizer
        // assets. Probe primary first, then fall back to the repo root.
        var tokenizerDir = ResolveTokenizerDir(tokenizerPrimaryDir, tokenizerFallbackDir, modelId);

        // Sequence length: an explicit caller value wins; otherwise what the model declares in
        // sentence_bert_config.json (where sentence-transformers truncates - 256 for all-MiniLM-L6-v2,
        // not the 512 its architecture allows), then the catalog, then the default.
        var maxSequenceLength = SentenceBertConfig.ResolveMaxSequenceLength(
            options.MaxSequenceLength,
            SentenceBertConfig.TryReadMaxSequenceLength(tokenizerDir, tokenizerPrimaryDir, tokenizerFallbackDir),
            catalogMaxSequenceLength);
        options.MaxSequenceLength = maxSequenceLength;

        // Pooling: the same precedence. A model loaded by repository id used to be pooled with the option's
        // default (Mean) whatever its 1_Pooling/config.json said, so a CLS model loaded by id lived in a
        // different vector space from the same model loaded by alias.
        var poolingMode = options.PoolingMode
            ?? SentenceTransformersModules.TryReadPoolingMode(modelRootDir)
            ?? catalogPoolingMode
            ?? PoolingMode.Mean;
        options.PoolingMode = poolingMode;

        var filePrompts = SentenceTransformersModules.TryReadPrompts(modelRootDir);

        // The catalog is the authority for the prefixes of a model it knows by name (as before: its entry, not the
        // repository's prompts); a repository-id model declares them in its own files or not at all. A file on disk
        // that only names a catalog model in its config may be a fine-tune of it: prompts its own files declare win.
        var prompts = loadedModelInfo is not null && !(identityFromFiles && filePrompts.Any)
            ? PromptPrefixes.Of(loadedModelInfo)
            : filePrompts;

        return new OnnxSources(
            modelId, modelPath, tokenizerDir, modelRootDir, subfolder, repoIdForInfo, loadedModelInfo,
            maxSequenceLength, poolingMode, prompts);
    }

    /// <summary>
    /// The repository root for a model file in <paramref name="modelDir"/>: <paramref name="modelDir"/> itself unless it
    /// holds no sentence-transformers root file and its parent does (the <c>onnx/model.onnx</c> layout of a copied
    /// repository).
    /// </summary>
    internal static string FindRepositoryRoot(string modelDir)
    {
        static bool IsRoot(string dir) =>
            File.Exists(Path.Combine(dir, "modules.json"))
            || File.Exists(Path.Combine(dir, "config_sentence_transformers.json"))
            || Directory.Exists(Path.Combine(dir, "1_Pooling"));

        if (IsRoot(modelDir))
            return modelDir;

        var parent = Path.GetDirectoryName(Path.GetFullPath(modelDir));
        return parent is not null && IsRoot(parent) ? parent : modelDir;
    }

    /// <summary>
    /// The repository a model on disk says it came from: the download manifest LMSupply writes next to it, then
    /// <c>_name_or_path</c> in its <c>config.json</c> when that is a repository id (<c>org/name</c>), not a local path.
    /// </summary>
    internal static string? ReadDeclaredRepositoryId(string modelDir, string rootDir)
    {
        foreach (var dir in new[] { modelDir, rootDir }.Distinct(StringComparer.OrdinalIgnoreCase))
        {
            if (DownloadManifest.Read(dir)?.RepoId is { Length: > 0 } manifestRepo)
                return manifestRepo;
        }

        foreach (var dir in new[] { rootDir, modelDir }.Distinct(StringComparer.OrdinalIgnoreCase))
        {
            var config = Path.Combine(dir, "config.json");
            if (!File.Exists(config))
                continue;
            try
            {
                using var doc = System.Text.Json.JsonDocument.Parse(File.ReadAllText(config));
                if (doc.RootElement.ValueKind == System.Text.Json.JsonValueKind.Object
                    && doc.RootElement.TryGetProperty("_name_or_path", out var name)
                    && name.ValueKind == System.Text.Json.JsonValueKind.String
                    && name.GetString() is { } id
                    && id.Count(c => c == '/') == 1 && !Path.IsPathRooted(id) && !id.StartsWith('.') && !id.Contains('\\'))
                    return id;
            }
            catch (Exception e) when (e is System.Text.Json.JsonException or IOException or UnauthorizedAccessException)
            {
                // An unreadable config declares nothing.
            }
        }

        return null;
    }

    /// <summary>The dimension the files declare: the catalog entry, then the repository's <c>config.json</c>.</summary>
    private static int? TryReadDeclaredDimensions(OnnxSources sources) =>
        sources.CatalogInfo?.Dimensions is { } catalog && catalog > 0
            ? catalog
            : HuggingFaceConfig.TryReadHiddenSize(sources.ModelRootDir, sources.TokenizerDir);

    private static VectorSpaceDescriptor BuildVectorSpace(OnnxSources sources, string tokenizerSignature, bool normalize, int dimensions) =>
        new(
            Backend: "onnx",
            ModelFile: VectorSpaceDescriptor.RelativeModelFile(sources.ModelRootDir, sources.ModelPath),
            Tokenizer: tokenizerSignature,
            Pooling: sources.PoolingMode.ToString(),
            Normalize: normalize,
            MaxSequenceLength: sources.MaxSequenceLength,
            Dimensions: dimensions,
            QueryPrefix: sources.Prompts.Query,
            PassagePrefix: sources.Prompts.Passage,
            DefaultPrefix: sources.Prompts.Default);

    private static void LogProviderSelection(string modelId, ExecutionProvider requested, OnnxInferenceEngine engine)
    {
        var profile = HardwareProfile.For(requested);
        var gpu = profile.GpuInfo;
        var active = engine.ActiveProviders.Count > 0
            ? string.Join("+", engine.ActiveProviders)
            : "(none)";
        System.Diagnostics.Trace.TraceInformation(
            $"[LocalEmbedder.auto] Requested={requested}, Active={active}, " +
            $"GPU={gpu.Vendor} {gpu.DeviceName ?? "n/a"}, " +
            $"Recommended={gpu.RecommendedProvider}, model={modelId}");
    }

    /// <summary>
    /// Checks whether the ONNX Runtime native library can be loaded in the current environment.
    /// Call this before <see cref="LoadAsync"/> to verify that the host has the required
    /// shared libraries (e.g., libstdc++, libgomp on Linux; VC++ Redistributable on Windows).
    /// </summary>
    /// <returns>
    /// A tuple where <c>Available</c> is <see langword="true"/> when the runtime can be loaded,
    /// and <c>ErrorMessage</c> contains a diagnostic string when it cannot.
    /// </returns>
    public static (bool Available, string? ErrorMessage) CheckRuntimeAvailability()
        => OnnxSessionFactory.CheckOnnxRuntimeAvailability();

    /// <summary>
    /// Checks whether a model is already downloaded and available in the local cache.
    /// This does NOT load the model into memory or initialize the ONNX Runtime.
    /// </summary>
    /// <param name="modelId">
    /// A model alias (e.g., "default"), a known model ID, or a HuggingFace repo ID.
    /// </param>
    /// <param name="cacheDirectory">Custom cache directory, or <see langword="null"/> for default.</param>
    /// <returns><see langword="true"/> if the model files exist in cache and are not LFS pointers.</returns>
    public static bool IsModelDownloaded(string modelId, string? cacheDirectory = null)
    {
        var cacheDir = cacheDirectory ?? CacheManager.GetDefaultCacheDirectory();

        // Resolve alias to repo ID
        string repoId;
        string? subfolder = null;
        if (EmbedderModelRegistry.Default.TryResolve(modelId, out var modelInfo))
        {
            repoId = modelInfo!.RepoId;
            subfolder = modelInfo.Subfolder;
        }
        else
        {
            repoId = modelId;
        }

        // Any snapshot of the repository in the cache — this library's own, or one another Hugging Face tool
        // wrote under the commit "main" points at.
        return CacheManager.GetSnapshotDirectories(cacheDir, repoId).Any(snapshotDir =>
        {
            // When a subfolder is specified, model.onnx lives inside it
            var modelDir = subfolder != null ? Path.Combine(snapshotDir, subfolder) : snapshotDir;

            // Check for the essential model file
            if (!CacheManager.IsCachedFile(Path.Combine(modelDir, "model.onnx")))
                return false;

            // Check for at least one tokenizer file (model dir first, then snapshot root)
            return DirectoryHasTokenizer(modelDir)
                || (subfolder != null && DirectoryHasTokenizer(snapshotDir));
        });
    }

    /// <summary>
    /// Downloads a model without loading it into memory.
    /// If the model is already cached, this is a no-op.
    /// Use this to pre-fetch models (e.g., during container build or CI) without requiring
    /// the ONNX Runtime to be available at download time.
    /// </summary>
    /// <param name="modelId">
    /// A model alias (e.g., "default"), a known model ID, or a HuggingFace repo ID.
    /// </param>
    /// <param name="options">Optional configuration (cache directory, quantization hint).</param>
    /// <param name="progress">Optional progress reporting for downloads.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>The local directory path containing the downloaded model files.</returns>
    public static async Task<string> DownloadModelAsync(
        string modelId,
        EmbedderOptions? options = null,
        IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        options = options?.Clone() ?? new EmbedderOptions();
        // An unsupported provider is refused here, before any model resolution or download (0.67.1).
        ExecutionProviderSupport.ThrowIfUnsupported(options.Provider);

        // Parse variant qualifier
        var (baseId, qualifier) = LMSupplyOptionsBase.SplitQualifier(modelId);
        modelId = baseId;
        options.QuantizationHint ??= qualifier;

        var cacheDir = options.CacheDirectory ?? CacheManager.GetDefaultCacheDirectory();
        using var downloader = new HuggingFaceDownloader(cacheDir, localFilesOnly: options.DisableAutoDownload);

        // Known model alias / registered model
        if (EmbedderModelRegistry.Default.TryResolve(modelId, out var modelInfo))
        {
            return await downloader.DownloadModelAsync(
                modelInfo!.RepoId,
                subfolder: modelInfo.Subfolder,
                progress: progress,
                cancellationToken: cancellationToken);
        }

        // HuggingFace repo ID with auto-discovery
        if (modelId.Contains('/'))
        {
            var hwPrefs = ModelPreferences.ForProvider(options.Provider);
            var preferences = options.QuantizationHint is { } hint
                ? new ModelPreferences
                {
                    PreferLowMemory = hwPrefs.PreferLowMemory,
                    QuantizationPriority = ModelPreferences.ForQuantizationHint(hint).QuantizationPriority,
                    PreferredProvider = options.Provider != ExecutionProvider.Auto
                        ? options.Provider : hwPrefs.PreferredProvider
                }
                : hwPrefs;

            var (downloadedDir, _) = await downloader.DownloadWithDiscoveryAsync(
                modelId,
                preferences: preferences,
                progress: progress,
                cancellationToken: cancellationToken);
            return downloadedDir;
        }

        throw new ModelNotFoundException(
            $"Unknown model '{modelId}'. Use a known model alias, a HuggingFace repo ID, " +
            "or a local path to an ONNX model file.",
            modelId);
    }

    /// <summary>
    /// Gets a list of pre-configured model IDs available for download.
    /// </summary>
    public static IEnumerable<string> GetAvailableModels()
        => EmbedderModelRegistry.Default.GetAliases().Select(a => a.Name);

    /// <summary>
    /// Gets all registered model information (deduplicated by RepoId).
    /// </summary>
    public static IEnumerable<ModelInfo> GetAllModels()
        => EmbedderModelRegistry.Default.GetAvailableModels();

    /// <summary>
    /// Computes cosine similarity between two embedding vectors.
    /// </summary>
    public static float CosineSimilarity(ReadOnlySpan<float> embedding1, ReadOnlySpan<float> embedding2)
    {
        return VectorOperations.CosineSimilarity(embedding1, embedding2);
    }

    /// <summary>
    /// Computes Euclidean distance between two embedding vectors.
    /// </summary>
    public static float EuclideanDistance(ReadOnlySpan<float> embedding1, ReadOnlySpan<float> embedding2)
    {
        return VectorOperations.EuclideanDistance(embedding1, embedding2);
    }

    /// <summary>
    /// Computes dot product of two embedding vectors.
    /// </summary>
    public static float DotProduct(ReadOnlySpan<float> embedding1, ReadOnlySpan<float> embedding2)
    {
        return VectorOperations.DotProduct(embedding1, embedding2);
    }

    /// <summary>
    /// Tokenizer file names recognised by <see cref="TokenizerFactory.CreateAutoSequenceAsync"/>.
    /// Used to detect which directory actually contains tokenizer assets.
    /// </summary>
    private static readonly string[] _tokenizerProbeFiles =
    [
        "vocab.txt",
        "tokenizer.json",
        "sentencepiece.bpe.model",
    ];

    /// <summary>
    /// Picks the directory that contains tokenizer assets, preferring the model directory and
    /// falling back to the repo root when present. Throws a descriptive
    /// <see cref="ModelNotFoundException"/> when nothing usable is found.
    /// </summary>
    private static string ResolveTokenizerDir(string primaryDir, string? fallbackDir, string modelId)
    {
        if (DirectoryHasTokenizer(primaryDir))
            return primaryDir;

        if (fallbackDir is not null && DirectoryHasTokenizer(fallbackDir))
            return fallbackDir;

        // Build a helpful error listing every path we tried.
        var triedDirs = fallbackDir is null
            ? new[] { primaryDir }
            : new[] { primaryDir, fallbackDir };
        var triedFiles = string.Join(", ", triedDirs.SelectMany(d => _tokenizerProbeFiles.Select(f => Path.Combine(d, f))));

        throw new ModelNotFoundException(
            $"No tokenizer file found for '{modelId}'. " +
            $"Tried: {triedFiles}. Expected one of: vocab.txt (WordPiece), " +
            "tokenizer.json (WordPiece/Unigram/BPE), sentencepiece.bpe.model (SentencePiece).",
            modelId);
    }

    private static bool DirectoryHasTokenizer(string dir)
    {
        if (!Directory.Exists(dir))
            return false;

        foreach (var name in _tokenizerProbeFiles)
        {
            if (File.Exists(Path.Combine(dir, name)))
                return true;
        }

        // Also accept any *.spm file (used by translation/multilingual models)
        return Directory.EnumerateFiles(dir, "*.spm").Any();
    }

    /// <summary>
    /// Checks if the model identifier refers to a GGUF model.
    /// </summary>
    private static bool IsGgufModel(string modelIdOrPath)
    {
        // Check for "gguf:" prefix
        if (modelIdOrPath.StartsWith("gguf:", StringComparison.OrdinalIgnoreCase))
            return true;

        // Check for .gguf extension
        if (modelIdOrPath.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase))
            return true;

        // Check if local file exists and has .gguf extension
        if (File.Exists(modelIdOrPath) &&
            modelIdOrPath.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase))
            return true;

        // Check for GGUF indicators in HuggingFace repo name
        var lowerPath = modelIdOrPath.ToLowerInvariant();
        if (lowerPath.Contains("-gguf") || lowerPath.Contains("_gguf"))
            return true;

        return false;
    }

    /// <summary>The quantization a GGUF repository load prefers when the memory budget allows it.</summary>
    private const string GgufPreferredQuantization = "Q4_K_M";

    private static string StripGgufPrefix(string modelIdOrPath) =>
        modelIdOrPath.StartsWith("gguf:", StringComparison.OrdinalIgnoreCase) ? modelIdOrPath[5..] : modelIdOrPath;

    private static ModelNotFoundException GgufModelNotFound(string modelIdOrPath) => new(
        $"GGUF model not found: '{modelIdOrPath}'. " +
        "Provide a local path to a .gguf file or a HuggingFace repo ID " +
        "(e.g., 'gguf:nomic-ai/nomic-embed-text-v1.5-GGUF').",
        modelIdOrPath);

    /// <summary>
    /// Loads a GGUF embedding model.
    /// </summary>
    private static async Task<IEmbeddingModel> LoadGgufAsync(
        string modelIdOrPath,
        EmbedderOptions options,
        IProgress<DownloadProgress>? progress,
        CancellationToken cancellationToken)
    {
        string modelPath;
        string modelId;

        var cleanPath = StripGgufPrefix(modelIdOrPath);

        // Check if it's a local file
        if (File.Exists(cleanPath))
        {
            modelPath = cleanPath;
            modelId = Path.GetFileNameWithoutExtension(modelPath);
        }
        // Check if it's a HuggingFace repo ID
        else if (cleanPath.Contains('/'))
        {
            var cacheDir = options.CacheDirectory ?? CacheManager.GetDefaultCacheDirectory();

            // Download the GGUF file from HuggingFace
            using var downloader = new GgufDownloader(cacheDir, localFilesOnly: options.DisableAutoDownload);
            modelPath = await downloader.DownloadAsync(
                cleanPath,
                preferredQuantization: GgufPreferredQuantization,
                progress: progress,
                cancellationToken: cancellationToken);

            modelId = cleanPath.Split('/').Last();
        }
        else
        {
            throw GgufModelNotFound(modelIdOrPath);
        }

        return await LlamaServerEmbeddingModel.LoadAsync(
            modelId,
            modelPath,
            options,
            ResolveGgufPrefixes(cleanPath),
            progress,
            cancellationToken);
    }

    /// <summary>
    /// The prompt prefixes of the model a GGUF repository was converted from. A GGUF file carries no
    /// sentence-transformers prompts, so a repository named "&lt;org&gt;/&lt;model&gt;-GGUF" takes the catalog entry of
    /// "&lt;org&gt;/&lt;model&gt;"; anything else (a local file, an unknown model) has none.
    /// </summary>
    internal static PromptPrefixes ResolveGgufPrefixes(string repoIdOrPath) =>
        TryResolveGgufSourceModel(repoIdOrPath, out var info) ? PromptPrefixes.Of(info!) : PromptPrefixes.None;

    /// <summary>
    /// The catalog entry of the model a GGUF repository was converted from: "&lt;org&gt;/&lt;model&gt;-GGUF" →
    /// "&lt;org&gt;/&lt;model&gt;". False for a local file or a model the catalog does not know.
    /// </summary>
    internal static bool TryResolveGgufSourceModel(string repoIdOrPath, out ModelInfo? info)
    {
        info = null;
        if (!repoIdOrPath.Contains('/') || File.Exists(repoIdOrPath))
            return false;

        foreach (var suffix in new[] { "-GGUF", "_GGUF", ".GGUF" })
        {
            if (!repoIdOrPath.EndsWith(suffix, StringComparison.OrdinalIgnoreCase))
                continue;
            return EmbedderModelRegistry.Default.TryResolveCatalog(repoIdOrPath[..^suffix.Length], out info, out _) && info is not null;
        }
        return false;
    }

    /// <summary>
    /// What <see cref="LoadAsync"/> would load for <paramref name="modelId"/> — the repository, the backend, a name and
    /// the licence — without downloading or loading anything.
    /// </summary>
    /// <remarks>
    /// Follows the load's own resolution: a user alias (which may point at a <c>gguf:</c> repository), then the catalog
    /// (<c>auto</c> included). A GGUF repository the catalog knows the source of (<c>&lt;org&gt;/&lt;model&gt;-GGUF</c>)
    /// reports that model's licence; any other repository or path reports its own name and a <see langword="null"/>
    /// licence. A <c>:variant</c> qualifier is ignored.
    /// </remarks>
    /// <param name="modelId">Anything <see cref="LoadAsync"/> accepts.</param>
    public static ModelDescription Describe(string modelId)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(modelId);
        var target = PrepareLoadTarget(modelId, new EmbedderOptions());

        if (IsGgufModel(target))
        {
            var repo = StripGgufPrefix(target);
            return TryResolveGgufSourceModel(repo, out var source)
                ? new ModelDescription(modelId, repo, ModelBackend.Gguf, $"{OwnName(source!.RepoId)} (GGUF)", source.License, null)
                : new ModelDescription(modelId, repo, ModelBackend.Gguf, OwnName(repo), null, null);
        }

        return EmbedderModelRegistry.Default.TryResolveCatalog(target, out var info, out var resolvedId) && info is not null
            ? new ModelDescription(modelId, info.RepoId, ModelBackend.Onnx, OwnName(info.RepoId), info.License, info)
            : new ModelDescription(modelId, resolvedId, ModelBackend.Onnx, OwnName(resolvedId), null, null);
    }

    /// <summary>The last segment of a repository id or path, as its own name.</summary>
    private static string OwnName(string repoIdOrPath) =>
        repoIdOrPath.TrimEnd('/', '\\').Split('/', '\\')[^1];
}
