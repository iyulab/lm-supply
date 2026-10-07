using System.Diagnostics;
using LMSupply.Download;
using LMSupply.Exceptions;
using LMSupply.Ocr.Detection;
using LMSupply.Ocr.Models;
using LMSupply.Ocr.Recognition;

namespace LMSupply.Ocr;

/// <summary>
/// Main entry point for loading and using OCR models.
/// </summary>
public static class LocalOcr
{
    /// <summary>
    /// Gets the detection model registry.
    /// </summary>
    public static IModelRegistry<DetectionModelInfo> DetectionRegistry => OcrDetectionModelRegistry.Default;

    /// <summary>
    /// Shared pool for named model management. Supports GetOrLoadAsync / UnloadAsync by model ID.
    /// The model ID is treated as the detection model ID; the default recognition model is used.
    /// </summary>
    public static LMSupply.Pool.ModelPool<IOcr, OcrOptions> Pool { get; }
        = new(new Pool.OcrLoader());

    /// <summary>
    /// Gets the recognition model registry.
    /// </summary>
    public static IModelRegistry<RecognitionModelInfo> RecognitionRegistry => OcrRecognitionModelRegistry.Default;

    /// <summary>
    /// Loads an OCR pipeline with the specified detection and recognition models.
    /// </summary>
    /// <param name="detectionModel">
    /// Detection model ID (e.g., "default", "dbnet-v3") for auto-download,
    /// or a local path to a model file.
    /// </param>
    /// <param name="recognitionModel">
    /// Recognition model ID (e.g., "default", "crnn-en-v3", "crnn-korean-v3") for auto-download,
    /// or a local path to a model file. If null, uses the model for the language hint.
    /// </param>
    /// <param name="options">Optional configuration options.</param>
    /// <param name="progress">Optional progress reporting for downloads.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>A loaded OCR pipeline ready for inference.</returns>
    public static async Task<IOcr> LoadAsync(
        string detectionModel = "default",
        string? recognitionModel = null,
        OcrOptions? options = null,
        IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(detectionModel);
        options = options?.Clone() ?? new OcrOptions();
        // An unsupported provider is refused here, before any model resolution or download (0.67.1).
        ExecutionProviderSupport.ThrowIfUnsupported(options.Provider);

        // Parse variant qualifier (e.g., "default:fp16" → detectionModel="default", hint="fp16")
        var (baseDetId, detQualifier) = LMSupplyOptionsBase.SplitQualifier(detectionModel);
        detectionModel = baseDetId;
        options.QuantizationHint ??= detQualifier;

        if (recognitionModel is not null)
        {
            var (baseRecId, recQualifier) = LMSupplyOptionsBase.SplitQualifier(recognitionModel);
            recognitionModel = baseRecId;
            options.QuantizationHint ??= recQualifier;
        }

        // Resolve detection model
        var (detModelInfo, detModelPath) = await ResolveDetectionModelAsync(
            detectionModel, options, progress, cancellationToken).ConfigureAwait(false);

        // Resolve recognition model based on language hint if not specified
        recognitionModel ??= OcrRecognitionModelRegistry.Default.ResolveForLanguage(options.LanguageHint).AliasName;

        var (recModelInfo, recModelPath, dictPath) = await ResolveRecognitionModelAsync(
            recognitionModel, options, progress, cancellationToken).ConfigureAwait(false);

        // Create detector and recognizer
        var detector = await DbNetDetector.CreateAsync(detModelPath, detModelInfo, options)
            .ConfigureAwait(false);

        var recognizer = await CrnnRecognizer.CreateAsync(recModelPath, dictPath, recModelInfo, options)
            .ConfigureAwait(false);

        // Create and return pipeline
        return await OcrPipeline.CreateAsync(detector, recognizer, detModelInfo, recModelInfo, detModelPath, recModelPath)
            .ConfigureAwait(false);
    }

    /// <summary>
    /// Loads an OCR pipeline for a specific language.
    /// </summary>
    /// <param name="languageCode">ISO language code (e.g., "en", "ko", "zh", "ja").</param>
    /// <param name="options">Optional configuration options.</param>
    /// <param name="progress">Optional progress reporting for downloads.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>A loaded OCR pipeline ready for inference.</returns>
    public static async Task<IOcr> LoadForLanguageAsync(
        string languageCode,
        OcrOptions? options = null,
        IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        options = options?.Clone() ?? new OcrOptions { LanguageHint = languageCode };
        // An unsupported provider is refused here, before any model resolution or download (0.67.1).
        ExecutionProviderSupport.ThrowIfUnsupported(options.Provider);
        options.LanguageHint = languageCode;

        var recognitionModel = OcrRecognitionModelRegistry.Default.ResolveForLanguage(languageCode).AliasName;
        return await LoadAsync(DefaultDetectionModel, recognitionModel, options, progress, cancellationToken).ConfigureAwait(false);
    }

    /// <summary>
    /// Reports whether the model files <see cref="LoadForLanguageAsync"/> needs for a language are in the
    /// local cache — whether loading it would download anything. Makes no network request.
    /// </summary>
    /// <remarks>
    /// The files are the ones the loader fetches — the default detection model, plus the language's
    /// recognizer and its dictionary — checked with the downloader's own cache test. A language shares its
    /// recognizer with others of the same script, so it can be cached without having been loaded itself.
    /// To make a load fail instead of downloading, set <see cref="OcrOptions.DisableAutoDownload"/>.
    /// </remarks>
    /// <param name="languageCode">ISO language code (e.g., "en", "ko", "zh", "ja").</param>
    /// <param name="cacheDirectory">The cache directory; the default HuggingFace cache when null.</param>
    /// <returns>Each needed file and whether it is cached.</returns>
    public static OcrCacheStatus GetCacheStatusForLanguage(string languageCode, string? cacheDirectory = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(languageCode);
        var cacheDir = cacheDirectory ?? CacheManager.GetDefaultCacheDirectory();

        var detection = OcrDetectionModelRegistry.Default.Resolve(DefaultDetectionModel);
        var recognition = OcrRecognitionModelRegistry.Default.ResolveForLanguage(languageCode);

        OcrModelFile[] files =
        [
            .. Probe(cacheDir, detection.RepoId, detection.Subfolder, RequiredFiles(detection)),
            .. Probe(cacheDir, recognition.RepoId, recognition.Subfolder, RequiredFiles(recognition)),
        ];

        return new OcrCacheStatus(languageCode, recognition.AliasName, files);
    }

    /// <summary>
    /// Bytes <see cref="LoadForLanguageAsync"/> would download for the same language and options into an empty cache:
    /// the default detection model, plus the language's recognizer and its dictionary, at the lengths the repositories
    /// list. For a consent screen that states what a first run will fetch.
    /// </summary>
    /// <remarks>
    /// The files are the ones the loader fetches (the same lists <see cref="GetCacheStatusForLanguage"/> probes). Reads
    /// the repository listings (cached for a day and reused by the download that follows); downloads and loads nothing.
    /// The figure is the whole download whatever the cache already holds — <see cref="GetCacheStatusForLanguage"/>
    /// answers what is present. The native ONNX Runtime a first load also provisions, once per host and shared by every
    /// model, is not counted. With <see cref="OcrOptions.DisableAutoDownload"/> the listings come from the cache only,
    /// as the load's do.
    /// </remarks>
    /// <param name="languageCode">ISO language code (e.g., "en", "ko", "zh", "ja").</param>
    /// <param name="options">The options the load will use (<see cref="LMSupplyOptionsBase.CacheDirectory"/>, <see cref="OcrOptions.DisableAutoDownload"/>); not modified.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <exception cref="ModelNotFoundException">A repository does not exist, or (downloads disabled) was never listed into this cache.</exception>
    /// <exception cref="ModelDownloadException">A file the load needs is not in its repository.</exception>
    public static async Task<long> GetDownloadSizeBytesAsync(
        string languageCode,
        OcrOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(languageCode);
        options = options?.Clone() ?? new OcrOptions();
        ExecutionProviderSupport.ThrowIfUnsupported(options.Provider);

        var plans = await PlanDownloadsAsync(languageCode, options, cancellationToken).ConfigureAwait(false);
        return plans.Sum(p => p.TotalBytes);
    }

    /// <summary>
    /// Bytes a load with the same language and options would still download now: the files of
    /// <see cref="GetDownloadSizeBytesAsync"/> that the cache does not hold at the length the repository lists. 0 when
    /// the model is cached or on local disk — for deciding whether to ask the user at all.
    /// </summary>
    /// <remarks>
    /// Reads the repository listing as <see cref="GetDownloadSizeBytesAsync"/> does, and the cache; downloads nothing. A
    /// partly downloaded file counts in full. Runtimes are not counted.
    /// </remarks>
    public static async Task<long> GetRemainingDownloadBytesAsync(
        string languageCode,
        OcrOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(languageCode);
        options = options?.Clone() ?? new OcrOptions();
        ExecutionProviderSupport.ThrowIfUnsupported(options.Provider);

        var plans = await PlanDownloadsAsync(languageCode, options, cancellationToken).ConfigureAwait(false);
        var cacheDir = options.CacheDirectory ?? CacheManager.GetDefaultCacheDirectory();
        return plans.Sum(p => p.GetRemainingBytes(cacheDir));
    }

    // The plans a load with these options fetches; empty for a model on local disk. Shared by the total and the
    // remaining figure, so both count the files the load picks.
    private static async Task<IReadOnlyList<DownloadPlan>> PlanDownloadsAsync(
        string languageCode, OcrOptions options, CancellationToken cancellationToken)
    {
        var cacheDir = options.CacheDirectory ?? CacheManager.GetDefaultCacheDirectory();
        var detection = OcrDetectionModelRegistry.Default.Resolve(DefaultDetectionModel);
        var recognition = OcrRecognitionModelRegistry.Default.ResolveForLanguage(languageCode);

        using var downloader = new HuggingFaceDownloader(cacheDir, localFilesOnly: options.DisableAutoDownload);
        var detectionPlan = await downloader.PlanModelAsync(
            detection.RepoId, RequiredFiles(detection), subfolder: detection.Subfolder, cancellationToken: cancellationToken)
            .ConfigureAwait(false);
        var recognitionPlan = await downloader.PlanModelAsync(
            recognition.RepoId, RequiredFiles(recognition), subfolder: recognition.Subfolder, cancellationToken: cancellationToken)
            .ConfigureAwait(false);

        return [detectionPlan, recognitionPlan];
    }

    // The detection model a language load uses. The probe and the loader share it so they cannot disagree.
    private const string DefaultDetectionModel = "default";

    // The files a registered model is loaded from — what the loader downloads and what the probe checks.
    private static string[] RequiredFiles(DetectionModelInfo model) => [model.ModelFile];

    private static string[] RequiredFiles(RecognitionModelInfo model) => [model.ModelFile, model.DictFile];

    private static IEnumerable<OcrModelFile> Probe(string cacheDir, string repoId, string? subfolder, string[] files)
    {
        var missing = CacheManager.GetMissingFiles(cacheDir, repoId, files, subfolder);
        return files.Select(file => new OcrModelFile(repoId, subfolder, file, !missing.Contains(file)));
    }

    /// <summary>
    /// Gets a list of pre-configured detection model IDs available for download.
    /// </summary>
    public static IEnumerable<string> GetAvailableDetectionModels()
        => OcrDetectionModelRegistry.Default.GetAvailableModels().Select(m => m.AliasName).Distinct().Order();

    /// <summary>
    /// Gets a list of pre-configured recognition model IDs available for download.
    /// </summary>
    public static IEnumerable<string> GetAvailableRecognitionModels()
        => OcrRecognitionModelRegistry.Default.GetAvailableModels().Select(m => m.AliasName).Distinct().Order();

    /// <summary>
    /// Gets a list of supported language codes.
    /// </summary>
    public static IEnumerable<string> GetSupportedLanguages()
        => OcrRecognitionModelRegistry.Default.GetSupportedLanguages();

    private static async Task<(DetectionModelInfo info, string modelPath)> ResolveDetectionModelAsync(
        string modelIdOrPath,
        OcrOptions options,
        IProgress<DownloadProgress>? progress,
        CancellationToken cancellationToken)
    {
        // Check if it's a local file path
        if (File.Exists(modelIdOrPath))
        {
            // Try to find matching model info or create a default one
            var modelInfo = OcrDetectionModelRegistry.Default.TryResolve("default", out var info)
                ? info!
                : throw new ModelNotFoundException("No default detection model configured", modelIdOrPath);

            return (modelInfo, modelIdOrPath);
        }

        // Check if it's a known model alias
        if (OcrDetectionModelRegistry.Default.TryResolve(modelIdOrPath, out var knownModel) && knownModel is not null)
        {
            var cacheDir = options.CacheDirectory ?? CacheManager.GetDefaultCacheDirectory();
            using var downloader = new HuggingFaceDownloader(cacheDir, localFilesOnly: options.DisableAutoDownload);

            var modelDir = await downloader.DownloadModelAsync(
                knownModel.RepoId,
                files: RequiredFiles(knownModel),
                subfolder: knownModel.Subfolder,
                progress: progress,
                cancellationToken: cancellationToken).ConfigureAwait(false);

            var modelPath = Path.Combine(modelDir, knownModel.ModelFile);

            if (!File.Exists(modelPath))
            {
                throw new ModelNotFoundException(
                    $"Detection model file not found: {knownModel.ModelFile}",
                    modelIdOrPath);
            }

            return (knownModel, modelPath);
        }

        // Check if it's a HuggingFace repo ID (contains '/')
        if (modelIdOrPath.Contains('/'))
        {
            return await ResolveHuggingFaceDetectionModelAsync(
                modelIdOrPath, options, progress, cancellationToken).ConfigureAwait(false);
        }

        throw new ModelNotFoundException(
            $"Unknown detection model '{modelIdOrPath}'. Use GetAvailableDetectionModels() to list available models, or provide a HuggingFace repo ID.",
            modelIdOrPath);
    }

    private static async Task<(RecognitionModelInfo info, string modelPath, string dictPath)> ResolveRecognitionModelAsync(
        string modelIdOrPath,
        OcrOptions options,
        IProgress<DownloadProgress>? progress,
        CancellationToken cancellationToken)
    {
        // Check if it's a local file path
        if (File.Exists(modelIdOrPath))
        {
            // Try to find matching model info or create a default one
            var modelInfo = OcrRecognitionModelRegistry.Default.TryResolve("default", out var info)
                ? info!
                : throw new ModelNotFoundException("No default recognition model configured", modelIdOrPath);

            // Look for dictionary file in the same directory
            var dictPath = Path.Combine(Path.GetDirectoryName(modelIdOrPath)!, modelInfo.DictFile);
            if (!File.Exists(dictPath))
            {
                throw new ModelNotFoundException(
                    $"Dictionary file not found: {modelInfo.DictFile}",
                    modelIdOrPath);
            }

            return (modelInfo, modelIdOrPath, dictPath);
        }

        // Check if it's a known model alias
        if (OcrRecognitionModelRegistry.Default.TryResolve(modelIdOrPath, out var knownModel) && knownModel is not null)
        {
            var cacheDir = options.CacheDirectory ?? CacheManager.GetDefaultCacheDirectory();
            using var downloader = new HuggingFaceDownloader(cacheDir, localFilesOnly: options.DisableAutoDownload);

            var modelDir = await downloader.DownloadModelAsync(
                knownModel.RepoId,
                files: RequiredFiles(knownModel),
                subfolder: knownModel.Subfolder,
                progress: progress,
                cancellationToken: cancellationToken).ConfigureAwait(false);

            var modelPath = Path.Combine(modelDir, knownModel.ModelFile);
            var dictPath = Path.Combine(modelDir, knownModel.DictFile);

            if (!File.Exists(modelPath))
            {
                throw new ModelNotFoundException(
                    $"Recognition model file not found: {knownModel.ModelFile}",
                    modelIdOrPath);
            }

            if (!File.Exists(dictPath))
            {
                throw new ModelNotFoundException(
                    $"Dictionary file not found: {knownModel.DictFile}",
                    modelIdOrPath);
            }

            return (knownModel, modelPath, dictPath);
        }

        // Check if it's a HuggingFace repo ID (contains '/')
        if (modelIdOrPath.Contains('/'))
        {
            return await ResolveHuggingFaceRecognitionModelAsync(
                modelIdOrPath, options, progress, cancellationToken).ConfigureAwait(false);
        }

        throw new ModelNotFoundException(
            $"Unknown recognition model '{modelIdOrPath}'. Use GetAvailableRecognitionModels() to list available models, or provide a HuggingFace repo ID.",
            modelIdOrPath);
    }

    /// <summary>
    /// Resolves a detection model from a HuggingFace repository.
    /// Searches for common detection model patterns (det.onnx, detection.onnx, etc.)
    /// </summary>
    private static async Task<(DetectionModelInfo info, string modelPath)> ResolveHuggingFaceDetectionModelAsync(
        string repoId,
        OcrOptions options,
        IProgress<DownloadProgress>? progress,
        CancellationToken cancellationToken)
    {
        var cacheDir = options.CacheDirectory ?? CacheManager.GetDefaultCacheDirectory();
        using var downloader = new HuggingFaceDownloader(cacheDir, localFilesOnly: options.DisableAutoDownload);

        // Common detection model file patterns
        string[] detectionPatterns = ["det.onnx", "detection.onnx", "text_detection.onnx", "detector.onnx"];
        string[] subfolderPatterns = ["", "detection", "detection/v5", "detection/v3", "onnx"];

        string? modelPath = null;
        string? foundSubfolder = null;

        // Try to find detection model
        foreach (var subfolder in subfolderPatterns)
        {
            foreach (var pattern in detectionPatterns)
            {
                try
                {
                    var modelDir = await downloader.DownloadModelAsync(
                        repoId,
                        files: [pattern],
                        subfolder: string.IsNullOrEmpty(subfolder) ? null : subfolder,
                        progress: progress,
                        cancellationToken: cancellationToken).ConfigureAwait(false);

                    var candidatePath = Path.Combine(modelDir, pattern);
                    if (File.Exists(candidatePath))
                    {
                        modelPath = candidatePath;
                        foundSubfolder = subfolder;
                        break;
                    }
                }
                catch (Exception ex) when (ex is not OperationCanceledException || !cancellationToken.IsCancellationRequested)
                {
                    Trace.TraceInformation($"[LocalOcr] Model file check failed: {ex.Message}");
                }
            }
            if (modelPath != null) break;
        }

        if (modelPath == null)
        {
            throw new ModelNotFoundException(
                $"No detection model found in HuggingFace repository '{repoId}'. " +
                $"Expected one of: {string.Join(", ", detectionPatterns)}" + OfflineNote(options),
                repoId);
        }

        // Create model info for the discovered model
        var modelInfo = new DetectionModelInfo(
            RepoId: repoId,
            AliasName: repoId,
            DisplayName: $"HuggingFace: {repoId}",
            ModelFile: Path.GetFileName(modelPath),
            InputWidth: 960,
            InputHeight: 960)
        {
            Subfolder = foundSubfolder
        };

        return (modelInfo, modelPath);
    }

    /// <summary>
    /// Resolves a recognition model from a HuggingFace repository.
    /// Searches for common recognition model patterns (rec.onnx, recognition.onnx, etc.)
    /// </summary>
    private static async Task<(RecognitionModelInfo info, string modelPath, string dictPath)> ResolveHuggingFaceRecognitionModelAsync(
        string repoId,
        OcrOptions options,
        IProgress<DownloadProgress>? progress,
        CancellationToken cancellationToken)
    {
        var cacheDir = options.CacheDirectory ?? CacheManager.GetDefaultCacheDirectory();
        using var downloader = new HuggingFaceDownloader(cacheDir, localFilesOnly: options.DisableAutoDownload);

        // Parse repo ID for language hint (e.g., "org/paddleocr-japanese" -> try japanese subfolder)
        var repoName = repoId.Split('/').Last().ToLowerInvariant();

        // Common recognition model file patterns
        string[] recognitionPatterns = ["rec.onnx", "recognition.onnx", "text_recognition.onnx", "recognizer.onnx"];
        string[] dictPatterns = ["dict.txt", "dictionary.txt", "keys.txt", "vocab.txt"];

        // Build subfolder patterns based on language hint
        var subfolderPatterns = new List<string> { "" };
        if (options.LanguageHint != null)
        {
            var langSubfolders = GetLanguageSubfolders(options.LanguageHint);
            subfolderPatterns.InsertRange(0, langSubfolders);
        }
        subfolderPatterns.AddRange(["languages/english", "languages/latin", "onnx", "recognition"]);

        string? modelPath = null;
        string? dictPath = null;
        string? foundSubfolder = null;

        // Try to find recognition model and dictionary
        foreach (var subfolder in subfolderPatterns.Distinct())
        {
            foreach (var recPattern in recognitionPatterns)
            {
                foreach (var dictPattern in dictPatterns)
                {
                    try
                    {
                        var modelDir = await downloader.DownloadModelAsync(
                            repoId,
                            files: [recPattern, dictPattern],
                            subfolder: string.IsNullOrEmpty(subfolder) ? null : subfolder,
                            progress: progress,
                            cancellationToken: cancellationToken).ConfigureAwait(false);

                        var candidateModelPath = Path.Combine(modelDir, recPattern);
                        var candidateDictPath = Path.Combine(modelDir, dictPattern);

                        if (File.Exists(candidateModelPath) && File.Exists(candidateDictPath))
                        {
                            modelPath = candidateModelPath;
                            dictPath = candidateDictPath;
                            foundSubfolder = subfolder;
                            break;
                        }
                    }
                    catch (Exception ex) when (ex is not OperationCanceledException || !cancellationToken.IsCancellationRequested)
                    {
                        Trace.TraceInformation($"[LocalOcr] Recognition file check failed: {ex.Message}");
                    }
                }
                if (modelPath != null) break;
            }
            if (modelPath != null) break;
        }

        if (modelPath == null || dictPath == null)
        {
            throw new ModelNotFoundException(
                $"No recognition model found in HuggingFace repository '{repoId}'. " +
                $"Expected model file ({string.Join(", ", recognitionPatterns)}) and dictionary file ({string.Join(", ", dictPatterns)})" + OfflineNote(options),
                repoId);
        }

        // Create model info for the discovered model
        var modelInfo = new RecognitionModelInfo(
            RepoId: repoId,
            AliasName: repoId,
            DisplayName: $"HuggingFace: {repoId}",
            ModelFile: Path.GetFileName(modelPath),
            DictFile: Path.GetFileName(dictPath),
            LanguageCodes: [options.LanguageHint ?? "en"])
        {
            Subfolder = foundSubfolder
        };

        return (modelInfo, modelPath, dictPath);
    }

    // Repository searches try many file names and swallow each miss; offline, "not found" means "not cached".
    private static string OfflineNote(OcrOptions options) => options.DisableAutoDownload
        ? ". Downloads are disabled (DisableAutoDownload), so only the local cache was searched."
        : string.Empty;

    /// <summary>
    /// Gets possible subfolder names for a language code.
    /// </summary>
    private static string[] GetLanguageSubfolders(string languageCode)
    {
        var lang = languageCode.ToLowerInvariant().Split('-')[0];
        return lang switch
        {
            "en" => ["languages/english", "english", "en"],
            "ko" => ["languages/korean", "korean", "ko"],
            "zh" => ["languages/chinese", "chinese", "zh", "ch"],
            "ja" => ["languages/japanese", "japanese", "ja", "japan"],
            "ar" => ["languages/arabic", "arabic", "ar"],
            "ru" => ["languages/cyrillic", "languages/eslav", "cyrillic", "russian", "ru"],
            "de" or "fr" or "es" or "it" or "pt" => ["languages/latin", "latin"],
            "hi" => ["languages/hindi", "languages/devanagari", "hindi", "devanagari"],
            "th" => ["languages/thai", "thai", "th"],
            "el" => ["languages/greek", "greek", "el"],
            "ta" => ["languages/tamil", "tamil", "ta"],
            "te" => ["languages/telugu", "telugu", "te"],
            _ => [$"languages/{lang}", lang]
        };
    }
}
