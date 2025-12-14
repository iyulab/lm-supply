using LocalAI.Download;
using LocalAI.Exceptions;
using LocalAI.Ocr.Detection;
using LocalAI.Ocr.Models;
using LocalAI.Ocr.Recognition;

namespace LocalAI.Ocr;

/// <summary>
/// Main entry point for loading and using OCR models.
/// </summary>
public static class LocalOcr
{
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
        options ??= new OcrOptions();

        // Resolve detection model
        var (detModelInfo, detModelPath) = await ResolveDetectionModelAsync(
            detectionModel, options, progress, cancellationToken).ConfigureAwait(false);

        // Resolve recognition model based on language hint if not specified
        recognitionModel ??= ModelRegistry.GetRecognitionModelForLanguage(options.LanguageHint).Alias;

        var (recModelInfo, recModelPath, dictPath) = await ResolveRecognitionModelAsync(
            recognitionModel, options, progress, cancellationToken).ConfigureAwait(false);

        // Create detector and recognizer
        var detector = await DbNetDetector.CreateAsync(detModelPath, detModelInfo, options)
            .ConfigureAwait(false);

        var recognizer = await CrnnRecognizer.CreateAsync(recModelPath, dictPath, recModelInfo, options)
            .ConfigureAwait(false);

        // Create and return pipeline
        return await OcrPipeline.CreateAsync(detector, recognizer, detModelInfo, recModelInfo)
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
        options ??= new OcrOptions { LanguageHint = languageCode };
        options.LanguageHint = languageCode;

        var recognitionModel = ModelRegistry.GetRecognitionModelForLanguage(languageCode).Alias;
        return await LoadAsync("default", recognitionModel, options, progress, cancellationToken).ConfigureAwait(false);
    }

    /// <summary>
    /// Gets a list of pre-configured detection model IDs available for download.
    /// </summary>
    public static IEnumerable<string> GetAvailableDetectionModels()
        => ModelRegistry.GetAvailableDetectionModels();

    /// <summary>
    /// Gets a list of pre-configured recognition model IDs available for download.
    /// </summary>
    public static IEnumerable<string> GetAvailableRecognitionModels()
        => ModelRegistry.GetAvailableRecognitionModels();

    /// <summary>
    /// Gets a list of supported language codes.
    /// </summary>
    public static IEnumerable<string> GetSupportedLanguages()
        => ModelRegistry.GetSupportedLanguages();

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
            var modelInfo = ModelRegistry.TryGetDetectionModel("default", out var info)
                ? info
                : throw new ModelNotFoundException("No default detection model configured", modelIdOrPath);

            return (modelInfo, modelIdOrPath);
        }

        // Check if it's a known model alias
        if (ModelRegistry.TryGetDetectionModel(modelIdOrPath, out var knownModel))
        {
            var cacheDir = options.CacheDirectory ?? CacheManager.GetDefaultCacheDirectory();
            using var downloader = new HuggingFaceDownloader(cacheDir);

            var modelDir = await downloader.DownloadModelAsync(
                knownModel.RepoId,
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

        throw new ModelNotFoundException(
            $"Unknown detection model '{modelIdOrPath}'. Use GetAvailableDetectionModels() to list available models.",
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
            var modelInfo = ModelRegistry.TryGetRecognitionModel("default", out var info)
                ? info
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
        if (ModelRegistry.TryGetRecognitionModel(modelIdOrPath, out var knownModel))
        {
            var cacheDir = options.CacheDirectory ?? CacheManager.GetDefaultCacheDirectory();
            using var downloader = new HuggingFaceDownloader(cacheDir);

            var modelDir = await downloader.DownloadModelAsync(
                knownModel.RepoId,
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

        throw new ModelNotFoundException(
            $"Unknown recognition model '{modelIdOrPath}'. Use GetAvailableRecognitionModels() to list available models.",
            modelIdOrPath);
    }
}
