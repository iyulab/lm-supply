using System.Diagnostics;
using LMSupply.Ocr.Detection;
using LMSupply.Ocr.Models;
using LMSupply.Ocr.Recognition;
using LMSupply.Vision;

namespace LMSupply.Ocr;

/// <summary>
/// OCR pipeline that orchestrates text detection and recognition.
/// </summary>
internal sealed class OcrPipeline : IOcr
{
    private readonly DbNetDetector _detector;
    private readonly CrnnRecognizer _recognizer;
    private readonly DetectionModelInfo _detectionModel;
    private readonly RecognitionModelInfo _recognitionModel;
    private readonly string? _detModelPath;
    private readonly string? _recModelPath;
    private bool _disposed;

    private OcrPipeline(
        DbNetDetector detector,
        CrnnRecognizer recognizer,
        DetectionModelInfo detectionModel,
        RecognitionModelInfo recognitionModel,
        string? detModelPath = null,
        string? recModelPath = null)
    {
        _detector = detector;
        _recognizer = recognizer;
        _detectionModel = detectionModel;
        _recognitionModel = recognitionModel;
        _detModelPath = detModelPath;
        _recModelPath = recModelPath;
    }

    /// <inheritdoc />
    public bool IsGpuActive => _detector.IsGpuActive;

    /// <inheritdoc />
    public IReadOnlyList<string> ActiveProviders => _detector.ActiveProviders;

    /// <inheritdoc />
    public ExecutionProvider RequestedProvider => _detector.RequestedProvider;

    /// <inheritdoc />
    public long? EstimatedMemoryBytes
    {
        get
        {
            long total = 0;
            if (_detModelPath is not null && File.Exists(_detModelPath))
                total += new FileInfo(_detModelPath).Length;
            if (_recModelPath is not null && File.Exists(_recModelPath))
                total += new FileInfo(_recModelPath).Length;
            return total > 0 ? total * 2 : null;
        }
    }

    /// <summary>
    /// Creates a new OCR pipeline instance.
    /// </summary>
    public static async Task<OcrPipeline> CreateAsync(
        DbNetDetector detector,
        CrnnRecognizer recognizer,
        DetectionModelInfo detectionModel,
        RecognitionModelInfo recognitionModel,
        string? detModelPath = null,
        string? recModelPath = null)
    {
        return await Task.FromResult(new OcrPipeline(
            detector,
            recognizer,
            detectionModel,
            recognitionModel,
            detModelPath,
            recModelPath)).ConfigureAwait(false);
    }

    /// <inheritdoc />
    public string DetectionModelId => _detectionModel.AliasName;

    /// <inheritdoc />
    public string RecognitionModelId => _recognitionModel.AliasName;

    /// <inheritdoc />
    public IReadOnlyList<string> SupportedLanguages => _recognizer.SupportedLanguages;

    /// <inheritdoc />
    public Task WarmupAsync(CancellationToken cancellationToken = default)
    {
        // The detector and recognizer sessions are already loaded
        // This method ensures any lazy initialization is done
        return Task.CompletedTask;
    }

    /// <inheritdoc />
    public OcrModelInfo? GetModelInfo() => new OcrModelInfo(_detectionModel, _recognitionModel);

    /// <inheritdoc />
    public async Task<OcrResult> RecognizeAsync(string imagePath, CancellationToken cancellationToken = default)
    {
        var sw = Stopwatch.StartNew();

        var image = await DbNetDetector.LoadImageAsync(imagePath, cancellationToken).ConfigureAwait(false);
        var result = await RecognizeImageAsync(image, cancellationToken).ConfigureAwait(false);

        sw.Stop();
        return result with { ProcessingTimeMs = sw.Elapsed.TotalMilliseconds };
    }

    /// <inheritdoc />
    public async Task<OcrResult> RecognizeAsync(Stream imageStream, CancellationToken cancellationToken = default)
    {
        var sw = Stopwatch.StartNew();

        var image = await DbNetDetector.LoadImageAsync(imageStream, cancellationToken).ConfigureAwait(false);
        var result = await RecognizeImageAsync(image, cancellationToken).ConfigureAwait(false);

        sw.Stop();
        return result with { ProcessingTimeMs = sw.Elapsed.TotalMilliseconds };
    }

    /// <inheritdoc />
    public async Task<OcrResult> RecognizeAsync(byte[] imageData, CancellationToken cancellationToken = default)
    {
        var sw = Stopwatch.StartNew();

        var image = DbNetDetector.LoadImage(imageData);
        var result = await RecognizeImageAsync(image, cancellationToken).ConfigureAwait(false);

        sw.Stop();
        return result with { ProcessingTimeMs = sw.Elapsed.TotalMilliseconds };
    }

    /// <inheritdoc />
    public async Task<IReadOnlyList<DetectedRegion>> DetectAsync(string imagePath, CancellationToken cancellationToken = default)
    {
        return await _detector.DetectAsync(imagePath, cancellationToken).ConfigureAwait(false);
    }

    /// <inheritdoc />
    public async Task<IReadOnlyList<DetectedRegion>> DetectAsync(Stream imageStream, CancellationToken cancellationToken = default)
    {
        return await _detector.DetectAsync(imageStream, cancellationToken).ConfigureAwait(false);
    }

    private async Task<OcrResult> RecognizeImageAsync(RgbImage image, CancellationToken cancellationToken)
    {
        // Step 1: Detect text regions
        var detectedRegions = await _detector.DetectAsync(image, cancellationToken).ConfigureAwait(false);

        if (detectedRegions.Count == 0)
        {
            return new OcrResult([], 0);
        }

        // Step 2: Recognize text in each region
        var textRegions = await _recognizer.RecognizeAsync(image, detectedRegions, cancellationToken)
            .ConfigureAwait(false);

        return new OcrResult(textRegions, 0);
    }

    /// <inheritdoc />
    public ValueTask DisposeAsync()
    {
        if (_disposed) return ValueTask.CompletedTask;

        _detector.Dispose();
        _recognizer.Dispose();
        _disposed = true;

        return ValueTask.CompletedTask;
    }
}
