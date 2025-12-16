using LMSupply.Download;
using LMSupply.Inference;
using LMSupply.Detector.Models;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace LMSupply.Detector.Core;

/// <summary>
/// ONNX Runtime-based object detector implementation.
/// Supports RT-DETR (NMS-free) and other detection models.
/// </summary>
internal sealed class OnnxDetectorModel : IDetectorModel
{
    private readonly DetectorOptions _options;
    private readonly DetectorModelInfo _modelInfo;
    private readonly SemaphoreSlim _sessionLock = new(1, 1);

    private InferenceSession? _session;
    private bool _isInitialized;
    private bool _disposed;

    /// <inheritdoc />
    public string ModelId => _modelInfo.Id;

    /// <summary>
    /// Gets the COCO class labels.
    /// </summary>
    public IReadOnlyList<string> ClassLabels => CocoLabels.Labels;

    public OnnxDetectorModel(DetectorOptions options)
    {
        _options = options.Clone();
        _modelInfo = DetectorModelRegistry.Default.Resolve(options.ModelId);
    }

    public async Task WarmupAsync(CancellationToken cancellationToken = default)
    {
        await EnsureInitializedAsync(cancellationToken);
    }

    public DetectorModelInfo? GetModelInfo() => _modelInfo;

    public async Task<IReadOnlyList<DetectionResult>> DetectAsync(
        string imagePath,
        CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(imagePath);

        using var image = await Image.LoadAsync<Rgb24>(imagePath, cancellationToken);
        return await DetectCoreAsync(image, cancellationToken);
    }

    public async Task<IReadOnlyList<DetectionResult>> DetectAsync(
        Stream imageStream,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(imageStream);

        using var image = await Image.LoadAsync<Rgb24>(imageStream, cancellationToken);
        return await DetectCoreAsync(image, cancellationToken);
    }

    public async Task<IReadOnlyList<DetectionResult>> DetectAsync(
        byte[] imageData,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(imageData);

        using var image = Image.Load<Rgb24>(imageData);
        return await DetectCoreAsync(image, cancellationToken);
    }

    public async Task<IReadOnlyList<IReadOnlyList<DetectionResult>>> DetectBatchAsync(
        IEnumerable<string> imagePaths,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(imagePaths);

        var results = new List<IReadOnlyList<DetectionResult>>();

        foreach (var path in imagePaths)
        {
            var detections = await DetectAsync(path, cancellationToken);
            results.Add(detections);
        }

        return results;
    }

    private async Task<IReadOnlyList<DetectionResult>> DetectCoreAsync(
        Image<Rgb24> image,
        CancellationToken cancellationToken)
    {
        await EnsureInitializedAsync(cancellationToken);

        var originalWidth = image.Width;
        var originalHeight = image.Height;
        var inputSize = _modelInfo.InputSize;

        // Preprocess image
        var inputTensor = PreprocessImage(image, inputSize);

        // Run inference
        var outputs = await RunInferenceAsync(inputTensor, cancellationToken);

        // Parse detections based on model architecture
        var detections = _modelInfo.RequiresNms
            ? ParseWithNms(outputs, originalWidth, originalHeight, inputSize)
            : ParseNmsFree(outputs, originalWidth, originalHeight, inputSize);

        // Apply confidence threshold and max detections
        var filtered = detections
            .Where(d => d.Confidence >= _options.ConfidenceThreshold)
            .Where(d => _options.ClassFilter == null || _options.ClassFilter.Contains(d.ClassId))
            .OrderByDescending(d => d.Confidence)
            .Take(_options.MaxDetections)
            .ToList();

        return filtered;
    }

    private DenseTensor<float> PreprocessImage(Image<Rgb24> image, int targetSize)
    {
        // Resize to target size
        image.Mutate(x => x.Resize(targetSize, targetSize));

        // Create tensor in NCHW format with ImageNet normalization
        var tensor = new DenseTensor<float>([1, 3, targetSize, targetSize]);
        var mean = new[] { 0.485f, 0.456f, 0.406f };
        var std = new[] { 0.229f, 0.224f, 0.225f };

        image.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < targetSize; y++)
            {
                var row = accessor.GetRowSpan(y);
                for (int x = 0; x < targetSize; x++)
                {
                    var pixel = row[x];
                    tensor[0, 0, y, x] = (pixel.R / 255f - mean[0]) / std[0];
                    tensor[0, 1, y, x] = (pixel.G / 255f - mean[1]) / std[1];
                    tensor[0, 2, y, x] = (pixel.B / 255f - mean[2]) / std[2];
                }
            }
        });

        return tensor;
    }

    private async Task<IDisposableReadOnlyCollection<DisposableNamedOnnxValue>> RunInferenceAsync(
        DenseTensor<float> inputTensor,
        CancellationToken cancellationToken)
    {
        await _sessionLock.WaitAsync(cancellationToken);
        try
        {
            var inputName = _session!.InputNames[0];
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(inputName, inputTensor)
            };

            return _session.Run(inputs);
        }
        finally
        {
            _sessionLock.Release();
        }
    }

    /// <summary>
    /// Parses RT-DETR style output (NMS-free, direct detections).
    /// Output shape: [1, num_queries, 4+num_classes] or [1, num_queries, 6]
    /// </summary>
    private List<DetectionResult> ParseNmsFree(
        IDisposableReadOnlyCollection<DisposableNamedOnnxValue> outputs,
        int originalWidth,
        int originalHeight,
        int inputSize)
    {
        var results = new List<DetectionResult>();

        // RT-DETR outputs: logits [1, 300, num_classes] and boxes [1, 300, 4]
        var outputList = outputs.ToList();

        if (outputList.Count >= 2)
        {
            // Standard RT-DETR format: separate logits and boxes
            var logits = outputList[0].AsTensor<float>();
            var boxes = outputList[1].AsTensor<float>();

            var numQueries = (int)logits.Dimensions[1];
            var numClasses = (int)logits.Dimensions[2];

            for (int i = 0; i < numQueries; i++)
            {
                // Find best class (softmax already applied or use sigmoid)
                float maxScore = float.MinValue;
                int bestClass = 0;

                for (int c = 0; c < numClasses; c++)
                {
                    var score = Sigmoid(logits[0, i, c]);
                    if (score > maxScore)
                    {
                        maxScore = score;
                        bestClass = c;
                    }
                }

                if (maxScore < _options.ConfidenceThreshold)
                    continue;

                // Parse box [cx, cy, w, h] in normalized coordinates
                var cx = boxes[0, i, 0] * originalWidth;
                var cy = boxes[0, i, 1] * originalHeight;
                var w = boxes[0, i, 2] * originalWidth;
                var h = boxes[0, i, 3] * originalHeight;

                var box = BoundingBox.FromCenterSize(cx, cy, w, h)
                    .Clamp(originalWidth, originalHeight);

                results.Add(new DetectionResult(
                    ClassId: bestClass,
                    Label: CocoLabels.GetLabel(bestClass),
                    Confidence: maxScore,
                    Box: box));
            }
        }
        else if (outputList.Count == 1)
        {
            // Combined format: [1, num_queries, 4+num_classes] or [1, num_queries, 6]
            var output = outputList[0].AsTensor<float>();
            var numQueries = (int)output.Dimensions[1];
            var outputDim = (int)output.Dimensions[2];

            if (outputDim == 6)
            {
                // YOLOv10 style: [x1, y1, x2, y2, score, class_id]
                for (int i = 0; i < numQueries; i++)
                {
                    var score = output[0, i, 4];
                    if (score < _options.ConfidenceThreshold)
                        continue;

                    var classId = (int)output[0, i, 5];
                    var scaleX = originalWidth / (float)inputSize;
                    var scaleY = originalHeight / (float)inputSize;

                    var box = new BoundingBox(
                        output[0, i, 0] * scaleX,
                        output[0, i, 1] * scaleY,
                        output[0, i, 2] * scaleX,
                        output[0, i, 3] * scaleY)
                        .Clamp(originalWidth, originalHeight);

                    results.Add(new DetectionResult(
                        ClassId: classId,
                        Label: CocoLabels.GetLabel(classId),
                        Confidence: score,
                        Box: box));
                }
            }
            else
            {
                // Generic format: [cx, cy, w, h, class_scores...]
                var numClasses = outputDim - 4;
                for (int i = 0; i < numQueries; i++)
                {
                    float maxScore = float.MinValue;
                    int bestClass = 0;

                    for (int c = 0; c < numClasses; c++)
                    {
                        var score = output[0, i, 4 + c];
                        if (score > maxScore)
                        {
                            maxScore = score;
                            bestClass = c;
                        }
                    }

                    if (maxScore < _options.ConfidenceThreshold)
                        continue;

                    var cx = output[0, i, 0] * originalWidth;
                    var cy = output[0, i, 1] * originalHeight;
                    var w = output[0, i, 2] * originalWidth;
                    var h = output[0, i, 3] * originalHeight;

                    var box = BoundingBox.FromCenterSize(cx, cy, w, h)
                        .Clamp(originalWidth, originalHeight);

                    results.Add(new DetectionResult(
                        ClassId: bestClass,
                        Label: CocoLabels.GetLabel(bestClass),
                        Confidence: maxScore,
                        Box: box));
                }
            }
        }

        return results;
    }

    /// <summary>
    /// Parses detection output with NMS post-processing (for models that require it).
    /// </summary>
    private List<DetectionResult> ParseWithNms(
        IDisposableReadOnlyCollection<DisposableNamedOnnxValue> outputs,
        int originalWidth,
        int originalHeight,
        int inputSize)
    {
        var allDetections = new List<DetectionResult>();
        var output = outputs.First().AsTensor<float>();

        // Standard YOLO format: [1, num_boxes, 4 + num_classes]
        var numBoxes = (int)output.Dimensions[1];
        var outputDim = (int)output.Dimensions[2];
        var numClasses = outputDim - 4;

        var scaleX = originalWidth / (float)inputSize;
        var scaleY = originalHeight / (float)inputSize;

        for (int i = 0; i < numBoxes; i++)
        {
            float maxScore = float.MinValue;
            int bestClass = 0;

            for (int c = 0; c < numClasses; c++)
            {
                var score = output[0, i, 4 + c];
                if (score > maxScore)
                {
                    maxScore = score;
                    bestClass = c;
                }
            }

            if (maxScore < _options.ConfidenceThreshold)
                continue;

            // Box format: [cx, cy, w, h]
            var cx = output[0, i, 0] * scaleX;
            var cy = output[0, i, 1] * scaleY;
            var w = output[0, i, 2] * scaleX;
            var h = output[0, i, 3] * scaleY;

            var box = BoundingBox.FromCenterSize(cx, cy, w, h)
                .Clamp(originalWidth, originalHeight);

            allDetections.Add(new DetectionResult(
                ClassId: bestClass,
                Label: CocoLabels.GetLabel(bestClass),
                Confidence: maxScore,
                Box: box));
        }

        // Apply NMS
        return ApplyNms(allDetections);
    }

    private List<DetectionResult> ApplyNms(List<DetectionResult> detections)
    {
        var results = new List<DetectionResult>();
        var grouped = detections.GroupBy(d => d.ClassId);

        foreach (var group in grouped)
        {
            var sorted = group.OrderByDescending(d => d.Confidence).ToList();

            while (sorted.Count > 0)
            {
                var best = sorted[0];
                results.Add(best);
                sorted.RemoveAt(0);

                sorted = sorted.Where(d => best.Box.IoU(d.Box) <= _options.IouThreshold).ToList();
            }
        }

        return results;
    }

    private static float Sigmoid(float x) => 1f / (1f + MathF.Exp(-x));

    private async Task EnsureInitializedAsync(CancellationToken cancellationToken)
    {
        if (_isInitialized)
            return;

        await _sessionLock.WaitAsync(cancellationToken);
        try
        {
            if (_isInitialized)
                return;

            var modelPath = await ResolveModelPathAsync(cancellationToken);
            _session = await OnnxSessionFactory.CreateAsync(
                modelPath,
                _options.Provider,
                ConfigureSessionOptions,
                cancellationToken: cancellationToken);

            _isInitialized = true;
        }
        finally
        {
            _sessionLock.Release();
        }
    }

    private async Task<string> ResolveModelPathAsync(CancellationToken cancellationToken)
    {
        // If it's a local path, return directly
        if (File.Exists(_modelInfo.Id))
        {
            return _modelInfo.Id;
        }

        // Download from HuggingFace
        using var downloader = new HuggingFaceDownloader(_options.CacheDirectory);

        var modelDir = await downloader.DownloadModelAsync(
            _modelInfo.Id,
            files: [_modelInfo.OnnxFile, "config.json"],
            cancellationToken: cancellationToken);

        var modelPath = Path.Combine(modelDir, _modelInfo.OnnxFile);

        // Try common ONNX paths if default not found
        if (!File.Exists(modelPath))
        {
            var alternatives = new[] { "model.onnx", "onnx/model.onnx", "model_fp16.onnx", "model_fp32.onnx" };
            foreach (var alt in alternatives)
            {
                var altPath = Path.Combine(modelDir, alt);
                if (File.Exists(altPath))
                {
                    modelPath = altPath;
                    break;
                }
            }
        }

        if (!File.Exists(modelPath))
        {
            throw new FileNotFoundException(
                $"ONNX model not found in {modelDir}. Expected: {_modelInfo.OnnxFile}",
                modelPath);
        }

        return modelPath;
    }

    private void ConfigureSessionOptions(SessionOptions options)
    {
        if (_options.ThreadCount.HasValue)
        {
            options.IntraOpNumThreads = _options.ThreadCount.Value;
            options.InterOpNumThreads = _options.ThreadCount.Value;
        }

        options.EnableMemoryPattern = true;
        options.EnableCpuMemArena = true;
    }

    public async ValueTask DisposeAsync()
    {
        if (_disposed)
            return;

        await _sessionLock.WaitAsync();
        try
        {
            _session?.Dispose();
        }
        finally
        {
            _sessionLock.Release();
            _sessionLock.Dispose();
        }

        _disposed = true;
    }
}
