using LMSupply.Core.Download;
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
    private readonly DetectorOutputLayout _layout;
    private readonly DetectorInputFormat _inputFormat;

    // Generated once: the priors are a pure function of the declared input size, and regenerating 4385 of
    // them per frame would cost more than the inference does.
    private readonly Lazy<LpdPrior[]> _platePriors;
    private readonly SemaphoreSlim _sessionLock = new(1, 1);

    // Owns the ONNX session and recovers from a crashing or hanging execution provider by moving
    // to the next one in the fallback chain (see RecoverableOnnxSession).
    private RecoverableOnnxSession? _session;
    private bool _isInitialized;
    private bool _disposed;

    /// <inheritdoc />
    public string ModelId => _modelInfo.Id;

    /// <inheritdoc />
    public bool IsGpuActive => _session?.IsGpuActive ?? false;

    /// <inheritdoc />
    public IReadOnlyList<string> ActiveProviders => _session?.ActiveProviders ?? Array.Empty<string>();

    /// <inheritdoc />
    public ExecutionProvider RequestedProvider => _options.Provider;

    /// <summary>Keypoints this model's layout carries, from the one place that mapping is stated.</summary>
    private int KeypointCount => _layout.KeypointCount();

    /// <inheritdoc />
    public long? EstimatedMemoryBytes => _modelInfo.SizeBytes * 2;

    /// <summary>
    /// Gets the class labels of the loaded model — its own vocabulary, not COCO's by assumption.
    /// </summary>
    public IReadOnlyList<string> ClassLabels => _modelInfo.ClassLabels;

    public OnnxDetectorModel(DetectorOptions options)
    {
        _options = options.Clone();
        _modelInfo = DetectorModelRegistry.Default.Resolve(options.ModelId);
        _layout = options.OutputLayout ?? _modelInfo.OutputLayout;
        _inputFormat = options.InputFormat ?? _modelInfo.InputFormat;
        _platePriors = new Lazy<LpdPrior[]>(() =>
            LpdYuNetDecoder.GeneratePriors(_modelInfo.InputWidth, _modelInfo.InputHeight));
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

    public async Task<IReadOnlyList<IReadOnlyList<DetectionResult>>> DetectBatchAsync(
        IEnumerable<byte[]> imageDataList,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(imageDataList);

        var results = new List<IReadOnlyList<DetectionResult>>();

        foreach (var imageData in imageDataList)
        {
            var detections = await DetectAsync(imageData, cancellationToken);
            results.Add(detections);
        }

        return results;
    }

    public async Task<IReadOnlyList<IReadOnlyList<DetectionResult>>> DetectBatchAsync(
        IEnumerable<Stream> imageStreams,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(imageStreams);

        var results = new List<IReadOnlyList<DetectionResult>>();

        foreach (var stream in imageStreams)
        {
            var detections = await DetectAsync(stream, cancellationToken);
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
        var inputWidth = _modelInfo.InputWidth;
        var inputHeight = _modelInfo.InputHeight;

        // Preprocess image the way this particular model was exported to expect
        var inputTensor = PreprocessImage(image, inputWidth, inputHeight, _inputFormat);

        // Run inference
        var outputs = await RunInferenceAsync(inputTensor, originalWidth, originalHeight, cancellationToken);

        // Decode according to the layout this model declares. Selecting on the declared layout rather than
        // on a pair of booleans is what keeps a three-stride head from being read as a single pose tensor.
        var detections = _layout switch
        {
            DetectorOutputLayout.RtDetr => ParseNmsFree(outputs, originalWidth, originalHeight, inputWidth, inputHeight),
            DetectorOutputLayout.YoloDetect => ParseWithNms(outputs, originalWidth, originalHeight, inputWidth, inputHeight),
            DetectorOutputLayout.YoloPose => ParsePoseOutput(outputs, originalWidth, originalHeight, inputWidth, inputHeight),
            DetectorOutputLayout.YuNet => ParseYuNet(outputs, originalWidth, originalHeight, inputWidth),
            DetectorOutputLayout.YuNetPlate => ParseLpdYuNet(outputs, originalWidth, originalHeight),
            _ => throw new NotSupportedException($"Detector output layout '{_layout}' has no decoder.")
        };

        // Apply confidence threshold and max detections
        var filtered = detections
            .Where(d => d.Confidence >= _options.ConfidenceThreshold)
            .Where(d => _options.ClassFilter == null || _options.ClassFilter.Contains(d.ClassId))
            .OrderByDescending(d => d.Confidence)
            .Take(_options.MaxDetections)
            .ToList();

        return filtered;
    }

    /// <summary>
    /// Builds the NCHW input tensor in the channel order and value range the model was exported against.
    /// </summary>
    /// <remarks>
    /// Getting this wrong is silent. A model fed the opposite channel order raises nothing and returns an
    /// empty result, which reads exactly like a photograph containing none of what was being looked for -
    /// measured on YuNet, BGR finds seven faces in a street scene where RGB finds none.
    /// </remarks>
    internal static DenseTensor<float> PreprocessImage(
        Image<Rgb24> image, int targetWidth, int targetHeight, DetectorInputFormat format)
    {
        // Resize to the size this model was exported for. Aspect ratio is deliberately not preserved: the
        // reference preprocessing for both families stretches, and the offsets the models emit are read
        // against that stretched frame.
        image.Mutate(x => x.Resize(targetWidth, targetHeight));

        var tensor = new DenseTensor<float>([1, 3, targetHeight, targetWidth]);

        // The RT-DETR reference preprocessing rescales to 0..1 and stops there; its own image processor
        // ships the ImageNet statistics with normalisation switched off. YuNet takes raw BGR bytes.
        var scale = format switch
        {
            DetectorInputFormat.ScaledRgb => 1f / 255f,
            DetectorInputFormat.RawBgr => 1f,
            _ => throw new NotSupportedException($"Detector input format '{format}' is not supported.")
        };
        var bgr = format is DetectorInputFormat.RawBgr;

        // Written through the flat NCHW buffer rather than the four-index accessor: the tensor indexer
        // recomputes a stride product per element, and there are three million of them per 640x640 frame.
        var plane = targetWidth * targetHeight;

        image.ProcessPixelRows(accessor =>
        {
            // Taken inside the callback: a span cannot be captured by the lambda.
            var buffer = tensor.Buffer.Span;

            for (int y = 0; y < targetHeight; y++)
            {
                var row = accessor.GetRowSpan(y);
                var offset = y * targetWidth;

                for (int x = 0; x < targetWidth; x++)
                {
                    var pixel = row[x];
                    buffer[offset + x] = (bgr ? pixel.B : pixel.R) * scale;
                    buffer[plane + offset + x] = pixel.G * scale;
                    buffer[plane * 2 + offset + x] = (bgr ? pixel.R : pixel.B) * scale;
                }
            }
        });

        return tensor;
    }

    private async Task<IDisposableReadOnlyCollection<DisposableNamedOnnxValue>> RunInferenceAsync(
        DenseTensor<float> inputTensor,
        int originalWidth,
        int originalHeight,
        CancellationToken cancellationToken)
    {
        await _sessionLock.WaitAsync(cancellationToken);
        try
        {
            // Bounded run: if the native call hangs (e.g. a cold DirectML kernel init) or the
            // provider crashes, the session moves to the next provider and the run is retried once.
            // Input names/metadata are read from the session the delegate receives so a replacement
            // session created by that fallback is described correctly.
            return await _session!.RunWithRecoveryAsync((session, runOptions) =>
            {
                var inputName = session.InputNames[0];
                var inputs = new List<NamedOnnxValue>
                {
                    NamedOnnxValue.CreateFromTensor(inputName, inputTensor)
                };

                // RT-DETR v2 models with inline postprocessor require original image dimensions
                if (session.InputMetadata.ContainsKey("orig_target_sizes"))
                {
                    var origSizes = new DenseTensor<long>(
                        new long[] { originalHeight, originalWidth },
                        new int[] { 1, 2 });
                    inputs.Add(NamedOnnxValue.CreateFromTensor("orig_target_sizes", origSizes));
                }

                return session.Run(inputs, session.OutputNames, runOptions);
            }, cancellationToken: cancellationToken);
        }
        finally
        {
            _sessionLock.Release();
        }
    }

    /// <summary>
    /// Parses RT-DETR style output (NMS-free, direct detections).
    /// Handles two formats:
    /// - Postprocessed (orig_target_sizes used): named outputs "labels" (int64), "boxes" (float32 x1y1x2y2 in pixel coords), "scores" (float32)
    /// - Standard RT-DETR: "logits" [1, N, num_classes] + "boxes" [1, N, 4] (normalized cx,cy,w,h)
    /// </summary>
    private List<DetectionResult> ParseNmsFree(
        IDisposableReadOnlyCollection<DisposableNamedOnnxValue> outputs,
        int originalWidth,
        int originalHeight,
        int inputWidth,
        int inputHeight)
    {
        var results = new List<DetectionResult>();

        var outputList = outputs.ToList();

        if (outputList.Count >= 2)
        {
            // Check for postprocessed RT-DETR v2 output (inline postprocessor with orig_target_sizes)
            // Outputs: labels [int64], boxes [float32 x1,y1,x2,y2 in pixel coords], scores [float32]
            var labelsOutput = outputList.FirstOrDefault(o => o.Name == "labels");
            var boxesOutput = outputList.FirstOrDefault(o => o.Name == "boxes");
            var scoresOutput = outputList.FirstOrDefault(o => o.Name == "scores");

            if (labelsOutput is not null && boxesOutput is not null && scoresOutput is not null)
            {
                var labels = labelsOutput.AsTensor<long>();
                var boxes = boxesOutput.AsTensor<float>();
                var scores = scoresOutput.AsTensor<float>();

                // Support both [N] and [1, N] shaped outputs
                bool hasBatch = scores.Rank > 1;
                int numDetections = hasBatch ? (int)scores.Dimensions[1] : (int)scores.Dimensions[0];

                for (int i = 0; i < numDetections; i++)
                {
                    float score = hasBatch ? scores[0, i] : scores[i];
                    if (score < _options.ConfidenceThreshold)
                        continue;

                    int classId = (int)(hasBatch ? labels[0, i] : labels[i]);

                    // Boxes are already in original pixel coords: x1, y1, x2, y2
                    float x1 = hasBatch ? boxes[0, i, 0] : boxes[i, 0];
                    float y1 = hasBatch ? boxes[0, i, 1] : boxes[i, 1];
                    float x2 = hasBatch ? boxes[0, i, 2] : boxes[i, 2];
                    float y2 = hasBatch ? boxes[0, i, 3] : boxes[i, 3];

                    var box = new BoundingBox(x1, y1, x2, y2)
                        .Clamp(originalWidth, originalHeight);

                    results.Add(new DetectionResult(
                        ClassId: classId,
                        Label: _modelInfo.LabelFor(classId),
                        Confidence: score,
                        Box: box));
                }
            }
            else
            {
                // Standard RT-DETR format: separate logits and boxes
                var logits = outputList[0].AsTensor<float>();
                var boxes = outputList[1].AsTensor<float>();

                var numQueries = (int)logits.Dimensions[1];
                var numClasses = (int)logits.Dimensions[2];

                for (int i = 0; i < numQueries; i++)
                {
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
                        Label: _modelInfo.LabelFor(bestClass),
                        Confidence: maxScore,
                        Box: box));
                }
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
                    var scaleX = originalWidth / (float)inputWidth;
                    var scaleY = originalHeight / (float)inputHeight;

                    var box = new BoundingBox(
                        output[0, i, 0] * scaleX,
                        output[0, i, 1] * scaleY,
                        output[0, i, 2] * scaleX,
                        output[0, i, 3] * scaleY)
                        .Clamp(originalWidth, originalHeight);

                    results.Add(new DetectionResult(
                        ClassId: classId,
                        Label: _modelInfo.LabelFor(classId),
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
                        Label: _modelInfo.LabelFor(bestClass),
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
        int inputWidth,
        int inputHeight)
    {
        var allDetections = new List<DetectionResult>();
        var output = outputs[0].AsTensor<float>();

        // Standard YOLO format: [1, num_boxes, 4 + num_classes]
        var numBoxes = (int)output.Dimensions[1];
        var outputDim = (int)output.Dimensions[2];
        var numClasses = outputDim - 4;

        var scaleX = originalWidth / (float)inputWidth;
        var scaleY = originalHeight / (float)inputHeight;

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
                Label: _modelInfo.LabelFor(bestClass),
                Confidence: maxScore,
                Box: box));
        }

        // Apply NMS
        return DetectionNms.Apply(allDetections, _options.IouThreshold);
    }

    /// <summary>
    /// Parses licence-plate YuNet output: <c>loc</c>/<c>conf</c>/<c>iou</c> against generated priors,
    /// decoded by <see cref="LpdYuNetDecoder"/> and then suppressed.
    /// </summary>
    private List<DetectionResult> ParseLpdYuNet(
        IDisposableReadOnlyCollection<DisposableNamedOnnxValue> outputs,
        int originalWidth,
        int originalHeight)
    {
        var byName = outputs.ToDictionary(o => o.Name, StringComparer.Ordinal);

        var decoded = LpdYuNetDecoder.Decode(
            new LpdYuNetOutput(
                Values(byName, "loc"),
                Values(byName, "conf"),
                Values(byName, "iou")),
            _platePriors.Value,
            originalWidth,
            originalHeight,
            _options.ConfidenceThreshold,
            _modelInfo.LabelFor(0));

        return DetectionNms.Apply(decoded, _options.IouThreshold);

        static float[] Values(Dictionary<string, DisposableNamedOnnxValue> outputs, string name)
        {
            if (!outputs.TryGetValue(name, out var value))
            {
                throw new InvalidOperationException(
                    $"Model was declared as licence-plate YuNet but emits no output named '{name}'. " +
                    $"Available outputs: {string.Join(", ", outputs.Keys)}.");
            }

            return value.AsTensor<float>().ToArray();
        }
    }

    /// <summary>
    /// Parses YuNet output: one cls/obj/bbox/kps group per stride, decoded by <see cref="YuNetDecoder"/>
    /// and then suppressed, since the model emits several anchors per face.
    /// </summary>
    private List<DetectionResult> ParseYuNet(
        IDisposableReadOnlyCollection<DisposableNamedOnnxValue> outputs,
        int originalWidth,
        int originalHeight,
        int inputSize)
    {
        var byName = outputs.ToDictionary(o => o.Name, StringComparer.Ordinal);
        var branches = new Dictionary<int, YuNetStrideOutput>(YuNetDecoder.Strides.Count);

        foreach (var stride in YuNetDecoder.Strides)
        {
            branches[stride] = new YuNetStrideOutput(
                Values(byName, $"cls_{stride}"),
                Values(byName, $"obj_{stride}"),
                Values(byName, $"bbox_{stride}"),
                Values(byName, $"kps_{stride}"));
        }

        var decoded = YuNetDecoder.Decode(
            branches, inputSize, originalWidth, originalHeight,
            _options.ConfidenceThreshold, _modelInfo.LabelFor(0));

        return DetectionNms.Apply(decoded, _options.IouThreshold);

        static float[] Values(Dictionary<string, DisposableNamedOnnxValue> outputs, string name)
        {
            if (!outputs.TryGetValue(name, out var value))
            {
                throw new InvalidOperationException(
                    $"Model was declared as YuNet but emits no output named '{name}'. " +
                    $"Available outputs: {string.Join(", ", outputs.Keys)}.");
            }

            return value.AsTensor<float>().ToArray();
        }
    }

    /// <summary>
    /// Parses pose estimation output (e.g., YOLOv8-pose).
    /// Expected output format: [1, features, num_boxes] where features = 4(bbox) + 1(conf) + num_keypoints*3.
    /// Also handles [1, num_boxes, features] format.
    /// </summary>
    private List<DetectionResult> ParsePoseOutput(
        IDisposableReadOnlyCollection<DisposableNamedOnnxValue> outputs,
        int originalWidth,
        int originalHeight,
        int inputWidth,
        int inputHeight)
    {
        var results = new List<DetectionResult>();
        var output = outputs[0].AsTensor<float>();

        var dim0 = (int)output.Dimensions[1];
        var dim1 = (int)output.Dimensions[2];

        var expectedFeatures = 4 + 1 + KeypointCount * 3;

        // Auto-detect orientation: [1, features, boxes] vs [1, boxes, features]
        bool transposed = dim0 == expectedFeatures && dim1 != expectedFeatures;
        int numBoxes = transposed ? dim1 : dim0;
        int features = transposed ? dim0 : dim1;

        if (features < expectedFeatures)
            return results;

        var scaleX = originalWidth / (float)inputWidth;
        var scaleY = originalHeight / (float)inputHeight;

        for (int i = 0; i < numBoxes; i++)
        {
            float Val(int f) => transposed ? output[0, f, i] : output[0, i, f];

            var conf = Val(4);
            if (conf < _options.ConfidenceThreshold)
                continue;

            // Box: [cx, cy, w, h]
            var cx = Val(0) * scaleX;
            var cy = Val(1) * scaleY;
            var w = Val(2) * scaleX;
            var h = Val(3) * scaleY;

            var box = BoundingBox.FromCenterSize(cx, cy, w, h)
                .Clamp(originalWidth, originalHeight);

            // Parse keypoints: (x, y, confidence) × numKeypoints
            var keypoints = new Keypoint[KeypointCount];
            var kpOffset = 5;
            for (int k = 0; k < KeypointCount; k++)
            {
                var kx = Val(kpOffset + k * 3) * scaleX;
                var ky = Val(kpOffset + k * 3 + 1) * scaleY;
                var kc = Val(kpOffset + k * 3 + 2);
                keypoints[k] = new Keypoint(kx, ky, kc);
            }

            results.Add(new DetectionResult(
                ClassId: 0,
                Label: _modelInfo.LabelFor(0), // a pose model has one class; it names it
                Confidence: conf,
                Box: box,
                Keypoints: keypoints));
        }

        // Apply NMS if needed
        return _layout.RequiresNms() ? DetectionNms.Apply(results, _options.IouThreshold) : results;
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
            var result = await OnnxSessionFactory.CreateWithInfoAsync(
                modelPath,
                _options.Provider,
                ConfigureSessionOptions,
                cancellationToken: cancellationToken);

            _session = RecoverableOnnxSession.FromResult(
                result, modelPath, ConfigureSessionOptions, logPrefix: "[OnnxDetectorModel]");

            _isInitialized = true;
        }
        finally
        {
            _sessionLock.Release();
        }
    }

    private async Task<string> ResolveModelPathAsync(CancellationToken cancellationToken)
    {
        // Use centralized ModelPathResolver for consistent subfolder handling.
        // Variant suffix stripping (e.g., "owner/repo:variant") is handled by ModelPathResolver.
        using var resolver = new ModelPathResolver(_options.CacheDirectory, localFilesOnly: _options.DisableAutoDownload);

        var result = await resolver.ResolveModelAsync(
            _modelInfo.Id,
            expectedOnnxFile: _modelInfo.OnnxFile,
            cancellationToken: cancellationToken);

        return result.ModelPath;
    }

    private void ConfigureSessionOptions(SessionOptions options)
    {
        options.LogSeverityLevel = (OrtLoggingLevel)(int)_options.LogLevel;

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
