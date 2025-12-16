using LMSupply.Download;
using LMSupply.Inference;
using LMSupply.Segmenter.Models;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace LMSupply.Segmenter.Core;

/// <summary>
/// ONNX Runtime-based semantic segmentation implementation.
/// Supports SegFormer and similar single-session architectures.
/// </summary>
internal sealed class OnnxSegmenterModel : ISegmenterModel
{
    private readonly SegmenterOptions _options;
    private readonly SegmenterModelInfo _modelInfo;
    private readonly SemaphoreSlim _sessionLock = new(1, 1);

    private InferenceSession? _session;
    private bool _isInitialized;
    private bool _disposed;

    /// <inheritdoc />
    public string ModelId => _modelInfo.Id;

    /// <summary>
    /// Gets the class labels for this model.
    /// </summary>
    public IReadOnlyList<string> ClassLabels => Ade20kLabels.Labels;

    public OnnxSegmenterModel(SegmenterOptions options)
    {
        _options = options.Clone();
        _modelInfo = SegmenterModelRegistry.Default.Resolve(options.ModelId);
    }

    public async Task WarmupAsync(CancellationToken cancellationToken = default)
    {
        await EnsureInitializedAsync(cancellationToken);
    }

    public SegmenterModelInfo? GetModelInfo() => _modelInfo;

    public async Task<SegmentationResult> SegmentAsync(
        string imagePath,
        CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(imagePath);

        using var image = await Image.LoadAsync<Rgb24>(imagePath, cancellationToken);
        return await SegmentCoreAsync(image, cancellationToken);
    }

    public async Task<SegmentationResult> SegmentAsync(
        Stream imageStream,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(imageStream);

        using var image = await Image.LoadAsync<Rgb24>(imageStream, cancellationToken);
        return await SegmentCoreAsync(image, cancellationToken);
    }

    public async Task<SegmentationResult> SegmentAsync(
        byte[] imageData,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(imageData);

        using var image = Image.Load<Rgb24>(imageData);
        return await SegmentCoreAsync(image, cancellationToken);
    }

    public async Task<IReadOnlyList<SegmentationResult>> SegmentBatchAsync(
        IEnumerable<string> imagePaths,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(imagePaths);

        var results = new List<SegmentationResult>();

        foreach (var path in imagePaths)
        {
            var result = await SegmentAsync(path, cancellationToken);
            results.Add(result);
        }

        return results;
    }

    private async Task<SegmentationResult> SegmentCoreAsync(
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

        // Parse segmentation output
        var (classMap, confidenceMap, outputWidth, outputHeight) = ParseOutput(outputs);

        // Resize to original dimensions if requested
        if (_options.ResizeToOriginal && (outputWidth != originalWidth || outputHeight != originalHeight))
        {
            (classMap, confidenceMap) = ResizeOutput(
                classMap, confidenceMap,
                outputWidth, outputHeight,
                originalWidth, originalHeight);
            outputWidth = originalWidth;
            outputHeight = originalHeight;
        }

        return new SegmentationResult
        {
            Width = outputWidth,
            Height = outputHeight,
            ClassMap = classMap,
            ConfidenceMap = confidenceMap
        };
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
    /// Parses SegFormer output.
    /// Output shape: [1, num_classes, height, width] (logits)
    /// </summary>
    private (int[] classMap, float[] confidenceMap, int width, int height) ParseOutput(
        IDisposableReadOnlyCollection<DisposableNamedOnnxValue> outputs)
    {
        var output = outputs.First().AsTensor<float>();
        var dims = output.Dimensions.ToArray();

        // SegFormer output: [1, num_classes, H, W]
        var numClasses = dims[1];
        var height = dims[2];
        var width = dims[3];

        var classMap = new int[width * height];
        var confidenceMap = new float[width * height];

        // For each pixel, find the class with highest score
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                var pixelIndex = y * width + x;
                float maxScore = float.MinValue;
                int bestClass = 0;

                // Find argmax across classes
                for (int c = 0; c < numClasses; c++)
                {
                    var score = output[0, c, y, x];
                    if (score > maxScore)
                    {
                        maxScore = score;
                        bestClass = c;
                    }
                }

                classMap[pixelIndex] = bestClass;
                // Convert logit to probability using softmax approximation
                confidenceMap[pixelIndex] = Sigmoid(maxScore);
            }
        }

        return (classMap, confidenceMap, width, height);
    }

    /// <summary>
    /// Resizes the output to original image dimensions using nearest neighbor interpolation.
    /// </summary>
    private static (int[] classMap, float[] confidenceMap) ResizeOutput(
        int[] srcClassMap,
        float[] srcConfidenceMap,
        int srcWidth,
        int srcHeight,
        int dstWidth,
        int dstHeight)
    {
        var dstClassMap = new int[dstWidth * dstHeight];
        var dstConfidenceMap = new float[dstWidth * dstHeight];

        var scaleX = (float)srcWidth / dstWidth;
        var scaleY = (float)srcHeight / dstHeight;

        for (int y = 0; y < dstHeight; y++)
        {
            for (int x = 0; x < dstWidth; x++)
            {
                var srcX = (int)(x * scaleX);
                var srcY = (int)(y * scaleY);

                srcX = Math.Clamp(srcX, 0, srcWidth - 1);
                srcY = Math.Clamp(srcY, 0, srcHeight - 1);

                var srcIndex = srcY * srcWidth + srcX;
                var dstIndex = y * dstWidth + x;

                dstClassMap[dstIndex] = srcClassMap[srcIndex];
                dstConfidenceMap[dstIndex] = srcConfidenceMap[srcIndex];
            }
        }

        return (dstClassMap, dstConfidenceMap);
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
