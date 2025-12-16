using System.Diagnostics;
using LMSupply.Download;
using LMSupply.Inference;
using LMSupply.Segmenter.Interactive;
using LMSupply.Segmenter.Models;
using LMSupply.Vision;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace LMSupply.Segmenter.Core;

/// <summary>
/// MobileSAM implementation for interactive segmentation.
/// Supports point and box prompts for efficient mask generation.
/// </summary>
internal sealed class MobileSamModel : IInteractiveSegmenter
{
    private readonly SegmenterOptions _options;
    private readonly SegmenterModelInfo _modelInfo;
    private readonly SemaphoreSlim _sessionLock = new(1, 1);

    private InferenceSession? _encoderSession;
    private InferenceSession? _decoderSession;
    private bool _isInitialized;
    private bool _disposed;

    // MobileSAM constants
    private const int ImageEncoderSize = 1024;
    private const int EmbeddingSize = 256;

    public MobileSamModel(SegmenterOptions options, SegmenterModelInfo modelInfo)
    {
        _options = options.Clone();
        _modelInfo = modelInfo;

        if (!modelInfo.IsInteractive)
        {
            throw new ArgumentException("Model does not support interactive segmentation", nameof(modelInfo));
        }
    }

    public SegmenterModelInfo? GetModelInfo() => _modelInfo;

    public async Task WarmupAsync(CancellationToken cancellationToken = default)
    {
        await EnsureInitializedAsync(cancellationToken);
    }

    public async Task<IInteractiveSession> CreateSessionAsync(
        string imagePath,
        CancellationToken cancellationToken = default)
    {
        using var stream = File.OpenRead(imagePath);
        return await CreateSessionAsync(stream, cancellationToken);
    }

    public async Task<IInteractiveSession> CreateSessionAsync(
        Stream imageStream,
        CancellationToken cancellationToken = default)
    {
        await EnsureInitializedAsync(cancellationToken);

        using var image = await Image.LoadAsync<Rgb24>(imageStream, cancellationToken);
        return await CreateSessionFromImageAsync(image, cancellationToken);
    }

    public async Task<IInteractiveSession> CreateSessionAsync(
        byte[] imageData,
        CancellationToken cancellationToken = default)
    {
        await EnsureInitializedAsync(cancellationToken);

        using var image = Image.Load<Rgb24>(imageData);
        return await CreateSessionFromImageAsync(image, cancellationToken);
    }

    public async Task<InteractiveSegmentationResult> SegmentAsync(
        string imagePath,
        IEnumerable<PointPrompt> points,
        bool multimask = false,
        CancellationToken cancellationToken = default)
    {
        await using var session = await CreateSessionAsync(imagePath, cancellationToken);
        return await session.SegmentAsync(points, multimask, cancellationToken);
    }

    public async Task<InteractiveSegmentationResult> SegmentAsync(
        string imagePath,
        BoxPrompt box,
        bool multimask = false,
        CancellationToken cancellationToken = default)
    {
        await using var session = await CreateSessionAsync(imagePath, cancellationToken);
        return await session.SegmentAsync(box, multimask, cancellationToken);
    }

    private async Task<MobileSamSession> CreateSessionFromImageAsync(
        Image<Rgb24> image,
        CancellationToken cancellationToken)
    {
        var originalWidth = image.Width;
        var originalHeight = image.Height;

        // Preprocess image for encoder
        var inputTensor = PreprocessImage(image);

        // Run encoder to get image embedding
        DenseTensor<float> imageEmbedding;

        await _sessionLock.WaitAsync(cancellationToken);
        try
        {
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("image", inputTensor)
            };

            using var outputs = _encoderSession!.Run(inputs);
            var embeddingTensor = outputs.First().AsTensor<float>();

            // Clone the embedding
            var dims = embeddingTensor.Dimensions.ToArray();
            imageEmbedding = new DenseTensor<float>(dims);
            var totalElements = dims.Aggregate(1, (a, b) => a * b);
            for (int i = 0; i < totalElements; i++)
            {
                imageEmbedding.SetValue(i, embeddingTensor.GetValue(i));
            }
        }
        finally
        {
            _sessionLock.Release();
        }

        return new MobileSamSession(
            this,
            imageEmbedding,
            originalWidth,
            originalHeight,
            _sessionLock,
            _decoderSession!);
    }

    private static DenseTensor<float> PreprocessImage(Image<Rgb24> image)
    {
        // Resize to 1024x1024 (SAM input size)
        var resized = image.Clone();
        resized.Mutate(x => x.Resize(ImageEncoderSize, ImageEncoderSize));

        // SAM uses ImageNet normalization
        var mean = new[] { 0.485f, 0.456f, 0.406f };
        var std = new[] { 0.229f, 0.224f, 0.225f };

        var tensor = new DenseTensor<float>([1, 3, ImageEncoderSize, ImageEncoderSize]);

        resized.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < accessor.Height; y++)
            {
                var row = accessor.GetRowSpan(y);
                for (int x = 0; x < accessor.Width; x++)
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

    private async Task EnsureInitializedAsync(CancellationToken cancellationToken)
    {
        if (_isInitialized)
            return;

        await _sessionLock.WaitAsync(cancellationToken);
        try
        {
            if (_isInitialized)
                return;

            var modelDir = await ResolveModelPathAsync(cancellationToken);

            // Load encoder
            var encoderPath = Path.Combine(modelDir, _modelInfo.EncoderFile!);
            _encoderSession = await OnnxSessionFactory.CreateAsync(
                encoderPath,
                _options.Provider,
                ConfigureSessionOptions,
                cancellationToken: cancellationToken);

            // Load decoder
            var decoderPath = Path.Combine(modelDir, _modelInfo.DecoderFile!);
            _decoderSession = await OnnxSessionFactory.CreateAsync(
                decoderPath,
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
        if (Directory.Exists(_modelInfo.Id))
        {
            return _modelInfo.Id;
        }

        var parentDir = Path.GetDirectoryName(_modelInfo.Id);
        if (parentDir != null && Directory.Exists(parentDir))
        {
            return parentDir;
        }

        using var downloader = new HuggingFaceDownloader(_options.CacheDirectory);

        var modelDir = await downloader.DownloadModelAsync(
            _modelInfo.Id,
            files:
            [
                _modelInfo.EncoderFile!,
                _modelInfo.DecoderFile!,
                "config.json"
            ],
            cancellationToken: cancellationToken);

        return modelDir;
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

    public void Dispose()
    {
        if (_disposed)
            return;

        _encoderSession?.Dispose();
        _decoderSession?.Dispose();
        _sessionLock.Dispose();
        _disposed = true;
    }

    public async ValueTask DisposeAsync()
    {
        if (_disposed)
            return;

        await _sessionLock.WaitAsync();
        try
        {
            _encoderSession?.Dispose();
            _decoderSession?.Dispose();
        }
        finally
        {
            _sessionLock.Release();
            _sessionLock.Dispose();
        }

        _disposed = true;
    }
}

/// <summary>
/// Interactive session for MobileSAM that caches image embeddings.
/// </summary>
internal sealed class MobileSamSession : IInteractiveSession
{
    private readonly MobileSamModel _model;
    private readonly DenseTensor<float> _imageEmbedding;
    private readonly SemaphoreSlim _sessionLock;
    private readonly InferenceSession _decoderSession;
    private bool _disposed;

    private const int ImageEncoderSize = 1024;

    public int ImageWidth { get; }
    public int ImageHeight { get; }
    public bool IsReady => !_disposed;

    internal MobileSamSession(
        MobileSamModel model,
        DenseTensor<float> imageEmbedding,
        int imageWidth,
        int imageHeight,
        SemaphoreSlim sessionLock,
        InferenceSession decoderSession)
    {
        _model = model;
        _imageEmbedding = imageEmbedding;
        ImageWidth = imageWidth;
        ImageHeight = imageHeight;
        _sessionLock = sessionLock;
        _decoderSession = decoderSession;
    }

    public Task<InteractiveSegmentationResult> SegmentAsync(
        IEnumerable<PointPrompt> points,
        bool multimask = false,
        CancellationToken cancellationToken = default)
    {
        return SegmentAsync(points, null, multimask, cancellationToken);
    }

    public Task<InteractiveSegmentationResult> SegmentAsync(
        BoxPrompt box,
        bool multimask = false,
        CancellationToken cancellationToken = default)
    {
        return SegmentAsync([], box, multimask, cancellationToken);
    }

    public async Task<InteractiveSegmentationResult> SegmentAsync(
        IEnumerable<PointPrompt> points,
        BoxPrompt? box,
        bool multimask = false,
        CancellationToken cancellationToken = default)
    {
        var sw = Stopwatch.StartNew();
        var pointsList = points.ToList();

        // Prepare prompts
        var (pointCoords, pointLabels) = PreparePointPrompts(pointsList, box);
        var hasMask = new DenseTensor<float>(new float[] { 1 }, [1]);
        var maskInput = new DenseTensor<float>([1, 1, 256, 256]);
        var origImSize = new DenseTensor<float>(new float[] { ImageHeight, ImageWidth }, [2]);

        await _sessionLock.WaitAsync(cancellationToken);
        try
        {
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("image_embeddings", _imageEmbedding),
                NamedOnnxValue.CreateFromTensor("point_coords", pointCoords),
                NamedOnnxValue.CreateFromTensor("point_labels", pointLabels),
                NamedOnnxValue.CreateFromTensor("has_mask_input", hasMask),
                NamedOnnxValue.CreateFromTensor("mask_input", maskInput),
                NamedOnnxValue.CreateFromTensor("orig_im_size", origImSize)
            };

            using var outputs = _decoderSession.Run(inputs);
            var masksTensor = outputs.First(o => o.Name == "masks").AsTensor<float>();
            var iouTensor = outputs.First(o => o.Name == "iou_predictions").AsTensor<float>();

            sw.Stop();

            return CreateResult(masksTensor, iouTensor, multimask, sw.Elapsed.TotalMilliseconds);
        }
        finally
        {
            _sessionLock.Release();
        }
    }

    public async Task<InteractiveSegmentationResult> RefineAsync(
        IEnumerable<PointPrompt> points,
        MaskPrompt previousMask,
        CancellationToken cancellationToken = default)
    {
        var sw = Stopwatch.StartNew();
        var pointsList = points.ToList();

        // Prepare prompts
        var (pointCoords, pointLabels) = PreparePointPrompts(pointsList, null);
        var hasMask = new DenseTensor<float>(new float[] { 1 }, [1]);
        var maskInput = PrepareMaskInput(previousMask);
        var origImSize = new DenseTensor<float>(new float[] { ImageHeight, ImageWidth }, [2]);

        await _sessionLock.WaitAsync(cancellationToken);
        try
        {
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("image_embeddings", _imageEmbedding),
                NamedOnnxValue.CreateFromTensor("point_coords", pointCoords),
                NamedOnnxValue.CreateFromTensor("point_labels", pointLabels),
                NamedOnnxValue.CreateFromTensor("has_mask_input", hasMask),
                NamedOnnxValue.CreateFromTensor("mask_input", maskInput),
                NamedOnnxValue.CreateFromTensor("orig_im_size", origImSize)
            };

            using var outputs = _decoderSession.Run(inputs);
            var masksTensor = outputs.First(o => o.Name == "masks").AsTensor<float>();
            var iouTensor = outputs.First(o => o.Name == "iou_predictions").AsTensor<float>();

            sw.Stop();

            return CreateResult(masksTensor, iouTensor, multimask: false, sw.Elapsed.TotalMilliseconds);
        }
        finally
        {
            _sessionLock.Release();
        }
    }

    private (DenseTensor<float> Coords, DenseTensor<float> Labels) PreparePointPrompts(
        List<PointPrompt> points,
        BoxPrompt? box)
    {
        var totalPoints = points.Count + (box != null ? 2 : 0);
        if (totalPoints == 0)
        {
            // No prompts - use a dummy point (SAM requires at least one)
            totalPoints = 1;
        }

        var coords = new DenseTensor<float>([1, totalPoints, 2]);
        var labels = new DenseTensor<float>([1, totalPoints]);

        var scaleX = (float)ImageEncoderSize / ImageWidth;
        var scaleY = (float)ImageEncoderSize / ImageHeight;

        int idx = 0;

        // Add point prompts
        foreach (var point in points)
        {
            coords[0, idx, 0] = point.X * scaleX;
            coords[0, idx, 1] = point.Y * scaleY;
            labels[0, idx] = (int)point.Label;
            idx++;
        }

        // Add box prompts (top-left and bottom-right corners)
        if (box != null)
        {
            coords[0, idx, 0] = box.X * scaleX;
            coords[0, idx, 1] = box.Y * scaleY;
            labels[0, idx] = 2; // Box corner label
            idx++;

            coords[0, idx, 0] = box.Right * scaleX;
            coords[0, idx, 1] = box.Bottom * scaleY;
            labels[0, idx] = 3; // Box corner label
            idx++;
        }

        // If no prompts, add dummy point
        if (idx == 0)
        {
            coords[0, 0, 0] = ImageWidth / 2f * scaleX;
            coords[0, 0, 1] = ImageHeight / 2f * scaleY;
            labels[0, 0] = -1; // Ignored label
        }

        return (coords, labels);
    }

    private DenseTensor<float> PrepareMaskInput(MaskPrompt mask)
    {
        // Resize mask to 256x256 and convert to tensor
        var tensor = new DenseTensor<float>([1, 1, 256, 256]);

        var scaleX = (float)256 / mask.Mask.GetLength(1);
        var scaleY = (float)256 / mask.Mask.GetLength(0);

        for (int y = 0; y < 256; y++)
        {
            for (int x = 0; x < 256; x++)
            {
                var srcX = (int)(x / scaleX);
                var srcY = (int)(y / scaleY);
                srcX = Math.Clamp(srcX, 0, mask.Mask.GetLength(1) - 1);
                srcY = Math.Clamp(srcY, 0, mask.Mask.GetLength(0) - 1);

                tensor[0, 0, y, x] = mask.Mask[srcY, srcX] ? 1f : 0f;
            }
        }

        return tensor;
    }

    private InteractiveSegmentationResult CreateResult(
        Tensor<float> masksTensor,
        Tensor<float> iouTensor,
        bool multimask,
        double inferenceTimeMs)
    {
        var numMasks = multimask ? (int)masksTensor.Dimensions[1] : 1;
        var maskHeight = (int)masksTensor.Dimensions[2];
        var maskWidth = (int)masksTensor.Dimensions[3];

        var masks = new List<SegmentationMask>();
        var iouScores = new List<float>();

        // Get IoU scores
        var scores = new List<(int Index, float Score)>();
        for (int i = 0; i < numMasks; i++)
        {
            scores.Add((i, iouTensor[0, i]));
        }

        // Sort by score descending
        scores.Sort((a, b) => b.Score.CompareTo(a.Score));

        foreach (var (maskIdx, score) in scores)
        {
            // Convert mask tensor to boolean mask, resized to original image dimensions
            var boolMask = new bool[ImageHeight, ImageWidth];
            var scaleX = (float)maskWidth / ImageWidth;
            var scaleY = (float)maskHeight / ImageHeight;

            for (int y = 0; y < ImageHeight; y++)
            {
                for (int x = 0; x < ImageWidth; x++)
                {
                    var srcX = (int)(x * scaleX);
                    var srcY = (int)(y * scaleY);
                    srcX = Math.Clamp(srcX, 0, maskWidth - 1);
                    srcY = Math.Clamp(srcY, 0, maskHeight - 1);

                    boolMask[y, x] = masksTensor[0, maskIdx, srcY, srcX] > 0;
                }
            }

            masks.Add(new SegmentationMask(boolMask, 0) { Label = "object" });
            iouScores.Add(score);
        }

        return new InteractiveSegmentationResult
        {
            Masks = masks,
            IoUScores = iouScores,
            ImageWidth = ImageWidth,
            ImageHeight = ImageHeight,
            InferenceTimeMs = inferenceTimeMs
        };
    }

    public void Dispose()
    {
        if (_disposed)
            return;

        // Note: We don't dispose the decoder session as it's owned by the model
        _disposed = true;
    }

    public ValueTask DisposeAsync()
    {
        Dispose();
        return ValueTask.CompletedTask;
    }
}
