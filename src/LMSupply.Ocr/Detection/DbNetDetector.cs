using LMSupply.Inference;
using LMSupply.Ocr.Models;
using LMSupply.Ocr.PostProcessing;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace LMSupply.Ocr.Detection;

/// <summary>
/// DBNet-based text detection implementation.
/// </summary>
internal sealed class DbNetDetector : IDisposable
{
    // Owns the ONNX session and recovers from a crashing or hanging execution provider by moving
    // to the next one in the fallback chain (see RecoverableOnnxSession).
    private readonly RecoverableOnnxSession _session;
    private readonly DetectionModelInfo _modelInfo;
    private readonly DbNetPostProcessor _postProcessor;
    private readonly string _inputName;
    private readonly string _outputName;
    private bool _disposed;

    /// <summary>
    /// Gets whether GPU acceleration is being used for inference.
    /// </summary>
    public bool IsGpuActive => _session.IsGpuActive;

    /// <summary>
    /// Gets the list of active execution providers.
    /// </summary>
    public IReadOnlyList<string> ActiveProviders => _session.ActiveProviders;

    /// <summary>
    /// Gets the execution provider that was requested.
    /// </summary>
    public ExecutionProvider RequestedProvider => _session.RequestedProvider;

    private DbNetDetector(
        RecoverableOnnxSession session,
        DetectionModelInfo modelInfo,
        OcrOptions options)
    {
        _session = session;
        _modelInfo = modelInfo;
        _postProcessor = new DbNetPostProcessor(options);

        // Get input/output names from the model
        _inputName = session.Session.InputNames[0];
        _outputName = session.Session.OutputNames[0];
    }

    /// <summary>
    /// Creates a new DBNet detector instance.
    /// </summary>
    public static async Task<DbNetDetector> CreateAsync(
        string modelPath,
        DetectionModelInfo modelInfo,
        OcrOptions options)
    {
        Action<SessionOptions> configure = so => so.LogSeverityLevel = (OrtLoggingLevel)(int)options.LogLevel;

        var result = await OnnxSessionFactory.CreateWithInfoAsync(modelPath, options.Provider, configure)
            .ConfigureAwait(false);

        return new DbNetDetector(
            RecoverableOnnxSession.FromResult(result, modelPath, configure, logPrefix: "[DbNetDetector]"),
            modelInfo,
            options);
    }

    /// <summary>
    /// Detects text regions in an image.
    /// </summary>
    public async Task<IReadOnlyList<DetectedRegion>> DetectAsync(
        Image<Rgb24> image,
        CancellationToken cancellationToken = default)
    {
        // Calculate resize dimensions while maintaining aspect ratio
        var (resizeWidth, resizeHeight) = CalculateResizeDimensions(
            image.Width, image.Height,
            _modelInfo.InputWidth, _modelInfo.InputHeight);

        // Round to multiple of 32 for model compatibility
        resizeWidth = (resizeWidth / 32) * 32;
        resizeHeight = (resizeHeight / 32) * 32;

        if (resizeWidth == 0) resizeWidth = 32;
        if (resizeHeight == 0) resizeHeight = 32;

        // Create a copy and resize
        using var resizedImage = image.Clone();
        resizedImage.Mutate(x => x.Resize(resizeWidth, resizeHeight));

        // Convert to tensor
        var inputTensor = PreprocessImage(resizedImage);

        // Bounded run: if the native call hangs (e.g. a cold DirectML kernel init) or the
        // provider crashes, the session moves to the next provider and the run is retried once.
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(_inputName, inputTensor)
        };
        using var results = await _session.RunWithRecoveryAsync(
                (session, runOptions) => session.Run(inputs, [_outputName], runOptions),
                cancellationToken: cancellationToken)
            .ConfigureAwait(false);

        // Get output tensor (the collection owns and disposes its values)
        var outputTensor = results[0].AsTensor<float>();

        // Extract probability map
        var probabilityMap = ExtractProbabilityMap(outputTensor, resizeWidth, resizeHeight);

        // Post-process to get detected regions
        var regions = _postProcessor.Process(probabilityMap, image.Width, image.Height);

        return regions;
    }

    /// <summary>
    /// Detects text regions in an image from a file path.
    /// </summary>
    public async Task<IReadOnlyList<DetectedRegion>> DetectAsync(
        string imagePath,
        CancellationToken cancellationToken = default)
    {
        using var image = await Image.LoadAsync<Rgb24>(imagePath, cancellationToken).ConfigureAwait(false);
        return await DetectAsync(image, cancellationToken).ConfigureAwait(false);
    }

    /// <summary>
    /// Detects text regions in an image from a stream.
    /// </summary>
    public async Task<IReadOnlyList<DetectedRegion>> DetectAsync(
        Stream imageStream,
        CancellationToken cancellationToken = default)
    {
        using var image = await Image.LoadAsync<Rgb24>(imageStream, cancellationToken).ConfigureAwait(false);
        return await DetectAsync(image, cancellationToken).ConfigureAwait(false);
    }

    /// <summary>
    /// Gets the loaded image for use in recognition.
    /// </summary>
    public static async Task<Image<Rgb24>> LoadImageAsync(string imagePath, CancellationToken cancellationToken = default)
    {
        return await Image.LoadAsync<Rgb24>(imagePath, cancellationToken).ConfigureAwait(false);
    }

    /// <summary>
    /// Gets the loaded image for use in recognition.
    /// </summary>
    public static async Task<Image<Rgb24>> LoadImageAsync(Stream imageStream, CancellationToken cancellationToken = default)
    {
        return await Image.LoadAsync<Rgb24>(imageStream, cancellationToken).ConfigureAwait(false);
    }

    /// <summary>
    /// Gets the loaded image for use in recognition.
    /// </summary>
    public static Image<Rgb24> LoadImage(byte[] imageData)
    {
        return Image.Load<Rgb24>(imageData);
    }

    private DenseTensor<float> PreprocessImage(Image<Rgb24> image)
    {
        var height = image.Height;
        var width = image.Width;
        var tensor = new DenseTensor<float>([1, 3, height, width]);

        var mean = _modelInfo.Mean;
        var std = _modelInfo.Std;

        image.ProcessPixelRows(accessor =>
        {
            for (var y = 0; y < height; y++)
            {
                var row = accessor.GetRowSpan(y);
                for (var x = 0; x < width; x++)
                {
                    var pixel = row[x];
                    tensor[0, 0, y, x] = (pixel.R - mean[0]) / std[0];
                    tensor[0, 1, y, x] = (pixel.G - mean[1]) / std[1];
                    tensor[0, 2, y, x] = (pixel.B - mean[2]) / std[2];
                }
            }
        });

        return tensor;
    }

    private static float[,] ExtractProbabilityMap(Tensor<float> outputTensor, int width, int height)
    {
        var dims = outputTensor.Dimensions;

        // Output shape is typically [1, 1, H, W] or [1, H, W]
        int mapHeight, mapWidth;
        Func<int, int, float> getValue;

        if (dims.Length == 4)
        {
            mapHeight = dims[2];
            mapWidth = dims[3];
            getValue = (y, x) => outputTensor[0, 0, y, x];
        }
        else if (dims.Length == 3)
        {
            mapHeight = dims[1];
            mapWidth = dims[2];
            getValue = (y, x) => outputTensor[0, y, x];
        }
        else
        {
            throw new InvalidOperationException($"Unexpected output tensor shape: [{string.Join(", ", dims.ToArray())}]");
        }

        var probabilityMap = new float[mapHeight, mapWidth];

        for (var y = 0; y < mapHeight; y++)
        {
            for (var x = 0; x < mapWidth; x++)
            {
                probabilityMap[y, x] = getValue(y, x);
            }
        }

        return probabilityMap;
    }

    private static (int width, int height) CalculateResizeDimensions(
        int originalWidth, int originalHeight,
        int maxWidth, int maxHeight)
    {
        var ratio = Math.Min(
            (float)maxWidth / originalWidth,
            (float)maxHeight / originalHeight);

        // Limit ratio to not upscale too much
        ratio = Math.Min(ratio, 2.0f);

        return (
            (int)(originalWidth * ratio),
            (int)(originalHeight * ratio));
    }

    public void Dispose()
    {
        if (_disposed) return;
        _session.Dispose();
        _disposed = true;
    }
}
