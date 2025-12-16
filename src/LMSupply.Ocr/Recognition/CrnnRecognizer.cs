using LMSupply.Inference;
using LMSupply.Ocr.Models;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace LMSupply.Ocr.Recognition;

/// <summary>
/// CRNN-based text recognition implementation with CTC decoding.
/// </summary>
internal sealed class CrnnRecognizer : IDisposable
{
    private readonly InferenceSession _session;
    private readonly RecognitionModelInfo _modelInfo;
    private readonly CharacterDictionary _dictionary;
    private readonly float _confidenceThreshold;
    private readonly string _inputName;
    private readonly string _outputName;
    private bool _disposed;

    private CrnnRecognizer(
        InferenceSession session,
        RecognitionModelInfo modelInfo,
        CharacterDictionary dictionary,
        OcrOptions options)
    {
        _session = session;
        _modelInfo = modelInfo;
        _dictionary = dictionary;
        _confidenceThreshold = options.RecognitionThreshold;

        // Get input/output names from the model
        _inputName = session.InputNames.First();
        _outputName = session.OutputNames.First();
    }

    /// <summary>
    /// Creates a new CRNN recognizer instance.
    /// </summary>
    public static async Task<CrnnRecognizer> CreateAsync(
        string modelPath,
        string dictPath,
        RecognitionModelInfo modelInfo,
        OcrOptions options)
    {
        var session = await OnnxSessionFactory.CreateAsync(modelPath, options.Provider)
            .ConfigureAwait(false);

        var dictionary = new CharacterDictionary(dictPath, modelInfo.UseSpace);

        return new CrnnRecognizer(session, modelInfo, dictionary, options);
    }

    /// <summary>
    /// Gets the supported languages for this recognizer.
    /// </summary>
    public IReadOnlyList<string> SupportedLanguages => _modelInfo.LanguageCodes;

    /// <summary>
    /// Recognizes text in cropped image regions.
    /// </summary>
    /// <param name="image">The source image.</param>
    /// <param name="regions">Detected text regions.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>List of recognized text regions.</returns>
    public async Task<IReadOnlyList<TextRegion>> RecognizeAsync(
        Image<Rgb24> image,
        IReadOnlyList<DetectedRegion> regions,
        CancellationToken cancellationToken = default)
    {
        var results = new List<TextRegion>();

        foreach (var region in regions)
        {
            cancellationToken.ThrowIfCancellationRequested();

            // Crop the region from the image
            using var cropped = CropRegion(image, region);

            // Recognize text in the cropped region
            var (text, confidence) = await RecognizeSingleAsync(cropped, cancellationToken)
                .ConfigureAwait(false);

            if (!string.IsNullOrWhiteSpace(text))
            {
                results.Add(new TextRegion(
                    text.Trim(),
                    confidence,
                    region.BoundingBox,
                    region.Polygon));
            }
        }

        return results;
    }

    /// <summary>
    /// Recognizes text in a single cropped image.
    /// </summary>
    public async Task<(string text, float confidence)> RecognizeSingleAsync(
        Image<Rgb24> image,
        CancellationToken cancellationToken = default)
    {
        // Resize to target height while maintaining aspect ratio
        var targetHeight = _modelInfo.InputHeight;
        var aspectRatio = (float)image.Width / image.Height;
        var targetWidth = (int)(targetHeight * aspectRatio);

        // Clamp width
        var maxWidth = targetHeight * _modelInfo.MaxWidthRatio;
        targetWidth = Math.Min(targetWidth, maxWidth);
        targetWidth = Math.Max(targetWidth, targetHeight); // Minimum width

        // Round to multiple of 4 for efficiency
        targetWidth = ((targetWidth + 3) / 4) * 4;

        using var resized = image.Clone();
        resized.Mutate(x => x.Resize(targetWidth, targetHeight));

        // Convert to tensor
        var inputTensor = PreprocessImage(resized);

        // Run inference
        var logits = await Task.Run(() =>
        {
            cancellationToken.ThrowIfCancellationRequested();
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(_inputName, inputTensor)
            };
            var results = _session.Run(inputs);
            return ExtractLogits(results.First().AsTensor<float>());
        }, cancellationToken).ConfigureAwait(false);

        // CTC decode
        return CtcDecoder.GreedyDecode(logits, _dictionary);
    }

    private Image<Rgb24> CropRegion(Image<Rgb24> image, DetectedRegion region)
    {
        var box = region.BoundingBox;

        // Clamp coordinates to image bounds
        var x = Math.Max(0, box.X);
        var y = Math.Max(0, box.Y);
        var width = Math.Min(box.Width, image.Width - x);
        var height = Math.Min(box.Height, image.Height - y);

        if (width <= 0 || height <= 0)
        {
            // Return a minimal image if region is invalid
            return new Image<Rgb24>(1, 1);
        }

        var cropped = image.Clone();
        cropped.Mutate(ctx => ctx.Crop(new Rectangle(x, y, width, height)));
        return cropped;
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
                    // Normalize to [0, 1] then apply mean/std
                    tensor[0, 0, y, x] = (pixel.R / 255f - mean[0]) / std[0];
                    tensor[0, 1, y, x] = (pixel.G / 255f - mean[1]) / std[1];
                    tensor[0, 2, y, x] = (pixel.B / 255f - mean[2]) / std[2];
                }
            }
        });

        return tensor;
    }

    private static float[,] ExtractLogits(Tensor<float> outputTensor)
    {
        var dims = outputTensor.Dimensions;

        // Output shape is typically [B, T, V] or [T, V]
        int seqLength, vocabSize;

        if (dims.Length == 3)
        {
            seqLength = dims[1];
            vocabSize = dims[2];

            var logits = new float[seqLength, vocabSize];
            for (var t = 0; t < seqLength; t++)
            {
                for (var v = 0; v < vocabSize; v++)
                {
                    logits[t, v] = outputTensor[0, t, v];
                }
            }
            return logits;
        }
        else if (dims.Length == 2)
        {
            seqLength = dims[0];
            vocabSize = dims[1];

            var logits = new float[seqLength, vocabSize];
            for (var t = 0; t < seqLength; t++)
            {
                for (var v = 0; v < vocabSize; v++)
                {
                    logits[t, v] = outputTensor[t, v];
                }
            }
            return logits;
        }
        else
        {
            throw new InvalidOperationException($"Unexpected output tensor shape: [{string.Join(", ", dims.ToArray())}]");
        }
    }

    public void Dispose()
    {
        if (_disposed) return;
        _session.Dispose();
        _disposed = true;
    }
}
