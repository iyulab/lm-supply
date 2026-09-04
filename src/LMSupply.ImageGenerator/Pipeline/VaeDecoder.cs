using LMSupply.Inference;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace LMSupply.ImageGenerator.Pipeline;

/// <summary>
/// VAE decoder that converts latent representations to RGB images.
/// </summary>
internal sealed class VaeDecoder : IAsyncDisposable
{
    private readonly InferenceSession _session;
    private readonly SemaphoreSlim _sessionLock = new(1, 1);
    private readonly string _inputName;
    private readonly string _outputName;
    private readonly float _scalingFactor;
    private bool _disposed;

    private VaeDecoder(
        InferenceSession session,
        string inputName,
        string outputName,
        float scalingFactor)
    {
        _session = session;
        _inputName = inputName;
        _outputName = outputName;
        _scalingFactor = scalingFactor;
    }

    /// <summary>
    /// Loads the VAE decoder from the model directory.
    /// </summary>
    public static async Task<VaeDecoder> LoadAsync(
        string modelDir,
        SessionOptions? options = null,
        float scalingFactor = 0.18215f,
        CancellationToken cancellationToken = default)
    {
        var modelPath = FindVaePath(modelDir);

        options ??= new SessionOptions();
        var session = await Task.Run(
            () => new InferenceSession(modelPath, options),
            cancellationToken);

        var inputName = session.InputNames[0];
        var outputName = session.OutputNames[0];

        return new VaeDecoder(session, inputName, outputName, scalingFactor);
    }

    /// <summary>
    /// Decodes latents to an image.
    /// </summary>
    /// <param name="latents">Latent tensor [1, 4, h/8, w/8].</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Decoded image as PNG bytes.</returns>
    public async Task<byte[]> DecodeAsync(
        DenseTensor<float> latents,
        CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        // Scale latents (VAE scaling factor)
        var scaledLatents = ScaleLatents(latents, 1.0f / _scalingFactor);

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(_inputName, scaledLatents)
        };

        // CancellableInference guarantees control returns to the caller if the token is
        // cancelled, or after a bounded default timeout, even when the native ONNX call (e.g. a
        // cold DirectML kernel init) ignores cancellation and blocks indefinitely.
        var imageBytes = await CancellableInference.RunAsync(() =>
        {
            _sessionLock.Wait(cancellationToken);
            try
            {
                using var outputs = _session.Run(inputs);
                var output = outputs[0];

                var outputTensor = output.AsTensor<float>();
                return TensorToImage(outputTensor);
            }
            finally
            {
                _sessionLock.Release();
            }
        }, cancellationToken);

        return imageBytes;
    }

    private static DenseTensor<float> ScaleLatents(DenseTensor<float> latents, float scale)
    {
        var scaled = new DenseTensor<float>(latents.Dimensions.ToArray());

        for (int i = 0; i < latents.Length; i++)
        {
            scaled.Buffer.Span[i] = latents.Buffer.Span[i] * scale;
        }

        return scaled;
    }

    private static byte[] TensorToImage(Tensor<float> tensor)
    {
        // Expected shape: [1, 3, height, width]
        var dims = tensor.Dimensions;
        var batch = dims[0];
        var channels = dims[1];
        var height = dims[2];
        var width = dims[3];

        // Create image
        using var image = new Image<Rgb24>(width, height);

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                // Get RGB values and denormalize from [-1, 1] to [0, 255]
                var r = DenormalizePixel(tensor[0, 0, y, x]);
                var g = DenormalizePixel(tensor[0, 1, y, x]);
                var b = DenormalizePixel(tensor[0, 2, y, x]);

                image[x, y] = new Rgb24(r, g, b);
            }
        }

        // Encode to PNG
        using var ms = new MemoryStream();
        image.SaveAsPng(ms);
        return ms.ToArray();
    }

    private static byte DenormalizePixel(float value)
    {
        // From [-1, 1] to [0, 255]
        var normalized = (value + 1.0f) / 2.0f;
        var clamped = Math.Clamp(normalized, 0f, 1f);
        return (byte)(clamped * 255);
    }

    private static string FindVaePath(string modelDir)
    {
        var candidates = new[]
        {
            Path.Combine(modelDir, "vae_decoder", "model.onnx"),
            Path.Combine(modelDir, "vae_decoder.onnx"),
            Path.Combine(modelDir, "decoder.onnx")
        };

        foreach (var path in candidates)
        {
            if (File.Exists(path))
                return path;
        }

        var files = Directory.GetFiles(modelDir, "*vae*decoder*.onnx", SearchOption.AllDirectories);
        if (files.Length > 0)
            return files[0];

        // Try just "vae" if decoder not found separately
        files = Directory.GetFiles(modelDir, "*vae*.onnx", SearchOption.AllDirectories);
        if (files.Length > 0)
            return files[0];

        throw new FileNotFoundException($"Could not find VAE decoder ONNX file in: {modelDir}");
    }

    public ValueTask DisposeAsync()
    {
        if (_disposed) return ValueTask.CompletedTask;
        _disposed = true;

        _session.Dispose();
        _sessionLock.Dispose();
        return ValueTask.CompletedTask;
    }
}
