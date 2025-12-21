using LMSupply.Core;
using LMSupply.ImageGenerator.Tokenizers;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace LMSupply.ImageGenerator.Encoders;

/// <summary>
/// CLIP text encoder that converts text prompts to embeddings for image generation.
/// </summary>
internal sealed class ClipTextEncoder : IAsyncDisposable
{
    private readonly InferenceSession _session;
    private readonly ClipTokenizer _tokenizer;
    private readonly string _inputName;
    private readonly string _outputName;
    private bool _disposed;

    /// <summary>
    /// The embedding dimension of the text encoder output.
    /// </summary>
    public int EmbeddingDim { get; }

    /// <summary>
    /// Maximum sequence length supported.
    /// </summary>
    public int MaxLength => _tokenizer.MaxLength;

    private ClipTextEncoder(
        InferenceSession session,
        ClipTokenizer tokenizer,
        string inputName,
        string outputName,
        int embeddingDim)
    {
        _session = session;
        _tokenizer = tokenizer;
        _inputName = inputName;
        _outputName = outputName;
        EmbeddingDim = embeddingDim;
    }

    /// <summary>
    /// Loads the CLIP text encoder from model directory.
    /// </summary>
    /// <param name="modelDir">Path to model directory.</param>
    /// <param name="options">Session options for ONNX Runtime.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Loaded text encoder.</returns>
    public static async Task<ClipTextEncoder> LoadAsync(
        string modelDir,
        SessionOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        // Find text encoder ONNX file
        var encoderPath = FindTextEncoderPath(modelDir);

        // Load tokenizer
        var tokenizer = ClipTokenizer.FromDirectory(modelDir);

        // Create session
        options ??= new SessionOptions();
        var session = await Task.Run(
            () => new InferenceSession(encoderPath, options),
            cancellationToken);

        // Get input/output names
        var inputName = session.InputNames.First();
        var outputName = session.OutputNames.First();

        // Determine embedding dimension from output shape
        var outputMeta = session.OutputMetadata[outputName];
        var embeddingDim = outputMeta.Dimensions.Length > 2
            ? outputMeta.Dimensions[2]
            : outputMeta.Dimensions[^1];

        return new ClipTextEncoder(session, tokenizer, inputName, outputName, embeddingDim);
    }

    /// <summary>
    /// Encodes a text prompt into embeddings.
    /// </summary>
    /// <param name="prompt">Text prompt to encode.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Text embeddings tensor of shape [1, maxLength, embeddingDim].</returns>
    public async Task<DenseTensor<float>> EncodeAsync(
        string prompt,
        CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        // Tokenize
        var tokenIds = _tokenizer.EncodeForModel(prompt);

        // Create input tensor [1, maxLength]
        // Note: Most CLIP models expect Int32 input, not Int64
        var inputData = tokenIds.Select(id => (int)id).ToArray();
        var inputTensor = new DenseTensor<int>(inputData, [1, _tokenizer.MaxLength]);

        // Run inference
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(_inputName, inputTensor)
        };

        var result = await Task.Run(() =>
        {
            using var outputs = _session.Run(inputs);
            var output = outputs.First();

            // Copy to our own tensor
            var outputTensor = output.AsTensor<float>();
            var dims = outputTensor.Dimensions.ToArray();
            var data = outputTensor.ToArray();

            return new DenseTensor<float>(data, dims);
        }, cancellationToken);

        return result;
    }

    /// <summary>
    /// Encodes a prompt and its negative prompt for classifier-free guidance.
    /// </summary>
    /// <param name="prompt">Positive prompt.</param>
    /// <param name="negativePrompt">Negative prompt (empty string if none).</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Combined embeddings [2, maxLength, embeddingDim] where [0] is negative, [1] is positive.</returns>
    public async Task<DenseTensor<float>> EncodeWithNegativeAsync(
        string prompt,
        string? negativePrompt,
        CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        // Use empty string for null negative prompt
        negativePrompt ??= string.Empty;

        // Tokenize both prompts
        var positiveIds = _tokenizer.EncodeForModel(prompt);
        var negativeIds = _tokenizer.EncodeForModel(negativePrompt);

        // Create batched input tensor [2, maxLength]
        // Note: Most CLIP models expect Int32 input, not Int64
        var inputData = new int[2 * _tokenizer.MaxLength];
        for (int i = 0; i < _tokenizer.MaxLength; i++)
        {
            inputData[i] = (int)negativeIds[i];
            inputData[_tokenizer.MaxLength + i] = (int)positiveIds[i];
        }

        var inputTensor = new DenseTensor<int>(inputData, [2, _tokenizer.MaxLength]);

        // Run inference
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(_inputName, inputTensor)
        };

        var result = await Task.Run(() =>
        {
            using var outputs = _session.Run(inputs);
            var output = outputs.First();

            var outputTensor = output.AsTensor<float>();
            var dims = outputTensor.Dimensions.ToArray();
            var data = outputTensor.ToArray();

            return new DenseTensor<float>(data, dims);
        }, cancellationToken);

        return result;
    }

    private static string FindTextEncoderPath(string modelDir)
    {
        // Common paths for text encoder ONNX file
        var candidates = new[]
        {
            Path.Combine(modelDir, "text_encoder", "model.onnx"),
            Path.Combine(modelDir, "text_encoder.onnx"),
            Path.Combine(modelDir, "clip_text_encoder.onnx"),
            Path.Combine(modelDir, "encoder.onnx")
        };

        foreach (var path in candidates)
        {
            if (File.Exists(path))
                return path;
        }

        // Search for any text_encoder related ONNX file
        var files = Directory.GetFiles(modelDir, "*text*encoder*.onnx", SearchOption.AllDirectories);
        if (files.Length > 0)
            return files[0];

        throw new FileNotFoundException(
            $"Could not find CLIP text encoder ONNX file in: {modelDir}");
    }

    public async ValueTask DisposeAsync()
    {
        if (_disposed) return;
        _disposed = true;

        _tokenizer.Dispose();
        _session.Dispose();

        await Task.CompletedTask;
    }
}
