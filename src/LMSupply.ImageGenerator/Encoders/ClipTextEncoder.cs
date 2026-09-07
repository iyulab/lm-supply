using LMSupply.Core;
using LMSupply.ImageGenerator.Tokenizers;
using LMSupply.Inference;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace LMSupply.ImageGenerator.Encoders;

/// <summary>
/// CLIP text encoder that converts text prompts to embeddings for image generation.
/// </summary>
internal sealed class ClipTextEncoder : IAsyncDisposable
{
    // Owns the ONNX session and recovers from a crashing or hanging execution provider by moving
    // to the next one in the fallback chain (see RecoverableOnnxSession). The blacklist is shared
    // with the pipeline's other sessions.
    private readonly RecoverableOnnxSession _session;
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
        RecoverableOnnxSession session,
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
    /// <param name="provider">Execution provider to create the session with.</param>
    /// <param name="deviceId">GPU device index for CUDA/DirectML.</param>
    /// <param name="configureOptions">Session options to apply (log level, threads).</param>
    /// <param name="blacklist">Provider blacklist shared with the pipeline's other sessions.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Loaded text encoder.</returns>
    public static async Task<ClipTextEncoder> LoadAsync(
        string modelDir,
        ExecutionProvider provider,
        int deviceId,
        Action<SessionOptions>? configureOptions,
        ProviderBlacklist blacklist,
        CancellationToken cancellationToken = default)
    {
        // Find text encoder ONNX file
        var encoderPath = FindTextEncoderPath(modelDir);

        // Load tokenizer
        var tokenizer = ClipTokenizer.FromDirectory(modelDir);

        // Create session through the shared factory (provider chain, runtime provisioning)
        var result = await OnnxSessionFactory.CreateWithInfoAsync(
            encoderPath, provider, skipProviders: null, configureOptions,
            cancellationToken: cancellationToken, deviceId: deviceId);

        // Get input/output names (identical on every provider)
        var inputName = result.Session.InputNames[0];
        var outputName = result.Session.OutputNames[0];

        // Determine embedding dimension from output shape
        var outputMeta = result.Session.OutputMetadata[outputName];
        var embeddingDim = outputMeta.Dimensions.Length > 2
            ? outputMeta.Dimensions[2]
            : outputMeta.Dimensions[^1];

        var session = RecoverableOnnxSession.FromResult(
            result, encoderPath, configureOptions, logPrefix: "[ClipTextEncoder]", blacklist: blacklist);

        return new ClipTextEncoder(session, tokenizer, inputName, outputName, embeddingDim);
    }

    /// <summary>
    /// Encodes a text prompt into embeddings.
    /// </summary>
    /// <param name="prompt">Text prompt to encode.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Text embeddings tensor of shape [1, maxLength, embeddingDim].</returns>
    public Task<DenseTensor<float>> EncodeAsync(
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

        return RunAsync(inputTensor, cancellationToken);
    }

    /// <summary>
    /// Encodes a prompt and its negative prompt for classifier-free guidance.
    /// </summary>
    /// <param name="prompt">Positive prompt.</param>
    /// <param name="negativePrompt">Negative prompt (empty string if none).</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Combined embeddings [2, maxLength, embeddingDim] where [0] is negative, [1] is positive.</returns>
    public Task<DenseTensor<float>> EncodeWithNegativeAsync(
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

        return RunAsync(inputTensor, cancellationToken);
    }

    private Task<DenseTensor<float>> RunAsync(DenseTensor<int> inputTensor, CancellationToken cancellationToken)
    {
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(_inputName, inputTensor)
        };

        // Bounded run: if the native call hangs (e.g. a cold DirectML kernel init) or the provider
        // crashes, the session moves to the next provider and the run is retried once.
        return _session.RunWithRecoveryAsync((session, runOptions) =>
        {
            using var outputs = session.Run(inputs, [_outputName], runOptions);
            var outputTensor = outputs[0].AsTensor<float>();

            // Copy out of the native buffer since the outputs are disposed with this delegate
            return new DenseTensor<float>(outputTensor.ToArray(), outputTensor.Dimensions);
        }, cancellationToken: cancellationToken);
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

    public ValueTask DisposeAsync()
    {
        if (_disposed) return ValueTask.CompletedTask;
        _disposed = true;

        _tokenizer.Dispose();
        _session.Dispose();

        return ValueTask.CompletedTask;
    }
}
