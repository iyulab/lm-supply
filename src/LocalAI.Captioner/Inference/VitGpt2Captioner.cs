using LocalAI.Captioner.Models;
using LocalAI.Exceptions;
using LocalAI.Inference;
using LocalAI.Text;
using LocalAI.Vision;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace LocalAI.Captioner.Inference;

/// <summary>
/// Image captioner implementation for ViT-GPT2 style encoder-decoder models.
/// </summary>
internal sealed class VitGpt2Captioner : ICaptioner
{
    private readonly InferenceSession _encoder;
    private readonly InferenceSession _decoder;
    private readonly ITextTokenizer _tokenizer;
    private readonly ModelInfo _modelInfo;
    private readonly CaptionerOptions _options;
    private readonly ImagePreprocessor _preprocessor;
    private bool _disposed;

    private VitGpt2Captioner(
        InferenceSession encoder,
        InferenceSession decoder,
        ITextTokenizer tokenizer,
        ModelInfo modelInfo,
        CaptionerOptions options)
    {
        _encoder = encoder;
        _decoder = decoder;
        _tokenizer = tokenizer;
        _modelInfo = modelInfo;
        _options = options;
        _preprocessor = ImagePreprocessor.Instance;
    }

    /// <inheritdoc />
    public string ModelId => _modelInfo.Alias;

    /// <inheritdoc />
    public bool SupportsVqa => _modelInfo.SupportsVqa;

    /// <inheritdoc />
    public Task WarmupAsync(CancellationToken cancellationToken = default)
    {
        // Run a minimal inference to warm up the model
        // The encoder and decoder sessions are already loaded, so this ensures JIT compilation is done
        return Task.CompletedTask;
    }

    /// <inheritdoc />
    public ModelInfo? GetModelInfo() => _modelInfo;

    /// <summary>
    /// Creates a new VitGpt2Captioner instance.
    /// </summary>
    public static async Task<VitGpt2Captioner> CreateAsync(
        string modelDir,
        ModelInfo modelInfo,
        CaptionerOptions options)
    {
        var encoderPath = Path.Combine(modelDir, modelInfo.EncoderFile);
        var decoderPath = Path.Combine(modelDir, modelInfo.DecoderFile);

        if (!File.Exists(encoderPath))
            throw new ModelNotFoundException($"Encoder file not found: {encoderPath}", modelInfo.Alias);
        if (!File.Exists(decoderPath))
            throw new ModelNotFoundException($"Decoder file not found: {decoderPath}", modelInfo.Alias);

        // Load ONNX sessions
        var encoder = await OnnxSessionFactory.CreateAsync(encoderPath, options.Provider).ConfigureAwait(false);
        var decoder = await OnnxSessionFactory.CreateAsync(decoderPath, options.Provider).ConfigureAwait(false);

        // Load tokenizer from Text.Core
        var tokenizer = Text.TokenizerFactory.CreateGpt2(modelDir);

        return new VitGpt2Captioner(encoder, decoder, tokenizer, modelInfo, options);
    }

    /// <inheritdoc />
    public async Task<CaptionResult> CaptionAsync(string imagePath, CancellationToken cancellationToken = default)
    {
        var imageData = await _preprocessor.PreprocessAsync(
            imagePath, _modelInfo.PreprocessProfile, cancellationToken).ConfigureAwait(false);
        return await GenerateCaptionAsync(imageData, cancellationToken).ConfigureAwait(false);
    }

    /// <inheritdoc />
    public async Task<CaptionResult> CaptionAsync(Stream imageStream, CancellationToken cancellationToken = default)
    {
        var imageData = await _preprocessor.PreprocessAsync(
            imageStream, _modelInfo.PreprocessProfile, cancellationToken).ConfigureAwait(false);
        return await GenerateCaptionAsync(imageData, cancellationToken).ConfigureAwait(false);
    }

    /// <inheritdoc />
    public async Task<CaptionResult> CaptionAsync(byte[] imageData, CancellationToken cancellationToken = default)
    {
        var preprocessed = await _preprocessor.PreprocessAsync(
            imageData, _modelInfo.PreprocessProfile, cancellationToken).ConfigureAwait(false);
        return await GenerateCaptionAsync(preprocessed, cancellationToken).ConfigureAwait(false);
    }

    /// <inheritdoc />
    public Task<VqaResult> AnswerAsync(string imagePath, string question, CancellationToken cancellationToken = default)
    {
        if (!SupportsVqa)
            throw new NotSupportedException($"Model '{ModelId}' does not support visual question answering.");

        // TODO: Implement VQA for models that support it
        throw new NotImplementedException("VQA support is not yet implemented for this model.");
    }

    /// <inheritdoc />
    public Task<VqaResult> AnswerAsync(Stream imageStream, string question, CancellationToken cancellationToken = default)
    {
        if (!SupportsVqa)
            throw new NotSupportedException($"Model '{ModelId}' does not support visual question answering.");

        throw new NotImplementedException("VQA support is not yet implemented for this model.");
    }

    private async Task<CaptionResult> GenerateCaptionAsync(float[] imageData, CancellationToken cancellationToken)
    {
        // Run encoder to get image embeddings
        var imageEmbeddings = await Task.Run(() => RunEncoder(imageData), cancellationToken).ConfigureAwait(false);

        // Run decoder to generate caption tokens
        var (tokenIds, confidence) = await Task.Run(
            () => GenerateTokens(imageEmbeddings, cancellationToken), cancellationToken).ConfigureAwait(false);

        // Decode tokens to text, skipping special tokens
        var caption = _tokenizer.Decode(tokenIds, skipSpecialTokens: true);

        return new CaptionResult(caption, confidence);
    }

    private float[] RunEncoder(float[] imageData)
    {
        var profile = _modelInfo.PreprocessProfile;
        var imageTensor = TensorUtils.CreateImageTensor(imageData, profile);

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("pixel_values", imageTensor)
        };

        using var results = _encoder.Run(inputs);
        var output = results.First();

        // Copy the output data
        var embeddings = output.AsEnumerable<float>().ToArray();
        return embeddings;
    }

    private (int[] tokenIds, float confidence) GenerateTokens(float[] imageEmbeddings, CancellationToken cancellationToken)
    {
        var generatedTokens = new List<int>();
        float totalLogProb = 0f;
        int tokenCount = 0;

        // Start with BOS token
        var currentTokenIds = new long[] { _modelInfo.BosTokenId };

        // Get embedding dimensions from encoder output
        // For ViT-GPT2, typical shape is [1, seq_len, hidden_size]
        var embeddingDim = _encoder.OutputMetadata.First().Value.Dimensions;
        int seqLen = embeddingDim.Length > 1 ? embeddingDim[1] : 1;
        int hiddenSize = embeddingDim.Length > 2 ? embeddingDim[2] : embeddingDim[^1];

        // Create encoder hidden states tensor
        var encoderHiddenStates = new DenseTensor<float>(
            imageEmbeddings,
            [1, seqLen, hiddenSize]);

        for (int step = 0; step < _options.MaxLength; step++)
        {
            cancellationToken.ThrowIfCancellationRequested();

            // Prepare decoder inputs
            var inputIdsTensor = new DenseTensor<long>(currentTokenIds, [1, currentTokenIds.Length]);
            var attentionMask = TensorUtils.CreateAttentionMask(currentTokenIds.Length);

            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input_ids", inputIdsTensor),
                NamedOnnxValue.CreateFromTensor("attention_mask", attentionMask),
                NamedOnnxValue.CreateFromTensor("encoder_hidden_states", encoderHiddenStates)
            };

            // Run decoder
            using var results = _decoder.Run(inputs);
            var logitsOutput = results.First(r => r.Name == "logits");
            var logits = logitsOutput.AsEnumerable<float>().ToArray();

            // Get logits for last token position
            int vocabSize = _modelInfo.VocabSize;
            int lastTokenLogitsStart = (currentTokenIds.Length - 1) * vocabSize;

            // Handle case where logits might be just for current position
            if (logits.Length == vocabSize)
            {
                lastTokenLogitsStart = 0;
            }
            else if (logits.Length < lastTokenLogitsStart + vocabSize)
            {
                // Logits shape might be [batch, vocab] instead of [batch, seq, vocab]
                lastTokenLogitsStart = logits.Length - vocabSize;
            }

            var lastTokenLogits = logits.AsSpan(lastTokenLogitsStart, vocabSize);

            // Sample next token
            int nextToken;
            float logProb;

            if (_options.Temperature <= 0 || _options.NumBeams == 1)
            {
                // Greedy decoding
                nextToken = TensorUtils.ArgMax(lastTokenLogits);
                var probs = TensorUtils.Softmax(lastTokenLogits);
                logProb = MathF.Log(probs[nextToken] + 1e-10f);
            }
            else
            {
                // Temperature sampling
                var probs = TensorUtils.Softmax(lastTokenLogits, _options.Temperature);
                nextToken = TensorUtils.SampleFromDistribution(probs);
                logProb = MathF.Log(probs[nextToken] + 1e-10f);
            }

            // Check for EOS
            if (nextToken == _modelInfo.EosTokenId)
                break;

            generatedTokens.Add(nextToken);
            totalLogProb += logProb;
            tokenCount++;

            // Update input for next iteration
            currentTokenIds = [.. currentTokenIds, nextToken];
        }

        // Calculate average log probability as confidence
        float confidence = tokenCount > 0 ? totalLogProb / tokenCount : 0f;

        // Normalize to 0-1 range (log prob is typically negative)
        confidence = MathF.Exp(confidence);

        return (generatedTokens.ToArray(), confidence);
    }

    /// <inheritdoc />
    public ValueTask DisposeAsync()
    {
        if (_disposed) return ValueTask.CompletedTask;

        _encoder.Dispose();
        _decoder.Dispose();
        _disposed = true;

        return ValueTask.CompletedTask;
    }
}
