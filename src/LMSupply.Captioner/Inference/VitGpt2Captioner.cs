using LMSupply.Captioner.Models;
using LMSupply.Exceptions;
using LMSupply.Inference;
using LMSupply.Text;
using LMSupply.Vision;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace LMSupply.Captioner.Inference;

/// <summary>
/// Image captioner implementation for ViT-GPT2 style encoder-decoder models.
/// </summary>
internal sealed class VitGpt2Captioner : ICaptionerModel
{
    // Encoder and decoder each own a RecoverableOnnxSession and share one provider blacklist, so a
    // provider that crashes or hangs on one half of the model is left by the other half before its
    // next run (see RecoverableOnnxSession).
    private readonly RecoverableOnnxSession _encoder;
    private readonly RecoverableOnnxSession _decoder;
    private readonly ITextTokenizer _tokenizer;
    private readonly ModelInfo _modelInfo;
    private readonly CaptionerOptions _options;
    private readonly ImagePreprocessor _preprocessor;
    private readonly string _modelDir;
    private bool _disposed;

    private static readonly bool[] UseCacheBranchFalse = [false];
    private static readonly int[] UseCacheBranchDims = [1];

    private VitGpt2Captioner(
        RecoverableOnnxSession encoder,
        RecoverableOnnxSession decoder,
        ITextTokenizer tokenizer,
        ModelInfo modelInfo,
        CaptionerOptions options,
        string modelDir)
    {
        _encoder = encoder;
        _decoder = decoder;
        _tokenizer = tokenizer;
        _modelInfo = modelInfo;
        _options = options;
        _modelDir = modelDir;
        _preprocessor = ImagePreprocessor.Instance;
    }

    /// <inheritdoc />
    public string ModelId => _modelInfo.AliasName;

    /// <inheritdoc />
    public bool IsGpuActive => _encoder.IsGpuActive;

    /// <inheritdoc />
    public IReadOnlyList<string> ActiveProviders => _encoder.ActiveProviders;

    /// <inheritdoc />
    public ExecutionProvider RequestedProvider => _encoder.RequestedProvider;

    /// <inheritdoc />
    public long? EstimatedMemoryBytes => Directory.Exists(_modelDir)
        ? new DirectoryInfo(_modelDir).EnumerateFiles("*.onnx", SearchOption.AllDirectories).Sum(f => f.Length) * 2
        : null;

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
    /// <param name="modelDir">Directory containing ONNX model files.</param>
    /// <param name="modelInfo">Model configuration info.</param>
    /// <param name="options">Captioner options.</param>
    /// <param name="tokenizerDir">Optional directory containing tokenizer files. If null, uses modelDir.</param>
    public static async Task<VitGpt2Captioner> CreateAsync(
        string modelDir,
        ModelInfo modelInfo,
        CaptionerOptions options,
        string? tokenizerDir = null)
    {
        var encoderPath = Path.Combine(modelDir, modelInfo.EncoderFile);
        var decoderPath = Path.Combine(modelDir, modelInfo.DecoderFile);

        if (!File.Exists(encoderPath))
            throw new ModelNotFoundException($"Encoder file not found: {encoderPath}", modelInfo.AliasName);
        if (!File.Exists(decoderPath))
            throw new ModelNotFoundException($"Decoder file not found: {decoderPath}", modelInfo.AliasName);

        // Load ONNX sessions
        Action<SessionOptions> configureLog = so => so.LogSeverityLevel = (OrtLoggingLevel)(int)options.LogLevel;
        var blacklist = new ProviderBlacklist();

        var encoderResult = await OnnxSessionFactory.CreateWithInfoAsync(encoderPath, options.Provider, configureLog).ConfigureAwait(false);
        var encoder = RecoverableOnnxSession.FromResult(
            encoderResult, encoderPath, configureLog, logPrefix: "[VitGpt2Captioner:encoder]", blacklist: blacklist);

        RecoverableOnnxSession decoder;
        try
        {
            var decoderResult = await OnnxSessionFactory.CreateWithInfoAsync(decoderPath, options.Provider, configureLog).ConfigureAwait(false);
            decoder = RecoverableOnnxSession.FromResult(
                decoderResult, decoderPath, configureLog, logPrefix: "[VitGpt2Captioner:decoder]", blacklist: blacklist);
        }
        catch
        {
            encoder.Dispose();
            throw;
        }

        // Load tokenizer from Text.Core - tokenizer files may be in a different directory (e.g., base dir for HuggingFace repos)
        var tokenizer = Text.TokenizerFactory.CreateGpt2(tokenizerDir ?? modelDir);

        return new VitGpt2Captioner(encoder, decoder, tokenizer, modelInfo, options, modelDir);
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
        var imageEmbeddings = await RunEncoderAsync(imageData, cancellationToken).ConfigureAwait(false);

        // Run decoder to generate caption tokens
        var (tokenIds, confidence) = await GenerateTokensAsync(imageEmbeddings, cancellationToken).ConfigureAwait(false);

        // Decode tokens to text, skipping special tokens
        var caption = _tokenizer.Decode(tokenIds, skipSpecialTokens: true);

        return new CaptionResult(caption, confidence);
    }

    private Task<float[]> RunEncoderAsync(float[] imageData, CancellationToken cancellationToken)
    {
        var profile = _modelInfo.PreprocessProfile;
        var imageTensor = TensorUtils.CreateImageTensor(imageData, profile);

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("pixel_values", imageTensor)
        };

        // Bounded run: if the native call hangs (e.g. a cold DirectML kernel init) or the provider
        // crashes, the session moves to the next provider and the run is retried once.
        return _encoder.RunWithRecoveryAsync((session, runOptions) =>
        {
            using var results = session.Run(inputs, session.OutputNames, runOptions);

            // Copy the output data out of the native buffer
            return results[0].AsEnumerable<float>().ToArray();
        }, cancellationToken: cancellationToken);
    }

    private async Task<(int[] tokenIds, float confidence)> GenerateTokensAsync(float[] imageEmbeddings, CancellationToken cancellationToken)
    {
        var generatedTokens = new List<int>();
        float totalLogProb = 0f;
        int tokenCount = 0;

        // Start with BOS token
        var currentTokenIds = new long[] { _modelInfo.BosTokenId };

        // Get embedding dimensions from encoder output metadata (identical on every provider).
        // For ViT-GPT2, typical shape is [1, seq_len, hidden_size]
        // Note: ONNX dynamic dimensions are represented as -1 in metadata, so we infer from actual data
        var embeddingDim = _encoder.Session.OutputMetadata.First().Value.Dimensions;
        int seqLen = embeddingDim.Length > 1 ? embeddingDim[1] : 1;
        int hiddenSize = embeddingDim.Length > 2 ? embeddingDim[2] : embeddingDim[^1];

        // Resolve dynamic dimensions from actual embedding array length (dynamic dims are -1 in ONNX metadata)
        if (seqLen <= 0 && hiddenSize > 0)
        {
            seqLen = imageEmbeddings.Length / hiddenSize;
        }
        else if (hiddenSize <= 0 && seqLen > 0)
        {
            hiddenSize = imageEmbeddings.Length / seqLen;
        }
        else if (seqLen <= 0 && hiddenSize <= 0)
        {
            // Both dynamic: use ViT-GPT2 default hidden size of 768
            hiddenSize = 768;
            seqLen = imageEmbeddings.Length / hiddenSize;
        }

        // Create encoder hidden states tensor
        var encoderHiddenStates = new DenseTensor<float>(
            imageEmbeddings,
            [1, seqLen, hiddenSize]);

        // Pre-create tensors for optional inputs (check once, reuse per step)
        var decoderInputs = _decoder.Session.InputMetadata;
        bool needsAttentionMask = decoderInputs.ContainsKey("attention_mask");
        bool needsCacheBranch = decoderInputs.ContainsKey("use_cache_branch");
        // use_cache_branch=false: always use the non-cached decode path
        DenseTensor<bool>? useCacheTensor = needsCacheBranch
            ? new DenseTensor<bool>(UseCacheBranchFalse, UseCacheBranchDims)
            : null;

        for (int step = 0; step < _options.MaxLength; step++)
        {
            cancellationToken.ThrowIfCancellationRequested();

            // Prepare decoder inputs
            var inputIdsTensor = new DenseTensor<long>(currentTokenIds, [1, currentTokenIds.Length]);
            var attentionMask = TensorUtils.CreateAttentionMask(currentTokenIds.Length);

            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input_ids", inputIdsTensor),
                NamedOnnxValue.CreateFromTensor("encoder_hidden_states", encoderHiddenStates)
            };

            if (needsAttentionMask)
            {
                inputs.Add(NamedOnnxValue.CreateFromTensor("attention_mask", attentionMask));
            }

            if (needsCacheBranch)
            {
                inputs.Add(NamedOnnxValue.CreateFromTensor("use_cache_branch", useCacheTensor!));
            }

            // One bounded decode step; a provider crash or hang inside it moves the session to the
            // next provider and retries this step (decoder state is on the managed side).
            var logits = await _decoder.RunWithRecoveryAsync((session, runOptions) =>
            {
                using var results = session.Run(inputs, ["logits"], runOptions);
                return results[0].AsEnumerable<float>().ToArray();
            }, cancellationToken: cancellationToken).ConfigureAwait(false);

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
