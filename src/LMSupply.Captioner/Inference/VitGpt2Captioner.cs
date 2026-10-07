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
    // ViT-GPT2 is an image-to-text captioning architecture; it cannot take a question
    public bool SupportsVqa => false;

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
    /// <param name="progress">Receives the runtime download session creation may make; <see langword="null"/> when nobody is listening.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    public static async Task<VitGpt2Captioner> CreateAsync(
        string modelDir,
        ModelInfo modelInfo,
        CaptionerOptions options,
        string? tokenizerDir,
        IProgress<DownloadProgress>? progress,
        CancellationToken cancellationToken)
    {
        var encoderPath = Path.Combine(modelDir, modelInfo.EncoderFile);
        var decoderPath = Path.Combine(modelDir, modelInfo.DecoderFile);

        if (!File.Exists(encoderPath))
            throw new ModelNotFoundException($"Encoder file not found: {encoderPath}", modelInfo.AliasName);
        if (!File.Exists(decoderPath))
            throw new ModelNotFoundException($"Decoder file not found: {decoderPath}", modelInfo.AliasName);

        // Load ONNX sessions
        Action<SessionOptions> configureLog = so => so.ApplyCommonOptions(options);
        var blacklist = new ProviderBlacklist();

        var encoderResult = await OnnxSessionFactory.CreateWithInfoAsync(encoderPath, options.Provider, configureLog, progress, cancellationToken).ConfigureAwait(false);
        var encoder = RecoverableOnnxSession.FromResult(
            encoderResult, encoderPath, configureLog, logPrefix: "[VitGpt2Captioner:encoder]", blacklist: blacklist);

        RecoverableOnnxSession decoder;
        try
        {
            var decoderResult = await OnnxSessionFactory.CreateWithInfoAsync(decoderPath, options.Provider, configureLog, progress, cancellationToken).ConfigureAwait(false);
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
    public Task<VqaResult> AnswerAsync(string imagePath, string question, CancellationToken cancellationToken = default) =>
        throw new NotSupportedException($"Model '{ModelId}' does not support visual question answering.");

    /// <inheritdoc />
    public Task<VqaResult> AnswerAsync(Stream imageStream, string question, CancellationToken cancellationToken = default) =>
        throw new NotSupportedException($"Model '{ModelId}' does not support visual question answering.");

    private async Task<CaptionResult> GenerateCaptionAsync(float[] imageData, CancellationToken cancellationToken)
    {
        var imageEmbeddings = await RunEncoderAsync(imageData, cancellationToken).ConfigureAwait(false);
        var encoderHiddenStates = EncoderHiddenStates(imageEmbeddings);

        // Start with BOS, then the prompt when one is set: the decoder continues the caption from it (conditional
        // captioning — the prompt's tokens become the start of the caption, as with decoder_input_ids in Transformers).
        var promptTokens = string.IsNullOrWhiteSpace(_options.Prompt)
            ? []
            : _tokenizer.Encode(_options.Prompt.Trim(), addSpecialTokens: false);
        var prefix = new List<int>(1 + promptTokens.Length) { _modelInfo.BosTokenId };
        prefix.AddRange(promptTokens);

        IReadOnlyList<BeamSearch<object?>.Hypothesis> hypotheses;
        if (_options.NumBeams > 1)
        {
            hypotheses = await BeamSearch<object?>.RunAsync(
                prefix,
                null,
                _options.NumBeams,
                _options.MaxLength,
                _modelInfo.EosTokenId,
                async (sequence, state) => (await StepAsync(sequence, encoderHiddenStates, cancellationToken).ConfigureAwait(false), state),
                adjust: null,
                cancellationToken).ConfigureAwait(false);
        }
        else
        {
            var sequence = new List<int>(prefix);
            var generated = new List<int>();
            float totalLogProb = 0f;
            for (var step = 0; step < _options.MaxLength; step++)
            {
                cancellationToken.ThrowIfCancellationRequested();

                var logits = await StepAsync(sequence, encoderHiddenStates, cancellationToken).ConfigureAwait(false);
                var (token, logProb) = NextToken.Choose(logits, _options);
                if (token == _modelInfo.EosTokenId)
                    break;

                totalLogProb += logProb;
                generated.Add(token);
                sequence.Add(token);
            }

            hypotheses = [new(generated.ToArray(), generated.Count > 0 ? totalLogProb / generated.Count : float.NegativeInfinity)];
        }

        // The caption includes the prompt it was continued from.
        var captions = hypotheses
            .Select(h => _tokenizer.Decode([.. promptTokens, .. h.Tokens], skipSpecialTokens: true))
            .ToList();

        return new CaptionResult(
            captions[0],
            MathF.Exp(hypotheses[0].MeanLogProb),
            captions.Skip(1).Where(c => c != captions[0]).Distinct().ToList());
    }

    private Task<float[]> RunEncoderAsync(float[] imageData, CancellationToken cancellationToken)
    {
        var profile = _modelInfo.PreprocessProfile;
        var imageTensor = TensorUtils.CreateImageTensor(imageData, profile);

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("pixel_values", imageTensor)
        };

        // Bounded run: if the native call hangs (e.g. a cold GPU kernel init) or the provider
        // crashes, the session moves to the next provider and the run is retried once.
        return _encoder.RunWithRecoveryAsync((session, runOptions) =>
        {
            using var results = session.Run(inputs, session.OutputNames, runOptions);

            // Copy the output data out of the native buffer
            return results[0].AsEnumerable<float>().ToArray();
        }, cancellationToken: cancellationToken);
    }

    private DenseTensor<float> EncoderHiddenStates(float[] imageEmbeddings)
    {
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

        return new DenseTensor<float>(imageEmbeddings, [1, seqLen, hiddenSize]);
    }

    /// <summary>
    /// One decoder pass over the whole sequence (this export is run without its key/value cache): the logits for the
    /// token after the sequence's last.
    /// </summary>
    private async Task<float[]> StepAsync(
        IReadOnlyList<int> sequence, DenseTensor<float> encoderHiddenStates, CancellationToken cancellationToken)
    {
        var tokenIds = sequence.Select(t => (long)t).ToArray();
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input_ids", new DenseTensor<long>(tokenIds, [1, tokenIds.Length])),
            NamedOnnxValue.CreateFromTensor("encoder_hidden_states", encoderHiddenStates)
        };

        var decoderInputs = _decoder.Session.InputMetadata;
        if (decoderInputs.ContainsKey("attention_mask"))
            inputs.Add(NamedOnnxValue.CreateFromTensor("attention_mask", TensorUtils.CreateAttentionMask(tokenIds.Length)));

        // use_cache_branch=false: always the non-cached decode path
        if (decoderInputs.ContainsKey("use_cache_branch"))
            inputs.Add(NamedOnnxValue.CreateFromTensor("use_cache_branch", new DenseTensor<bool>(UseCacheBranchFalse, UseCacheBranchDims)));

        // One bounded decode step; a provider crash or hang inside it moves the session to the
        // next provider and retries this step (decoder state is on the managed side).
        var logits = await _decoder.RunWithRecoveryAsync((session, runOptions) =>
        {
            using var results = session.Run(inputs, ["logits"], runOptions);
            return results[0].AsEnumerable<float>().ToArray();
        }, cancellationToken: cancellationToken).ConfigureAwait(false);

        // Logits for the last position; some exports return only the current position, or [batch, vocab].
        var vocabSize = _modelInfo.VocabSize;
        var lastTokenLogitsStart = (tokenIds.Length - 1) * vocabSize;
        if (logits.Length == vocabSize)
            lastTokenLogitsStart = 0;
        else if (logits.Length < lastTokenLogitsStart + vocabSize)
            lastTokenLogitsStart = logits.Length - vocabSize;

        return logits.AsSpan(lastTokenLogitsStart, vocabSize).ToArray();
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
