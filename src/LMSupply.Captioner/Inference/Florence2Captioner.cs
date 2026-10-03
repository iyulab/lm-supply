using LMSupply.Captioner.Models;
using LMSupply.Exceptions;
using LMSupply.Inference;
using LMSupply.Text;
using LMSupply.Vision;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace LMSupply.Captioner.Inference;

/// <summary>
/// Image captioner for Florence-2 ONNX exports (the four-graph layout of onnx-community/Florence-2-*): a vision
/// encoder, a token embedding, a text encoder over the image features followed by the task prompt, and a merged
/// BART-style decoder with a key/value cache.
/// </summary>
/// <remarks>
/// Generation follows the model's <c>generation_config.json</c> except for the beam width: the decoder starts from
/// <see cref="ModelInfo.DecoderStartTokenId"/>, the first generated token is forced to <see cref="ModelInfo.BosTokenId"/>,
/// and no 3-gram repeats. Decoding is greedy (or temperature sampling, as <see cref="NextToken"/> decides) rather than
/// the config's three beams — measured on everyday photos, greedy named the same subjects at a third of the cost.
/// </remarks>
internal sealed class Florence2Captioner : ICaptionerModel
{
    /// <summary><c>no_repeat_ngram_size</c> in the model's generation config.</summary>
    internal const int NoRepeatNgramSize = 3;

    /// <summary>Token ids from here up are Florence's added location/task tokens; none belongs in a caption.</summary>
    private const int FirstAddedTokenId = 50265;

    /// <summary>&lt;s&gt; &lt;pad&gt; &lt;/s&gt; &lt;unk&gt;.</summary>
    private const int FirstOrdinaryTokenId = 4;

    private readonly RecoverableOnnxSession _vision;
    private readonly RecoverableOnnxSession _embed;
    private readonly RecoverableOnnxSession _encoder;
    private readonly RecoverableOnnxSession _decoder;
    private readonly ITextTokenizer _tokenizer;
    private readonly ModelInfo _modelInfo;
    private readonly CaptionerOptions _options;
    private readonly long[] _promptIds;
    private readonly string[] _modelFiles;
    private bool _disposed;

    private Florence2Captioner(
        RecoverableOnnxSession vision,
        RecoverableOnnxSession embed,
        RecoverableOnnxSession encoder,
        RecoverableOnnxSession decoder,
        ITextTokenizer tokenizer,
        ModelInfo modelInfo,
        CaptionerOptions options,
        string[] modelFiles)
    {
        _vision = vision;
        _embed = embed;
        _encoder = encoder;
        _decoder = decoder;
        _tokenizer = tokenizer;
        _modelInfo = modelInfo;
        _options = options;
        _modelFiles = modelFiles;
        _promptIds = BuildPromptIds(tokenizer, modelInfo, options.Detail);
    }

    /// <summary>
    /// The task prompt Florence-2 was trained on for each level of detail (its processor's
    /// <c>task_prompts_without_inputs</c> for <c>&lt;CAPTION&gt;</c>, <c>&lt;DETAILED_CAPTION&gt;</c> and
    /// <c>&lt;MORE_DETAILED_CAPTION&gt;</c>).
    /// </summary>
    internal static string TaskPrompt(CaptionDetail detail) => detail switch
    {
        CaptionDetail.Brief => "What does the image describe?",
        CaptionDetail.Detailed => "Describe in detail what is shown in the image.",
        CaptionDetail.Paragraph => "Describe with a paragraph what is shown in the image.",
        _ => throw new ArgumentOutOfRangeException(nameof(detail), detail, "Unknown caption detail.")
    };

    /// <summary>&lt;s&gt; + the task prompt's tokens + &lt;/s&gt;, as the processor's tokenizer produces them.</summary>
    internal static long[] BuildPromptIds(ITextTokenizer tokenizer, ModelInfo modelInfo, CaptionDetail detail)
    {
        var text = tokenizer.Encode(TaskPrompt(detail), addSpecialTokens: false);
        var ids = new long[text.Length + 2];
        ids[0] = modelInfo.BosTokenId;
        for (var i = 0; i < text.Length; i++)
            ids[i + 1] = text[i];
        ids[^1] = modelInfo.EosTokenId;
        return ids;
    }

    /// <inheritdoc />
    public string ModelId => _modelInfo.AliasName;

    /// <inheritdoc />
    public bool IsGpuActive => _vision.IsGpuActive;

    /// <inheritdoc />
    public IReadOnlyList<string> ActiveProviders => _vision.ActiveProviders;

    /// <inheritdoc />
    public ExecutionProvider RequestedProvider => _vision.RequestedProvider;

    /// <inheritdoc />
    public long? EstimatedMemoryBytes => _modelFiles.All(File.Exists)
        ? _modelFiles.Sum(f => new FileInfo(f).Length) * 2
        : null;

    /// <inheritdoc />
    public bool SupportsVqa => false;

    /// <inheritdoc />
    public Task WarmupAsync(CancellationToken cancellationToken = default) => Task.CompletedTask;

    /// <inheritdoc />
    public ModelInfo? GetModelInfo() => _modelInfo;

    /// <summary>
    /// Opens the four graphs. <paramref name="files"/> are the vision encoder, token embedding, text encoder and merged
    /// decoder file names in <paramref name="modelDir"/>, in that order.
    /// </summary>
    public static async Task<Florence2Captioner> CreateAsync(
        string modelDir,
        IReadOnlyList<string> files,
        ModelInfo modelInfo,
        CaptionerOptions options,
        string? tokenizerDir = null)
    {
        if (files.Count != 4)
            throw new ArgumentException("Florence-2 needs its vision encoder, token embedding, text encoder and decoder files.", nameof(files));

        var paths = files.Select(f => Path.Combine(modelDir, f)).ToArray();
        foreach (var path in paths)
        {
            if (!File.Exists(path))
                throw new ModelNotFoundException($"Model file not found: {path}", modelInfo.AliasName);
        }

        // One provider blacklist for all four graphs: a provider that crashes or hangs on one is left by the others
        // before their next run (see RecoverableOnnxSession).
        Action<SessionOptions> configure = so => so.ApplyCommonOptions(options);
        var blacklist = new ProviderBlacklist();
        string[] roles = ["vision", "embed", "encoder", "decoder"];
        var sessions = new List<RecoverableOnnxSession>(4);
        try
        {
            for (var i = 0; i < paths.Length; i++)
            {
                var result = await OnnxSessionFactory.CreateWithInfoAsync(paths[i], options.Provider, configure).ConfigureAwait(false);
                sessions.Add(RecoverableOnnxSession.FromResult(
                    result, paths[i], configure, logPrefix: $"[Florence2Captioner:{roles[i]}]", blacklist: blacklist));
            }

            var tokenizer = TokenizerFactory.CreateGpt2(tokenizerDir ?? modelDir);
            return new Florence2Captioner(sessions[0], sessions[1], sessions[2], sessions[3], tokenizer, modelInfo, options, paths);
        }
        catch
        {
            foreach (var session in sessions)
                session.Dispose();
            throw;
        }
    }

    /// <inheritdoc />
    public async Task<CaptionResult> CaptionAsync(string imagePath, CancellationToken cancellationToken = default)
    {
        var pixels = await ImagePreprocessor.Instance.PreprocessAsync(
            imagePath, _modelInfo.PreprocessProfile, cancellationToken).ConfigureAwait(false);
        return await GenerateAsync(pixels, cancellationToken).ConfigureAwait(false);
    }

    /// <inheritdoc />
    public async Task<CaptionResult> CaptionAsync(Stream imageStream, CancellationToken cancellationToken = default)
    {
        var pixels = await ImagePreprocessor.Instance.PreprocessAsync(
            imageStream, _modelInfo.PreprocessProfile, cancellationToken).ConfigureAwait(false);
        return await GenerateAsync(pixels, cancellationToken).ConfigureAwait(false);
    }

    /// <inheritdoc />
    public async Task<CaptionResult> CaptionAsync(byte[] imageData, CancellationToken cancellationToken = default)
    {
        var pixels = await ImagePreprocessor.Instance.PreprocessAsync(
            imageData, _modelInfo.PreprocessProfile, cancellationToken).ConfigureAwait(false);
        return await GenerateAsync(pixels, cancellationToken).ConfigureAwait(false);
    }

    /// <inheritdoc />
    public Task<VqaResult> AnswerAsync(string imagePath, string question, CancellationToken cancellationToken = default)
        => throw new NotSupportedException($"Model '{ModelId}' does not support visual question answering.");

    /// <inheritdoc />
    public Task<VqaResult> AnswerAsync(Stream imageStream, string question, CancellationToken cancellationToken = default)
        => throw new NotSupportedException($"Model '{ModelId}' does not support visual question answering.");

    private async Task<CaptionResult> GenerateAsync(float[] pixels, CancellationToken cancellationToken)
    {
        var imageFeatures = await RunSingleAsync(
            _vision, "pixel_values", TensorUtils.CreateImageTensor(pixels, _modelInfo.PreprocessProfile), cancellationToken).ConfigureAwait(false);
        var promptEmbeds = await EmbedAsync(_promptIds, cancellationToken).ConfigureAwait(false);

        // The text encoder reads the image features followed by the task prompt.
        var encoderInput = Concat(imageFeatures, promptEmbeds);
        var sequenceLength = encoderInput.Dimensions[1];
        var attentionMask = TensorUtils.CreateAttentionMask(sequenceLength);

        var encoderInputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("inputs_embeds", encoderInput),
            NamedOnnxValue.CreateFromTensor("attention_mask", attentionMask)
        };
        var hiddenStates = await _encoder.RunWithRecoveryAsync(
            (session, runOptions) =>
            {
                using var results = session.Run(encoderInputs, session.OutputNames, runOptions);
                return Copy(results[0]);
            },
            cancellationToken: cancellationToken).ConfigureAwait(false);

        var (tokens, confidence) = await DecodeAsync(hiddenStates, attentionMask, cancellationToken).ConfigureAwait(false);
        var caption = _tokenizer.Decode(
            tokens.Where(id => id >= FirstOrdinaryTokenId && id < FirstAddedTokenId).ToArray(), skipSpecialTokens: false);

        return new CaptionResult(caption.Trim(), confidence);
    }

    private async Task<(int[] Tokens, float Confidence)> DecodeAsync(
        DenseTensor<float> hiddenStates, DenseTensor<long> attentionMask, CancellationToken cancellationToken)
    {
        var start = _modelInfo.DecoderStartTokenId ?? _modelInfo.BosTokenId;
        var sequence = new List<int> { start };
        var generated = new List<int>();
        var past = EmptyCache();
        float totalLogProb = 0f;

        // Step 0 emits the forced <s>; the caption is the MaxLength steps after it.
        for (var step = 0; step <= _options.MaxLength; step++)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var embeds = await EmbedAsync(new long[] { sequence[^1] }, cancellationToken).ConfigureAwait(false);
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("encoder_attention_mask", attentionMask),
                NamedOnnxValue.CreateFromTensor("encoder_hidden_states", hiddenStates),
                NamedOnnxValue.CreateFromTensor("inputs_embeds", embeds),
                NamedOnnxValue.CreateFromTensor("use_cache_branch", new DenseTensor<bool>(new[] { step > 0 }, [1]))
            };
            inputs.AddRange(past.Select(kv => NamedOnnxValue.CreateFromTensor(kv.Key, kv.Value)));

            var firstStep = step == 0;
            var (logits, present) = await _decoder.RunWithRecoveryAsync(
                (session, runOptions) =>
                {
                    using var results = session.Run(inputs, session.OutputNames, runOptions);
                    var row = LastRow(results[0]);
                    var cache = new Dictionary<string, DenseTensor<float>>();
                    for (var i = 1; i < results.Count; i++)
                    {
                        var name = results[i].Name.Replace("present", "past_key_values", StringComparison.Ordinal);
                        // The cross-attention cache is computed once, from the encoder output, at the first step.
                        if (!firstStep && name.Contains(".encoder.", StringComparison.Ordinal))
                            continue;
                        cache[name] = Copy(results[i]);
                    }

                    return (row, cache);
                },
                cancellationToken: cancellationToken).ConfigureAwait(false);

            foreach (var (name, value) in present)
                past[name] = value;

            int next;
            if (firstStep)
            {
                next = _modelInfo.BosTokenId; // forced_bos_token_id
            }
            else
            {
                NextToken.BanRepeatedNgrams(logits, sequence, NoRepeatNgramSize);
                (next, var logProb) = NextToken.Choose(logits, _options);
                if (next == _modelInfo.EosTokenId)
                    break;
                totalLogProb += logProb;
                generated.Add(next);
            }

            sequence.Add(next);
        }

        var confidence = generated.Count > 0 ? MathF.Exp(totalLogProb / generated.Count) : 0f;
        return (generated.ToArray(), confidence);
    }

    private Dictionary<string, DenseTensor<float>> EmptyCache()
    {
        var cache = new Dictionary<string, DenseTensor<float>>();
        foreach (var (name, meta) in _decoder.Session.InputMetadata)
        {
            if (!name.StartsWith("past_key_values", StringComparison.Ordinal))
                continue;

            // [batch, heads, sequence, head_dim] with an empty sequence.
            var dims = meta.Dimensions;
            cache[name] = new DenseTensor<float>([1, dims[1], 0, dims[3]]);
        }

        return cache;
    }

    private Task<DenseTensor<float>> EmbedAsync(long[] ids, CancellationToken cancellationToken)
        => RunSingleAsync(_embed, "input_ids", new DenseTensor<long>(ids, [1, ids.Length]), cancellationToken);

    private static Task<DenseTensor<float>> RunSingleAsync<T>(
        RecoverableOnnxSession session, string inputName, DenseTensor<T> input, CancellationToken cancellationToken)
    {
        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(inputName, input) };
        return session.RunWithRecoveryAsync(
            (s, runOptions) =>
            {
                using var results = s.Run(inputs, s.OutputNames, runOptions);
                return Copy(results[0]);
            },
            cancellationToken: cancellationToken);
    }

    // Results live in native buffers released with the result collection; everything kept is copied out.
    private static DenseTensor<float> Copy(DisposableNamedOnnxValue value)
    {
        var tensor = value.AsTensor<float>();
        var data = tensor is DenseTensor<float> dense ? dense.Buffer.ToArray() : tensor.ToArray();
        return new DenseTensor<float>(data, tensor.Dimensions);
    }

    private static float[] LastRow(DisposableNamedOnnxValue logits)
    {
        var tensor = logits.AsTensor<float>();
        var vocab = tensor.Dimensions[^1];
        if (tensor is DenseTensor<float> dense)
            return dense.Buffer.Span[^vocab..].ToArray();
        return tensor.ToArray()[^vocab..];
    }

    private static DenseTensor<float> Concat(DenseTensor<float> first, DenseTensor<float> second)
    {
        var hidden = first.Dimensions[2];
        var data = new float[first.Buffer.Length + second.Buffer.Length];
        first.Buffer.Span.CopyTo(data);
        second.Buffer.Span.CopyTo(data.AsSpan(first.Buffer.Length));
        return new DenseTensor<float>(data, [1, first.Dimensions[1] + second.Dimensions[1], hidden]);
    }

    /// <inheritdoc />
    public ValueTask DisposeAsync()
    {
        if (_disposed)
            return ValueTask.CompletedTask;

        _vision.Dispose();
        _embed.Dispose();
        _encoder.Dispose();
        _decoder.Dispose();
        _tokenizer.Dispose();
        _disposed = true;
        return ValueTask.CompletedTask;
    }
}
