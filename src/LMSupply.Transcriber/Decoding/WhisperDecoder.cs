using System.Diagnostics;
using System.IO.Compression;
using System.Text;
using LMSupply.Inference;
using LMSupply.Transcriber.Internal;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace LMSupply.Transcriber.Decoding;

/// <summary>
/// Whisper autoregressive decoder using greedy search.
/// Supports both standard and merged (with KV cache) decoder models.
/// </summary>
internal sealed class WhisperDecoder
{
    // Each decode step runs through the session's recovery path: a provider crash or a hang inside
    // one step moves the session to the next provider and retries that step. Decoder state (the
    // token list and the KV cache) lives on the managed side, so a mid-loop provider switch is
    // transparent to the loop.
    private readonly RecoverableOnnxSession _decoderSession;
    private readonly WhisperTokenizer _tokenizer;
    private readonly int _maxLength;

    // Repetition-cycle detection (docket iyulab/lm-supply#59). A greedy decoder that has lost the
    // audio settles into a *periodic* token cycle, not a run of one identical token -- and the
    // three-identical-token guard in SelectNextToken cannot see it. Worse, a captured trace showed
    // that guard *stabilising* one: suppressing the repeated token at every third occurrence is
    // exactly what turns ".. x x x y x x x y .." into a fixed point. These thresholds decide when a
    // tail is degenerate enough to end the chunk on. Deliberately conservative -- a unit must both
    // repeat MinCycleRepeats times and cover MinCycleTokens generated tokens, so a single stutter
    // ("the the the") still decodes on, as the 0.42.6 EOT penalty intends.
    private const int MaxCyclePeriod = 8;
    private const int MinCycleRepeats = 3;
    private const int MinCycleTokens = 12;

    // Input/output names for onnx-community models
    private const string InputTokensName = "input_ids";
    private const string InputEncoderHiddenStates = "encoder_hidden_states";
    private const string OutputLogitsName = "logits";
    private const string UseCacheBranchName = "use_cache_branch";

    // Alternative names used by some ONNX exports
    private static readonly string[] TokenInputNames = ["input_ids", "tokens", "decoder_input_ids"];
    private static readonly string[] EncoderInputNames = ["encoder_hidden_states", "audio", "encoder_outputs"];

    // Cached constant arrays for tensor creation (CA1861)
    private static readonly bool[] s_falseArray = [false];
    private static readonly int[] s_oneDimension = [1];

    private readonly string _actualTokenInputName;
    private readonly string _actualEncoderInputName;
    private readonly string _actualLogitsOutputName;

    // Merged model support
    private readonly bool _isMergedModel;
    private readonly List<(string Name, int[] Dims)> _pastKeyValueInputs = [];
    private readonly int _numAttentionHeads;
    private readonly int _headDim;

    /// <summary>
    /// Creates a decoder for testing probability computation only (no ONNX session).
    /// </summary>
    internal static WhisperDecoder CreateForTesting(WhisperTokenizer tokenizer)
    {
        return new WhisperDecoder(tokenizer);
    }

    private WhisperDecoder(WhisperTokenizer tokenizer)
    {
        _decoderSession = null!;
        _tokenizer = tokenizer;
        _maxLength = 0;
        _actualTokenInputName = "";
        _actualEncoderInputName = "";
        _actualLogitsOutputName = "";
    }

    public WhisperDecoder(
        RecoverableOnnxSession decoderSession,
        WhisperTokenizer tokenizer,
        int maxLength = 448)
    {
        _decoderSession = decoderSession;
        _tokenizer = tokenizer;
        _maxLength = maxLength;

        // Detect input/output names from session metadata (identical on every provider, so reading
        // it from the current underlying session once is safe across later recoveries)
        var metadataSession = decoderSession.Session;
        var inputNames = metadataSession.InputMetadata.Keys.ToHashSet(StringComparer.OrdinalIgnoreCase);
        var outputNames = metadataSession.OutputMetadata.Keys.ToHashSet(StringComparer.OrdinalIgnoreCase);

        _actualTokenInputName = TokenInputNames.FirstOrDefault(n => inputNames.Contains(n))
            ?? throw new InvalidOperationException(
                $"Could not find token input. Available inputs: {string.Join(", ", inputNames)}");

        _actualEncoderInputName = EncoderInputNames.FirstOrDefault(n => inputNames.Contains(n))
            ?? throw new InvalidOperationException(
                $"Could not find encoder input. Available inputs: {string.Join(", ", inputNames)}");

        _actualLogitsOutputName = outputNames.Contains(OutputLogitsName) ? OutputLogitsName
            : outputNames.FirstOrDefault(n => n.Contains("logit", StringComparison.OrdinalIgnoreCase))
            ?? outputNames.First();

        // Detect if this is a merged model (has use_cache_branch input)
        _isMergedModel = inputNames.Contains(UseCacheBranchName);

        if (_isMergedModel)
        {
            // Collect past_key_values input metadata
            foreach (var input in metadataSession.InputMetadata)
            {
                if (input.Key.StartsWith("past_key_values", StringComparison.OrdinalIgnoreCase))
                {
                    var dims = input.Value.Dimensions;
                    _pastKeyValueInputs.Add((input.Key, dims));

                    // Extract attention head info from first past_key_values tensor
                    // Shape is typically [batch, num_heads, seq_len, head_dim]
                    if (_numAttentionHeads == 0 && dims.Length >= 4)
                    {
                        _numAttentionHeads = dims[1] > 0 ? dims[1] : 8; // Default to 8 if dynamic
                        _headDim = dims[3] > 0 ? dims[3] : 64; // Default to 64 if dynamic
                    }
                }
            }

            // Sort by name for consistent ordering
            _pastKeyValueInputs.Sort((a, b) => string.Compare(a.Name, b.Name, StringComparison.Ordinal));
        }
    }

    /// <summary>
    /// Decodes encoder output to text using greedy search.
    /// </summary>
    public async Task<DecodingResult> DecodeAsync(
        float[] encoderOutput,
        int encoderSequenceLength,
        int hiddenSize,
        double chunkDurationSeconds,
        TranscribeOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        // Initialize tokens with SOT sequence
        var useTimestamps = options?.WordTimestamps ?? false;
        var translate = options?.Translate ?? false;
        var initialTokens = _tokenizer.GetSotSequence(options?.Language, useTimestamps, translate);
        var tokens = new List<int>(initialTokens);

        var segments = new List<TranscriptionSegment>();
        var currentSegmentTokens = new List<int>();
        var currentSegmentStart = 0.0;

        // Create encoder output tensor [1, seq_len, hidden_size]
        var encoderTensor = new DenseTensor<float>(
            encoderOutput,
            [1, encoderSequenceLength, hiddenSize]);

        string? detectedLanguage = null;
        float? languageProbability = null;
        int segmentId = 0;

        // Per-segment metric tracking
        var currentSegmentLogProbs = new List<float>();
        float? chunkNoSpeechProb = null;

        // For merged models, we'll track KV cache state
        Dictionary<string, DenseTensor<float>>? kvCache = null;

        if (_isMergedModel)
        {
            // Initialize empty KV cache tensors
            kvCache = CreateEmptyKvCache();
        }

        // Autoregressive generation loop
        while (tokens.Count < _maxLength)
        {
            cancellationToken.ThrowIfCancellationRequested();

            // Create input tensor
            var tokenArray = tokens.ToArray();
            var tokenTensor = new DenseTensor<long>(
                tokenArray.Select(t => (long)t).ToArray(),
                [1, tokens.Count]);

            // Build inputs list
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(_actualTokenInputName, tokenTensor),
                NamedOnnxValue.CreateFromTensor(_actualEncoderInputName, encoderTensor)
            };

            // Add merged model specific inputs
            if (_isMergedModel && kvCache != null)
            {
                // Add use_cache_branch = false (we're not using cache efficiently yet)
                var useCacheTensor = new DenseTensor<bool>(s_falseArray, s_oneDimension);
                inputs.Add(NamedOnnxValue.CreateFromTensor(UseCacheBranchName, useCacheTensor));

                // Add past_key_values tensors
                foreach (var (name, _) in _pastKeyValueInputs)
                {
                    if (kvCache.TryGetValue(name, out var tensor))
                    {
                        inputs.Add(NamedOnnxValue.CreateFromTensor(name, tensor));
                    }
                }
            }

            // One bounded decode step. The bound (and provider recovery) applies per step: a cold
            // GPU kernel hang is a property of a single native call, and a long transcript that
            // legitimately takes many steps must not be cut off by a whole-loop timeout.
            var lastLogits = await _decoderSession.RunWithRecoveryAsync((session, runOptions) =>
            {
                using var results = session.Run(inputs, [_actualLogitsOutputName], runOptions);
                return ExtractLastPositionLogits(results[0].AsTensor<float>());
            }, cancellationToken: cancellationToken);

            // Greedy selection: argmax, after repetition-penalty/temperature/hallucination-guard
            var nextToken = SelectNextToken(lastLogits, tokens, initialTokens, options);

            // Compute log probability of selected token for AvgLogProb metric
            if (!_tokenizer.IsSpecialToken(nextToken))
            {
                currentSegmentLogProbs.Add(ComputeLogProb(lastLogits, nextToken));
            }

            // Detect language from first generated token after SOT
            if (tokens.Count == initialTokens.Length)
            {
                // Compute no-speech probability at the first decoder step
                chunkNoSpeechProb = ComputeNoSpeechProb(lastLogits);

                if (_tokenizer.IsLanguageToken(nextToken))
                {
                    detectedLanguage = _tokenizer.GetLanguageFromToken(nextToken);
                    languageProbability = ComputeLanguageTokenProbability(lastLogits, nextToken);
                }
            }

            // Check for end of text
            if (nextToken == _tokenizer.EndOfTextToken)
            {
                Trace.TraceInformation($"[WhisperDecoder] EOT at step {tokens.Count - initialTokens.Length}, totalTokens={tokens.Count}");
                break;
            }

            // Handle timestamp tokens
            if (_tokenizer.IsTimestampToken(nextToken))
            {
                var timestamp = _tokenizer.TimestampTokenToSeconds(nextToken);

                // Start timestamp
                if (currentSegmentTokens.Count == 0)
                {
                    currentSegmentStart = timestamp;
                }
                else
                {
                    // End timestamp - create segment
                    var segmentText = _tokenizer.Decode(
                        currentSegmentTokens.ToArray().AsSpan(),
                        skipSpecialTokens: true);

                    if (!string.IsNullOrWhiteSpace(segmentText))
                    {
                        var trimmedText = segmentText.Trim();
                        segments.Add(new TranscriptionSegment
                        {
                            Id = segmentId++,
                            Start = currentSegmentStart,
                            End = timestamp,
                            Text = trimmedText,
                            AvgLogProb = currentSegmentLogProbs.Count > 0
                                ? currentSegmentLogProbs.Average() : null,
                            NoSpeechProb = chunkNoSpeechProb,
                            CompressionRatio = SegmentPostProcessor.ComputeCompressionRatio(trimmedText)
                        });
                    }

                    currentSegmentLogProbs.Clear();
                    currentSegmentTokens.Clear();
                }
            }
            else if (!_tokenizer.IsSpecialToken(nextToken))
            {
                // Regular text token
                currentSegmentTokens.Add(nextToken);
            }

            tokens.Add(nextToken);

            // A detected cycle means the decoder stopped tracking the audio: everything from where
            // the cycle began is hallucination, so end the chunk here instead of grinding on to
            // _maxLength. Ending (rather than penalising yet another token) is the only exit that
            // works -- the cycle is a fixed point of the greedy selection, so any per-token nudge
            // is absorbed by the next turn of the same loop.
            if (TryDetectRepetitionCycle(tokens, initialTokens.Length, out var cycleStart))
            {
                Trace.TraceWarning(
                    $"[WhisperDecoder] Repetition cycle detected at step {tokens.Count - initialTokens.Length}; " +
                    $"ending chunk and discarding {tokens.Count - cycleStart} degenerate tokens.");

                // Drop the degenerate tail from the pending segment, keeping whatever real text
                // preceded it. Only non-special tokens ever reached currentSegmentTokens, so the
                // discard count is the number of those within the cyclic span.
                var degenerateTextTokens = 0;
                for (int i = cycleStart; i < tokens.Count; i++)
                {
                    if (!_tokenizer.IsSpecialToken(tokens[i]))
                        degenerateTextTokens++;
                }

                var trim = Math.Min(degenerateTextTokens, currentSegmentTokens.Count);
                currentSegmentTokens.RemoveRange(currentSegmentTokens.Count - trim, trim);
                break;
            }
        }

        FinalizeSegments(
            segments,
            tokens,
            initialTokens,
            currentSegmentTokens,
            currentSegmentLogProbs,
            segmentId,
            currentSegmentStart,
            chunkNoSpeechProb,
            chunkDurationSeconds);

        // Combine all segment texts for full transcription
        var fullTranscription = string.Join(" ", segments.Select(s => s.Text));

        return new DecodingResult
        {
            Text = fullTranscription,
            Language = detectedLanguage ?? options?.Language ?? "en",
            LanguageProbability = languageProbability,
            Segments = segments,
            TokenCount = tokens.Count - initialTokens.Length
        };
    }

    /// <summary>
    /// Copies the last position's logits out of a decoder output so the native result can be
    /// disposed before the step returns. Handles both export shapes: <c>[batch, seq_len, vocab]</c>
    /// (standard transformer output) and <c>[batch, vocab]</c> (decoder that only emits the last
    /// position). The vocab size is the model's, not the tokenizer's.
    /// </summary>
    private static float[] ExtractLastPositionLogits(Tensor<float> logits)
    {
        if (logits.Dimensions.Length == 3)
        {
            var vocabSize = (int)logits.Dimensions[2];
            var lastLogits = new float[vocabSize];
            var lastPosition = (int)logits.Dimensions[1] - 1;
            for (int i = 0; i < vocabSize; i++)
            {
                lastLogits[i] = logits[0, lastPosition, i];
            }
            return lastLogits;
        }

        if (logits.Dimensions.Length == 2)
        {
            var vocabSize = (int)logits.Dimensions[1];
            var lastLogits = new float[vocabSize];
            for (int i = 0; i < vocabSize; i++)
            {
                lastLogits[i] = logits[0, i];
            }
            return lastLogits;
        }

        throw new InvalidOperationException($"Unexpected logits shape: [{string.Join(", ", logits.Dimensions.ToArray())}]");
    }

    /// <summary>
    /// Applies repetition penalty, temperature scaling, and the hallucination-suppression guard to
    /// one decode step's raw logits, then selects the next token via greedy argmax. Mutates
    /// <paramref name="logits"/> in place. Extracted from the decode loop (same reasoning as
    /// <see cref="FinalizeSegments"/>) so a captured decode-step logit vector can exercise this
    /// exact selection logic directly, without a real ONNX session.
    /// </summary>
    internal int SelectNextToken(float[] logits, List<int> tokens, int[] initialTokens, TranscribeOptions? options)
    {
        // Apply repetition penalty to discourage repeating tokens
        const float repetitionPenalty = 1.2f;
        var recentTokens = tokens.Skip(Math.Max(0, tokens.Count - 10)).ToHashSet();
        for (int i = 0; i < logits.Length; i++)
        {
            if (recentTokens.Contains(i))
            {
                // Penalize recently used tokens
                if (logits[i] > 0)
                    logits[i] /= repetitionPenalty;
                else
                    logits[i] *= repetitionPenalty;
            }
        }

        // Apply temperature if specified
        if (options is { Temperature: > 0 and < 1 })
        {
            for (int i = 0; i < logits.Length; i++)
            {
                logits[i] /= options.Temperature;
            }
        }

        // Suppress specific tokens that cause hallucination loops
        if (tokens.Count > initialTokens.Length + 3)
        {
            var last3 = tokens.Skip(tokens.Count - 3).ToArray();
            if (last3[0] == last3[1] && last3[1] == last3[2])
            {
                // Three consecutive same tokens - heavily suppress this token
                var repeatedToken = last3[0];
                if (repeatedToken < logits.Length)
                {
                    logits[repeatedToken] = float.NegativeInfinity;
                }

                // A hard-suppression event means the model was mid-hallucination; end-of-text is
                // the other easy way out at exactly this point, and it was previously untouched
                // by any penalty (it is never a "recently used" token). See docket
                // iyulab/lm-supply#59: a captured decode trace showed EOT beating the real
                // continuation token by a margin as small as 0.165 immediately after this guard
                // fired. Give EOT the same soft penalty a recently-used token already gets here,
                // so it has to clearly beat a real continuation rather than merely edge it out
                // right when the decoder was caught repeating itself.
                var eot = _tokenizer.EndOfTextToken;
                if (eot < logits.Length)
                {
                    if (logits[eot] > 0)
                        logits[eot] /= repetitionPenalty;
                    else
                        logits[eot] *= repetitionPenalty;
                }
            }
        }

        return ArgMax(logits);
    }

    /// <summary>
    /// Reports whether the generated token tail has collapsed into a repeating cycle, and if so
    /// where that cycle starts in <paramref name="tokens"/>.
    /// </summary>
    /// <remarks>
    /// Finds the shortest period (the fundamental one) whose repetition explains the tail, then
    /// extends that period backwards as far as it holds. A tail qualifies only when the unit
    /// repeats at least <see cref="MinCycleRepeats"/> times <i>and</i> spans at least
    /// <see cref="MinCycleTokens"/> tokens, so ordinary repeated words are never mistaken for a
    /// decoder collapse: a period-1 tail has to be a dozen identical tokens, not three.
    /// <para>
    /// Only generated tokens are considered -- the prompt/SOT prefix is never part of a cycle.
    /// </para>
    /// </remarks>
    internal static bool TryDetectRepetitionCycle(
        IReadOnlyList<int> tokens,
        int generatedStart,
        out int cycleStartIndex)
    {
        cycleStartIndex = -1;
        var generated = tokens.Count - generatedStart;
        if (generated < MinCycleTokens)
            return false;

        for (int period = 1; period <= MaxCyclePeriod; period++)
        {
            if (generated < period * MinCycleRepeats)
                break;

            // The last `period` tokens are the candidate unit; walk backwards while each earlier
            // token still matches the one a full period ahead of it.
            var length = period;
            while (generated - length - 1 >= 0 &&
                   tokens[generatedStart + generated - length - 1] ==
                   tokens[generatedStart + generated - length - 1 + period])
            {
                length++;
            }

            if (length >= MinCycleTokens && length / period >= MinCycleRepeats)
            {
                cycleStartIndex = tokens.Count - length;
                return true;
            }
        }

        return false;
    }

    /// <summary>
    /// Creates empty KV cache tensors for merged model initialization.
    /// </summary>
    private Dictionary<string, DenseTensor<float>> CreateEmptyKvCache()
    {
        var cache = new Dictionary<string, DenseTensor<float>>();

        foreach (var (name, dims) in _pastKeyValueInputs)
        {
            // Create zero-sized tensor for initial state
            // Shape: [batch=1, num_heads, seq_len=0, head_dim]
            var numHeads = dims[1] > 0 ? dims[1] : _numAttentionHeads;
            var headDim = dims[3] > 0 ? dims[3] : _headDim;

            var tensor = new DenseTensor<float>([1, numHeads, 0, headDim]);
            cache[name] = tensor;
        }

        return cache;
    }

    /// <summary>
    /// Flushes any segment still open when the decode loop ended (no closing timestamp token was
    /// generated), or — if no segment was ever opened at all (no-timestamps mode) — builds one
    /// segment from the full decoded token stream. Pure token/text logic with no ONNX involved,
    /// so it is independently unit-testable without a real decoder session.
    /// </summary>
    internal void FinalizeSegments(
        List<TranscriptionSegment> segments,
        List<int> tokens,
        int[] initialTokens,
        List<int> currentSegmentTokens,
        List<float> currentSegmentLogProbs,
        int segmentId,
        double currentSegmentStart,
        float? chunkNoSpeechProb,
        double chunkDurationSeconds)
    {
        // Handle remaining tokens as final segment
        if (currentSegmentTokens.Count > 0)
        {
            var segmentText = _tokenizer.Decode(
                currentSegmentTokens.ToArray().AsSpan(),
                skipSpecialTokens: true);

            if (!string.IsNullOrWhiteSpace(segmentText))
            {
                var trimmedText = segmentText.Trim();
                segments.Add(new TranscriptionSegment
                {
                    Id = segmentId,
                    Start = currentSegmentStart,
                    // No timestamp token was ever generated for this segment (decoder hit EOT
                    // before emitting one) — fall back to the chunk's actual pre-padding
                    // duration rather than the fixed 30s the encoder always pads/truncates to.
                    End = chunkDurationSeconds,
                    Text = trimmedText,
                    AvgLogProb = currentSegmentLogProbs.Count > 0
                        ? currentSegmentLogProbs.Average() : null,
                    NoSpeechProb = chunkNoSpeechProb,
                    CompressionRatio = SegmentPostProcessor.ComputeCompressionRatio(trimmedText)
                });
            }
        }

        // If no segments created (no timestamps mode), create single segment
        if (segments.Count == 0)
        {
            var allTokens = tokens.Skip(initialTokens.Length).ToArray();
            var fullText = _tokenizer.Decode(allTokens.AsSpan(), skipSpecialTokens: true);

            if (!string.IsNullOrWhiteSpace(fullText))
            {
                var trimmedText = fullText.Trim();
                segments.Add(new TranscriptionSegment
                {
                    Id = 0,
                    Start = 0,
                    // No timestamp tokens at all (no-timestamps mode, or decoder hit EOT
                    // immediately) — same fallback as above.
                    End = chunkDurationSeconds,
                    Text = trimmedText,
                    AvgLogProb = currentSegmentLogProbs.Count > 0
                        ? currentSegmentLogProbs.Average() : null,
                    NoSpeechProb = chunkNoSpeechProb,
                    CompressionRatio = SegmentPostProcessor.ComputeCompressionRatio(trimmedText)
                });
            }
        }
    }

    /// <summary>
    /// Computes the softmax probability of the selected language token
    /// over all language tokens in the logits.
    /// </summary>
    internal float ComputeLanguageTokenProbability(float[] logits, int selectedToken)
    {
        var start = _tokenizer.LanguageTokenStart;
        var end = Math.Min(_tokenizer.LanguageTokenEnd, logits.Length - 1);

        if (start > end || selectedToken < start || selectedToken > end)
            return 0f;

        // Numerically stable softmax: subtract max before exp
        var maxLogit = float.NegativeInfinity;
        for (int i = start; i <= end; i++)
        {
            if (logits[i] > maxLogit) maxLogit = logits[i];
        }

        float sumExp = 0f;
        for (int i = start; i <= end; i++)
        {
            sumExp += MathF.Exp(logits[i] - maxLogit);
        }

        return MathF.Exp(logits[selectedToken] - maxLogit) / sumExp;
    }

    /// <summary>
    /// Computes log probability of the selected token using log-softmax.
    /// </summary>
    private static float ComputeLogProb(float[] logits, int token)
    {
        // log P(token) = logits[token] - log(sum(exp(logits)))
        // For numerical stability: subtract max first
        var max = float.NegativeInfinity;
        for (int i = 0; i < logits.Length; i++)
        {
            if (logits[i] > max) max = logits[i];
        }

        float sumExp = 0f;
        for (int i = 0; i < logits.Length; i++)
        {
            sumExp += MathF.Exp(logits[i] - max);
        }

        return (logits[token] - max) - MathF.Log(sumExp);
    }

    /// <summary>
    /// Computes the probability that the current audio chunk contains no speech.
    /// Uses the no_speech token (50362) softmax probability over the full vocabulary.
    /// </summary>
    private float ComputeNoSpeechProb(float[] logits)
    {
        var noSpeechToken = _tokenizer.NoSpeechToken;
        if (noSpeechToken >= logits.Length)
            return 0f;

        var max = float.NegativeInfinity;
        for (int i = 0; i < logits.Length; i++)
        {
            if (logits[i] > max) max = logits[i];
        }

        float sumExp = 0f;
        for (int i = 0; i < logits.Length; i++)
        {
            sumExp += MathF.Exp(logits[i] - max);
        }

        return MathF.Exp(logits[noSpeechToken] - max) / sumExp;
    }

    private static int ArgMax(float[] values)
    {
        int maxIndex = 0;
        float maxValue = values[0];

        for (int i = 1; i < values.Length; i++)
        {
            if (values[i] > maxValue)
            {
                maxValue = values[i];
                maxIndex = i;
            }
        }

        return maxIndex;
    }
}

/// <summary>
/// Result from the Whisper decoder.
/// </summary>
internal sealed class DecodingResult
{
    public required string Text { get; init; }
    public required string Language { get; init; }
    public float? LanguageProbability { get; init; }
    public required List<TranscriptionSegment> Segments { get; init; }
    public int TokenCount { get; init; }
}
