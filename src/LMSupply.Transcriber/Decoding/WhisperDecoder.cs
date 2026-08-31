using System.Diagnostics;
using System.IO.Compression;
using System.Text;
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
    private readonly InferenceSession _decoderSession;
    private readonly WhisperTokenizer _tokenizer;
    private readonly int _maxLength;

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
        InferenceSession decoderSession,
        WhisperTokenizer tokenizer,
        int maxLength = 448)
    {
        _decoderSession = decoderSession;
        _tokenizer = tokenizer;
        _maxLength = maxLength;

        // Detect input/output names from session metadata
        var inputNames = _decoderSession.InputMetadata.Keys.ToHashSet(StringComparer.OrdinalIgnoreCase);
        var outputNames = _decoderSession.OutputMetadata.Keys.ToHashSet(StringComparer.OrdinalIgnoreCase);

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
            foreach (var input in _decoderSession.InputMetadata)
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
        return await Task.Run(() =>
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

                using var results = _decoderSession.Run(inputs);
                var logits = results.First(r => r.Name == _actualLogitsOutputName).AsTensor<float>();

                // Get logits for last position - use actual model vocab size, not tokenizer
                int vocabSize;
                float[] lastLogits;

                // Handle different logits shapes:
                // - [batch, seq_len, vocab_size] - standard transformer output
                // - [batch, vocab_size] - optimized decoder that only outputs last position
                if (logits.Dimensions.Length == 3)
                {
                    vocabSize = (int)logits.Dimensions[2];
                    lastLogits = new float[vocabSize];
                    var lastPosition = (int)logits.Dimensions[1] - 1;
                    for (int i = 0; i < vocabSize; i++)
                    {
                        lastLogits[i] = logits[0, lastPosition, i];
                    }
                }
                else if (logits.Dimensions.Length == 2)
                {
                    vocabSize = (int)logits.Dimensions[1];
                    lastLogits = new float[vocabSize];
                    for (int i = 0; i < vocabSize; i++)
                    {
                        lastLogits[i] = logits[0, i];
                    }
                }
                else
                {
                    throw new InvalidOperationException($"Unexpected logits shape: [{string.Join(", ", logits.Dimensions.ToArray())}]");
                }

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
        }, cancellationToken);
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
