using LMSupply.Inference;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace LMSupply.Translator.Decoding;

/// <summary>
/// Beam search decoding for sequence generation.
/// Maintains multiple hypotheses and selects the best overall sequences.
/// </summary>
internal sealed class BeamSearchDecoder
{
    private static readonly bool[] s_falseArray = [false];
    private static readonly int[] s_oneDimension = [1];

    private readonly int _beamSize;
    private readonly int _maxLength;
    private readonly int _eosTokenId;
    private readonly int _padTokenId;
    private readonly float _lengthPenalty;
    private readonly float _repetitionPenalty;

    /// <summary>
    /// Initializes a new instance of the BeamSearchDecoder.
    /// </summary>
    /// <param name="beamSize">Number of beams to maintain.</param>
    /// <param name="maxLength">Maximum output sequence length.</param>
    /// <param name="eosTokenId">End of sequence token ID.</param>
    /// <param name="padTokenId">Padding token ID.</param>
    /// <param name="lengthPenalty">Length penalty factor (alpha). Values > 1.0 favor longer sequences.</param>
    /// <param name="repetitionPenalty">Repetition penalty factor. Values > 1.0 penalize repeated tokens.</param>
    public BeamSearchDecoder(
        int beamSize = 4,
        int maxLength = 512,
        int eosTokenId = 0,
        int padTokenId = 0,
        float lengthPenalty = 1.0f,
        float repetitionPenalty = 1.0f)
    {
        _beamSize = beamSize;
        _maxLength = maxLength;
        _eosTokenId = eosTokenId;
        _padTokenId = padTokenId;
        _lengthPenalty = lengthPenalty;
        _repetitionPenalty = repetitionPenalty;
    }

    /// <summary>
    /// Performs beam search decoding.
    /// </summary>
    /// <param name="encoderOutput">Encoder hidden states.</param>
    /// <param name="encoderAttentionMask">Encoder attention mask.</param>
    /// <param name="startTokenIds">Initial decoder input token IDs.</param>
    /// <param name="decoderSession">
    /// Decoder session. Each decode step runs through its recovery path: a provider crash or a hang
    /// inside one step moves the session to the next provider and retries that step — beam state is
    /// on the managed side, so the switch is transparent to the search.
    /// </param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Best decoded sequence.</returns>
    public async Task<long[]> DecodeAsync(
        DenseTensor<float> encoderOutput,
        long[] encoderAttentionMask,
        long[] startTokenIds,
        RecoverableOnnxSession decoderSession,
        CancellationToken cancellationToken = default)
    {
        // Initialize beams with start tokens
        var beams = new List<BeamHypothesis>
        {
            new(startTokenIds.ToList(), score: 0.0f, isFinished: false)
        };

        var finishedBeams = new List<BeamHypothesis>();

        // use_cache_branch is required by merged decoder exports; always false (no KV-caching).
        // Metadata is identical on every provider, so reading it once is safe across recoveries.
        var needsCacheBranch = decoderSession.Session.InputMetadata.ContainsKey("use_cache_branch");

        for (int step = 0; step < _maxLength && beams.Count > 0; step++)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var allCandidates = new List<BeamHypothesis>();

            foreach (var beam in beams)
            {
                if (beam.IsFinished)
                {
                    finishedBeams.Add(beam);
                    continue;
                }

                // Run decoder for this beam
                var logits = await RunDecoderStepAsync(
                    beam.TokenIds,
                    encoderOutput,
                    encoderAttentionMask,
                    needsCacheBranch,
                    decoderSession,
                    cancellationToken);

                // Apply repetition penalty
                ApplyRepetitionPenalty(logits, beam.TokenIds);

                // Get top-k candidates for this beam
                var topK = GetTopK(logits, _beamSize * 2);

                foreach (var (tokenId, logProb) in topK)
                {
                    var newTokenIds = new List<long>(beam.TokenIds) { tokenId };
                    var newScore = beam.Score + logProb;

                    var isFinished = tokenId == _eosTokenId;
                    if (isFinished)
                    {
                        // Apply length penalty for finished sequences
                        var normalizedScore = ApplyLengthPenalty(newScore, newTokenIds.Count);
                        finishedBeams.Add(new BeamHypothesis(newTokenIds, normalizedScore, isFinished: true));
                    }
                    else
                    {
                        allCandidates.Add(new BeamHypothesis(newTokenIds, newScore, isFinished: false));
                    }
                }
            }

            // Select top beams for next step
            beams = allCandidates
                .OrderByDescending(b => b.Score)
                .Take(_beamSize)
                .ToList();

            // Early stopping if we have enough finished beams
            if (finishedBeams.Count >= _beamSize)
            {
                var minFinishedScore = finishedBeams
                    .OrderByDescending(b => b.Score)
                    .Take(_beamSize)
                    .Min(b => b.Score);

                // If best active beam can't beat worst finished beam, stop
                if (beams.Count == 0 || beams[0].Score < minFinishedScore)
                {
                    break;
                }
            }
        }

        // Add any remaining unfinished beams to finished (with EOS appended)
        foreach (var beam in beams)
        {
            var finalTokens = new List<long>(beam.TokenIds) { _eosTokenId };
            var normalizedScore = ApplyLengthPenalty(beam.Score, finalTokens.Count);
            finishedBeams.Add(new BeamHypothesis(finalTokens, normalizedScore, isFinished: true));
        }

        // Return best hypothesis
        var bestBeam = finishedBeams
            .OrderByDescending(b => b.Score)
            .FirstOrDefault();

        return bestBeam?.TokenIds.ToArray() ?? startTokenIds;
    }

    /// <summary>
    /// One bounded, recoverable decoder step. Returns the last position's log-probabilities.
    /// </summary>
    private static Task<float[]> RunDecoderStepAsync(
        List<long> inputIds,
        DenseTensor<float> encoderOutput,
        long[] encoderAttentionMask,
        bool needsCacheBranch,
        RecoverableOnnxSession decoderSession,
        CancellationToken cancellationToken)
    {
        var decoderInputIds = new DenseTensor<long>(inputIds.ToArray(), [1, inputIds.Count]);
        var encoderAttention = new DenseTensor<long>(encoderAttentionMask, [1, encoderAttentionMask.Length]);

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input_ids", decoderInputIds),
            NamedOnnxValue.CreateFromTensor("encoder_attention_mask", encoderAttention),
            NamedOnnxValue.CreateFromTensor("encoder_hidden_states", encoderOutput)
        };

        if (needsCacheBranch)
        {
            var useCacheBranch = new DenseTensor<bool>(s_falseArray, s_oneDimension);
            inputs.Add(NamedOnnxValue.CreateFromTensor("use_cache_branch", useCacheBranch));
        }

        var lastPosition = inputIds.Count - 1;

        return decoderSession.RunWithRecoveryAsync((session, runOptions) =>
        {
            using var outputs = session.Run(inputs, [session.OutputNames[0]], runOptions);
            var logitsTensor = outputs[0].AsTensor<float>();

            // Get logits for last position
            var vocabSize = (int)logitsTensor.Dimensions[2];
            var logits = new float[vocabSize];

            for (int v = 0; v < vocabSize; v++)
            {
                logits[v] = logitsTensor[0, lastPosition, v];
            }

            // Convert to log probabilities (log softmax)
            var maxLogit = logits.Max();
            var expSum = logits.Select(x => MathF.Exp(x - maxLogit)).Sum();
            var logSumExp = maxLogit + MathF.Log(expSum);

            for (int i = 0; i < vocabSize; i++)
            {
                logits[i] = logits[i] - logSumExp;
            }

            return logits;
        }, cancellationToken: cancellationToken);
    }

    private void ApplyRepetitionPenalty(float[] logits, List<long> previousTokens)
    {
        if (_repetitionPenalty == 1.0f)
            return;

        var tokenSet = previousTokens.ToHashSet();

        foreach (var tokenId in tokenSet)
        {
            if (tokenId >= 0 && tokenId < logits.Length)
            {
                var idx = (int)tokenId;
                if (logits[idx] > 0)
                {
                    logits[idx] /= _repetitionPenalty;
                }
                else
                {
                    logits[idx] *= _repetitionPenalty;
                }
            }
        }
    }

    private float ApplyLengthPenalty(float score, int length)
    {
        if (_lengthPenalty == 1.0f)
            return score;

        // Normalize by length^alpha
        return score / MathF.Pow(length, _lengthPenalty);
    }

    private static List<(long TokenId, float LogProb)> GetTopK(float[] logProbs, int k)
    {
        return logProbs
            .Select((prob, idx) => (TokenId: (long)idx, LogProb: prob))
            .OrderByDescending(x => x.LogProb)
            .Take(k)
            .ToList();
    }

    /// <summary>
    /// Represents a beam hypothesis during beam search.
    /// </summary>
    private sealed class BeamHypothesis
    {
        public List<long> TokenIds { get; }
        public float Score { get; }
        public bool IsFinished { get; }

        public BeamHypothesis(List<long> tokenIds, float score, bool isFinished)
        {
            TokenIds = tokenIds;
            Score = score;
            IsFinished = isFinished;
        }
    }
}
