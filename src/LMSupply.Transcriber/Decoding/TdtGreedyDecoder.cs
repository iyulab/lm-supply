namespace LMSupply.Transcriber.Decoding;

/// <summary>
/// Result of running the prediction network + joint for one frame.
/// <see cref="Logits"/> is the raw joint output: token logits (vocabulary size) followed by duration-bin logits.
/// </summary>
internal readonly record struct TdtJointOutput(float[] Logits, float[] StateH, float[] StateC);

/// <summary>
/// Runs the joint once with the encoder output at frame <paramref name="frameIndex"/>, the last token, and the LSTM state.
/// Because it is a delegate wrapping the decoder ONNX session, the pure <see cref="TdtGreedyDecoder"/> can be tested with a fake.
/// </summary>
internal delegate TdtJointOutput TdtJointStep(int frameIndex, int lastToken, float[] stateH, float[] stateC);

/// <summary>A single decoded token, with the encoder frame it came from and its probability.</summary>
internal readonly record struct TdtToken(int Id, int Frame, float LogProb);

/// <summary>
/// Greedy decoding for the Token-and-Duration Transducer (following onnx-asr's `asr.py` rules exactly).
/// For each frame, runs the joint with (last token, state) and reads both the token and how many frames to skip.
/// Emits a token and updates the state only for non-blank outputs; when the duration is 0 it keeps emitting on the same
/// frame, but is forced forward one frame at <see cref="MaxSymbolsPerStep"/> — which is why Whisper-style runaway decoding cannot occur by construction.
/// </summary>
internal static class TdtGreedyDecoder
{
    /// <summary>Maximum number of consecutive symbols emitted on the same frame (NeMo `max_symbols`, onnx-asr default 10).</summary>
    public const int MaxSymbolsPerStep = 10;

    /// <param name="encodedLength">Number of valid encoder frames.</param>
    /// <param name="vocabSize">Length of the token logits (including blank). The logits after it are duration bins.</param>
    /// <param name="blankId">Blank token id.</param>
    /// <param name="stateSize">Number of elements in each of the LSTM h and c states (layers × hidden).</param>
    /// <param name="step">Runs the joint.</param>
    /// <param name="maxSymbolsPerStep">Maximum number of consecutive symbols on the same frame; reaching it forces a one-frame advance.</param>
    public static List<TdtToken> Decode(
        int encodedLength,
        int vocabSize,
        int blankId,
        int stateSize,
        TdtJointStep step,
        int maxSymbolsPerStep = MaxSymbolsPerStep)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(encodedLength);
        ArgumentOutOfRangeException.ThrowIfLessThan(vocabSize, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(maxSymbolsPerStep, 1);

        var tokens = new List<TdtToken>();
        var stateH = new float[stateSize];
        var stateC = new float[stateSize];
        var t = 0;
        var emittedAtFrame = 0;

        while (t < encodedLength)
        {
            var last = tokens.Count > 0 ? tokens[^1].Id : blankId;
            var joint = step(t, last, stateH, stateC);
            var logits = joint.Logits;
            if (logits.Length <= vocabSize)
                throw new InvalidDataException($"Joint output has {logits.Length} logits but at least {vocabSize + 1} were expected (tokens + duration bins).");

            var token = ArgMax(logits, 0, vocabSize);
            var duration = ArgMax(logits, vocabSize, logits.Length) - vocabSize;

            if (token != blankId)
            {
                stateH = joint.StateH;
                stateC = joint.StateC;
                tokens.Add(new TdtToken(token, t, LogSoftmaxAt(logits, vocabSize, token)));
                emittedAtFrame++;
            }

            if (duration > 0)
            {
                t += duration;
                emittedAtFrame = 0;
            }
            else if (token == blankId || emittedAtFrame >= maxSymbolsPerStep)
            {
                t += 1;
                emittedAtFrame = 0;
            }
        }

        return tokens;
    }

    private static int ArgMax(float[] values, int start, int end)
    {
        var best = start;
        for (var i = start + 1; i < end; i++)
            if (values[i] > values[best]) best = i;
        return best;
    }

    private static float LogSoftmaxAt(float[] logits, int count, int index)
    {
        var max = float.NegativeInfinity;
        for (var i = 0; i < count; i++) max = Math.Max(max, logits[i]);
        double sum = 0;
        for (var i = 0; i < count; i++) sum += Math.Exp(logits[i] - max);
        return (float)(logits[index] - max - Math.Log(sum));
    }
}
