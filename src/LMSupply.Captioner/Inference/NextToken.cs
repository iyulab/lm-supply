using LMSupply.Vision;

namespace LMSupply.Captioner.Inference;

/// <summary>
/// Picks the next caption token from a logits row, the same way for every captioner: greedy when
/// <see cref="CaptionerOptions.Temperature"/> is not positive or <see cref="CaptionerOptions.NumBeams"/> is 1, otherwise
/// temperature sampling.
/// </summary>
internal static class NextToken
{
    public static (int Token, float LogProb) Choose(ReadOnlySpan<float> logits, CaptionerOptions options)
    {
        if (options.Temperature <= 0 || options.NumBeams == 1)
        {
            var token = TensorUtils.ArgMax(logits);
            var probs = TensorUtils.Softmax(logits);
            return (token, MathF.Log(probs[token] + 1e-10f));
        }

        var sampled = TensorUtils.Softmax(logits, options.Temperature);
        var next = TensorUtils.SampleFromDistribution(sampled);
        return (next, MathF.Log(sampled[next] + 1e-10f));
    }

    /// <summary>
    /// Bans every token that would repeat an n-gram of <paramref name="size"/> already in <paramref name="sequence"/>
    /// (the <c>no_repeat_ngram_size</c> rule of Hugging Face generation).
    /// </summary>
    public static void BanRepeatedNgrams(Span<float> logits, IReadOnlyList<int> sequence, int size)
    {
        if (size <= 0 || sequence.Count < size - 1)
            return;

        var prefixStart = sequence.Count - (size - 1);
        for (var start = 0; start + size - 1 < sequence.Count; start++)
        {
            var match = true;
            for (var k = 0; k < size - 1; k++)
            {
                if (sequence[start + k] != sequence[prefixStart + k])
                {
                    match = false;
                    break;
                }
            }

            if (match)
                logits[sequence[start + size - 1]] = float.NegativeInfinity;
        }
    }
}
