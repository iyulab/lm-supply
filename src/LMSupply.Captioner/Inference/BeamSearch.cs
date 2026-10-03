namespace LMSupply.Captioner.Inference;

/// <summary>
/// Beam search over a decoder step, shared by every captioner: each step extends the live beams by their most likely
/// next tokens, keeps the best <c>beams</c> by summed log-probability, and retires a beam when it emits the end
/// token. Finished hypotheses are ranked by mean log-probability per token (length penalty 1.0, as in Hugging Face
/// generation), and the search stops once <c>beams</c> of them are finished (<c>early_stopping</c>) or the step budget
/// is spent.
/// </summary>
/// <typeparam name="TState">What a step needs besides the sequence — nothing for a decoder that re-reads the whole
/// sequence, the key/value cache for one that does not. A step returns the state of the extended sequence; states are
/// never mutated, so beams that share a parent can share its state.</typeparam>
internal static class BeamSearch<TState>
{
    /// <summary>A finished hypothesis: the generated tokens (end token excluded) and its mean log-probability.</summary>
    internal sealed record Hypothesis(int[] Tokens, float MeanLogProb);

    private sealed record Beam(List<int> Sequence, int Generated, float LogProb, TState State);

    /// <param name="prefix">Tokens already in the sequence (decoder start, forced tokens); not part of the result.</param>
    /// <param name="initialState">The state after <paramref name="prefix"/>.</param>
    /// <param name="beams">Beam width (≥ 2).</param>
    /// <param name="maxNewTokens">Most tokens a hypothesis may generate.</param>
    /// <param name="endToken">The token that finishes a hypothesis.</param>
    /// <param name="step">Logits for the token after a sequence, and the state that includes the sequence's last token.</param>
    /// <param name="adjust">Optional in-place logits rule applied per beam (for example no-repeat n-grams).</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Finished hypotheses, best first (at least one).</returns>
    public static async Task<IReadOnlyList<Hypothesis>> RunAsync(
        IReadOnlyList<int> prefix,
        TState initialState,
        int beams,
        int maxNewTokens,
        int endToken,
        Func<IReadOnlyList<int>, TState, Task<(float[] Logits, TState State)>> step,
        Action<float[], IReadOnlyList<int>>? adjust,
        CancellationToken cancellationToken)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(beams, 2);

        var live = new List<Beam> { new([.. prefix], 0, 0f, initialState) };
        var finished = new List<Hypothesis>();

        for (var t = 0; t < maxNewTokens && live.Count > 0 && finished.Count < beams; t++)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var candidates = new List<(Beam Parent, int Token, float LogProb, TState State)>();
            foreach (var beam in live)
            {
                var (logits, state) = await step(beam.Sequence, beam.State).ConfigureAwait(false);
                adjust?.Invoke(logits, beam.Sequence);
                var logProbs = LogSoftmax(logits);

                foreach (var token in TopK(logProbs, beams * 2))
                    candidates.Add((beam, token, beam.LogProb + logProbs[token], state));
            }

            var next = new List<Beam>();
            foreach (var c in candidates.OrderByDescending(c => c.LogProb))
            {
                if (c.Token == endToken)
                {
                    // A hypothesis that ends here is ranked by the tokens it generated before the end token.
                    var length = Math.Max(1, c.Parent.Generated);
                    finished.Add(new Hypothesis(c.Parent.Sequence.Skip(prefix.Count).ToArray(), c.LogProb / length));
                    if (finished.Count >= beams)
                        break;
                    continue;
                }

                next.Add(new Beam([.. c.Parent.Sequence, c.Token], c.Parent.Generated + 1, c.LogProb, c.State));
                if (next.Count == beams)
                    break;
            }

            live = next;
        }

        // Budget spent: the beams still open count as finished, so there is always an answer.
        if (finished.Count == 0)
        {
            finished.AddRange(live.Select(b =>
                new Hypothesis(b.Sequence.Skip(prefix.Count).ToArray(), b.LogProb / Math.Max(1, b.Generated))));
        }

        return finished.OrderByDescending(h => h.MeanLogProb).ToList();
    }

    private static float[] LogSoftmax(float[] logits)
    {
        var max = float.NegativeInfinity;
        foreach (var v in logits)
            if (v > max) max = v;

        double sum = 0;
        foreach (var v in logits)
            sum += Math.Exp(v - max);

        var logSum = max + (float)Math.Log(sum);
        var result = new float[logits.Length];
        for (var i = 0; i < logits.Length; i++)
            result[i] = logits[i] - logSum;
        return result;
    }

    private static IEnumerable<int> TopK(float[] values, int k)
        => Enumerable.Range(0, values.Length)
            .Where(i => !float.IsNegativeInfinity(values[i]))
            .OrderByDescending(i => values[i])
            .Take(k);
}
