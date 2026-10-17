namespace LMSupply.Decider.Internal;

/// <summary>
/// The arithmetic both paths share: a probability per answer slot of a question becomes a <see cref="Decision"/>.
/// A question's slots are its options (then «none of these» last, when allowed), its levels, or yes and no.
/// </summary>
internal static class DecisionMath
{
    /// <summary>The number of answer slots of <paramref name="question"/>.</summary>
    public static int SlotCount(DecisionQuestion question) => question switch
    {
        ChoiceQuestion c => c.Options.Count + (c.AllowNone ? 1 : 0),
        ScoreQuestion s => s.Levels.Count,
        YesNoQuestion => 2,
        _ => throw new ArgumentOutOfRangeException(nameof(question)),
    };

    /// <summary>
    /// <paramref name="weights"/> scaled to sum to 1. All-zero weights stay zero (the caller decides what that means).
    /// </summary>
    public static double[] Normalize(IReadOnlyList<double> weights)
    {
        var sum = weights.Sum();
        return sum > 0 ? weights.Select(w => w / sum).ToArray() : weights.ToArray();
    }

    /// <summary>
    /// The distribution p ∝ p^(1/T), renormalized; computed in the log domain so a small p and a small T do not
    /// underflow to all zeros. A zero probability stays zero.
    /// </summary>
    public static double[] ApplyTemperature(IReadOnlyList<double> probabilities, double temperature)
    {
        var scaled = probabilities.Select(p => p > 0 ? Math.Log(p) / temperature : double.NegativeInfinity).ToArray();
        var max = scaled.Max();
        if (double.IsNegativeInfinity(max))
            return probabilities.ToArray();
        return Normalize(scaled.Select(s => Math.Exp(s - max)).ToArray());
    }

    /// <summary>
    /// The decision for <paramref name="question"/> from a probability per slot (already normalized), after the
    /// temperature for its kind when <paramref name="temperatures"/> has one.
    /// </summary>
    public static Decision Build(
        DecisionQuestion question,
        IReadOnlyList<double> slotProbabilities,
        DecisionPath path,
        double labelMass,
        IReadOnlyDictionary<DecisionKind, double>? temperatures)
    {
        if (slotProbabilities.Count != SlotCount(question))
            throw new ArgumentException("One probability per answer slot is required.", nameof(slotProbabilities));

        var calibrated = false;
        var p = slotProbabilities.ToArray();
        if (temperatures is not null && temperatures.TryGetValue(question.Kind, out var temperature))
        {
            p = ApplyTemperature(p, temperature);
            calibrated = true;
        }

        var confidence = p.Max();
        var best = Array.IndexOf(p, confidence);

        switch (question)
        {
            case ChoiceQuestion c:
            {
                var byOption = new Dictionary<string, double>(StringComparer.Ordinal);
                for (var i = 0; i < c.Options.Count; i++)
                    byOption[c.Options[i]] = p[i];
                var none = c.AllowNone ? p[c.Options.Count] : 0;
                var choice = best < c.Options.Count ? c.Options[best] : null;
                return new Decision(DecisionKind.Choice, path, byOption, none, confidence, labelMass, calibrated, choice: choice);
            }

            case ScoreQuestion s:
            {
                var byLevel = new Dictionary<string, double>(StringComparer.Ordinal);
                var expected = 0.0;
                for (var i = 0; i < s.Levels.Count; i++)
                {
                    byLevel[s.Levels[i]] = p[i];
                    expected += i * p[i];
                }
                return new Decision(DecisionKind.Score, path, byLevel, 0, confidence, labelMass, calibrated, score: expected, level: best);
            }

            case YesNoQuestion:
            {
                var byAnswer = new Dictionary<string, double>(StringComparer.Ordinal) { ["yes"] = p[0], ["no"] = p[1] };
                return new Decision(DecisionKind.YesNo, path, byAnswer, 0, confidence, labelMass, calibrated, probabilityTrue: p[0]);
            }

            default:
                throw new ArgumentOutOfRangeException(nameof(question));
        }
    }
}
