using System.Runtime.CompilerServices;

namespace LMSupply.Reranker.Core;

/// <summary>
/// Provides score normalization functions for model outputs.
/// </summary>
internal static class ScoreNormalizer
{
    /// <summary>
    /// Applies sigmoid activation to convert logits to probabilities.
    /// </summary>
    /// <param name="logit">Raw logit value.</param>
    /// <returns>Probability between 0 and 1.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Sigmoid(float logit)
    {
        // Handle extreme values to avoid overflow
        if (logit >= 20f) return 1f;
        if (logit <= -20f) return 0f;

        return 1f / (1f + MathF.Exp(-logit));
    }

    /// <summary>
    /// Applies sigmoid activation to an array of logits in place.
    /// </summary>
    /// <param name="logits">Array of logits to normalize.</param>
    public static void SigmoidInPlace(Span<float> logits)
    {
        for (var i = 0; i < logits.Length; i++)
        {
            logits[i] = Sigmoid(logits[i]);
        }
    }

    /// <summary>
    /// Applies sigmoid activation to an array of logits.
    /// </summary>
    /// <param name="logits">Array of logits.</param>
    /// <returns>Array of probabilities.</returns>
    public static float[] Sigmoid(ReadOnlySpan<float> logits)
    {
        var result = new float[logits.Length];
        for (var i = 0; i < logits.Length; i++)
        {
            result[i] = Sigmoid(logits[i]);
        }
        return result;
    }

    /// <summary>
    /// Applies softmax to convert binary classification logits to probabilities.
    /// Returns the probability of the positive class (index 1).
    /// </summary>
    /// <param name="logit0">Logit for negative class.</param>
    /// <param name="logit1">Logit for positive class.</param>
    /// <returns>Probability of positive class.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float SoftmaxPositive(float logit0, float logit1)
    {
        var maxLogit = MathF.Max(logit0, logit1);
        var exp0 = MathF.Exp(logit0 - maxLogit);
        var exp1 = MathF.Exp(logit1 - maxLogit);
        return exp1 / (exp0 + exp1);
    }

    /// <summary>
    /// Normalizes scores to a 0-1 range using min-max scaling.
    /// </summary>
    /// <param name="scores">Scores to normalize.</param>
    /// <returns>Normalized scores.</returns>
    public static float[] MinMaxNormalize(ReadOnlySpan<float> scores)
    {
        if (scores.Length == 0) return [];
        if (scores.Length == 1) return [1f];

        var min = float.MaxValue;
        var max = float.MinValue;

        foreach (var score in scores)
        {
            if (score < min) min = score;
            if (score > max) max = score;
        }

        var range = max - min;
        if (range < float.Epsilon)
        {
            var result = new float[scores.Length];
            Array.Fill(result, 0.5f);
            return result;
        }

        var normalized = new float[scores.Length];
        for (var i = 0; i < scores.Length; i++)
        {
            normalized[i] = (scores[i] - min) / range;
        }

        return normalized;
    }
}
