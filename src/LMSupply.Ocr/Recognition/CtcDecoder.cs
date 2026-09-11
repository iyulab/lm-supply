namespace LMSupply.Ocr.Recognition;

/// <summary>
/// CTC (Connectionist Temporal Classification) decoder for text recognition.
/// </summary>
internal static class CtcDecoder
{
    /// <summary>How far a row's sum may stray from 1 and still count as a probability distribution.</summary>
    private const float DistributionTolerance = 1e-3f;

    /// <summary>
    /// Performs greedy CTC decoding on model output.
    /// </summary>
    /// <param name="logits">Model output logits of shape [T, V] where T is sequence length and V is vocabulary size.</param>
    /// <param name="dictionary">Character dictionary for index-to-char mapping.</param>
    /// <returns>Decoded text and average confidence score.</returns>
    public static (string text, float confidence) GreedyDecode(float[,] logits, CharacterDictionary dictionary)
    {
        var seqLength = logits.GetLength(0);
        var vocabSize = logits.GetLength(1);

        var indices = new List<int>(seqLength);
        var maxValues = new float[seqLength];
        var isDistribution = true;

        for (var t = 0; t < seqLength; t++)
        {
            // Find argmax for this timestep, and whether the row is already a probability distribution
            var maxIndex = 0;
            var maxValue = logits[t, 0];
            var rowSum = maxValue;
            var rowMin = maxValue;

            for (var v = 1; v < vocabSize; v++)
            {
                var value = logits[t, v];
                rowSum += value;
                rowMin = MathF.Min(rowMin, value);
                if (value > maxValue)
                {
                    maxValue = value;
                    maxIndex = v;
                }
            }

            indices.Add(maxIndex);
            maxValues[t] = maxValue;
            if (rowMin < 0f || MathF.Abs(rowSum - 1f) > DistributionTolerance)
            {
                isDistribution = false;
            }
        }

        // PaddleOCR recognizers end in a softmax, so their rows are already probabilities. Applying
        // softmax again flattens every score towards 1/vocabulary-size (~0.01 for a perfect read on a
        // 438-entry vocabulary). Decided once for the whole output: a model emits one or the other.
        var scores = new List<float>(seqLength);
        for (var t = 0; t < seqLength; t++)
        {
            scores.Add(isDistribution ? maxValues[t] : Softmax(logits, t, indices[t]));
        }

        // Decode using dictionary (handles blank removal and deduplication)
        var text = dictionary.Decode(indices);

        // Calculate average confidence (excluding blank tokens)
        var validScores = indices
            .Select((idx, i) => (idx, scores[i]))
            .Where(x => x.idx != dictionary.BlankIndex)
            .Select(x => x.Item2)
            .ToList();

        var confidence = validScores.Count > 0 ? validScores.Average() : 0f;

        return (text, confidence);
    }

    /// <summary>
    /// Performs greedy CTC decoding on model output from a 3D tensor.
    /// </summary>
    /// <param name="logits">Model output logits of shape [B, T, V] where B=1, T is sequence length, V is vocabulary size.</param>
    /// <param name="dictionary">Character dictionary for index-to-char mapping.</param>
    /// <returns>Decoded text and average confidence score.</returns>
    public static (string text, float confidence) GreedyDecode(float[,,] logits, CharacterDictionary dictionary)
    {
        var seqLength = logits.GetLength(1);
        var vocabSize = logits.GetLength(2);

        // Convert to 2D array (assuming batch size of 1)
        var logits2D = new float[seqLength, vocabSize];
        for (var t = 0; t < seqLength; t++)
        {
            for (var v = 0; v < vocabSize; v++)
            {
                logits2D[t, v] = logits[0, t, v];
            }
        }

        return GreedyDecode(logits2D, dictionary);
    }

    private static float Softmax(float[,] logits, int timestep, int index)
    {
        var vocabSize = logits.GetLength(1);

        // Find max for numerical stability
        var maxVal = float.MinValue;
        for (var v = 0; v < vocabSize; v++)
        {
            if (logits[timestep, v] > maxVal)
                maxVal = logits[timestep, v];
        }

        // Calculate exp sum
        var expSum = 0f;
        for (var v = 0; v < vocabSize; v++)
        {
            expSum += MathF.Exp(logits[timestep, v] - maxVal);
        }

        // Return softmax probability for the target index
        return MathF.Exp(logits[timestep, index] - maxVal) / expSum;
    }
}
