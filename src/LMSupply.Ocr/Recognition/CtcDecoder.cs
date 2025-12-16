namespace LMSupply.Ocr.Recognition;

/// <summary>
/// CTC (Connectionist Temporal Classification) decoder for text recognition.
/// </summary>
internal static class CtcDecoder
{
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

        var indices = new List<int>();
        var scores = new List<float>();

        for (var t = 0; t < seqLength; t++)
        {
            // Find argmax for this timestep
            var maxIndex = 0;
            var maxValue = logits[t, 0];

            for (var v = 1; v < vocabSize; v++)
            {
                if (logits[t, v] > maxValue)
                {
                    maxValue = logits[t, v];
                    maxIndex = v;
                }
            }

            indices.Add(maxIndex);

            // Convert logit to probability using softmax
            var prob = Softmax(logits, t, maxIndex);
            scores.Add(prob);
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
