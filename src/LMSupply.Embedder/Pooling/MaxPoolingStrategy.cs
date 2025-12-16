namespace LMSupply.Embedder.Pooling;

/// <summary>
/// Max pooling strategy that takes the maximum value per dimension.
/// </summary>
internal sealed class MaxPoolingStrategy : IPoolingStrategy
{
    public void Pool(
        ReadOnlySpan<float> tokenEmbeddings,
        ReadOnlySpan<long> attentionMask,
        Span<float> result,
        int sequenceLength,
        int hiddenDim)
    {
        // Initialize with very negative values
        result.Fill(float.MinValue);

        for (int seqPos = 0; seqPos < sequenceLength; seqPos++)
        {
            if (attentionMask[seqPos] == 1)
            {
                var tokenSlice = tokenEmbeddings.Slice(seqPos * hiddenDim, hiddenDim);
                for (int dim = 0; dim < hiddenDim; dim++)
                {
                    if (tokenSlice[dim] > result[dim])
                    {
                        result[dim] = tokenSlice[dim];
                    }
                }
            }
        }

        // Handle edge case where no tokens were valid
        for (int dim = 0; dim < hiddenDim; dim++)
        {
            if (result[dim] == float.MinValue)
            {
                result[dim] = 0f;
            }
        }
    }
}
