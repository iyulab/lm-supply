using LMSupply.Embedder.Utils;

namespace LMSupply.Embedder.Pooling;

/// <summary>
/// Mean pooling strategy that averages all non-padding token embeddings.
/// </summary>
internal sealed class MeanPoolingStrategy : IPoolingStrategy
{
    public void Pool(
        ReadOnlySpan<float> tokenEmbeddings,
        ReadOnlySpan<long> attentionMask,
        Span<float> result,
        int sequenceLength,
        int hiddenDim)
    {
        result.Clear();
        float maskSum = 0f;

        for (int seqPos = 0; seqPos < sequenceLength; seqPos++)
        {
            if (attentionMask[seqPos] == 1)
            {
                maskSum += 1f;
                var tokenSlice = tokenEmbeddings.Slice(seqPos * hiddenDim, hiddenDim);
                VectorOperations.Add(result, tokenSlice);
            }
        }

        float denominator = Math.Max(maskSum, 1e-9f);
        VectorOperations.Divide(result, denominator);
    }
}
