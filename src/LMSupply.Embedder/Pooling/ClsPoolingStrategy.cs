namespace LMSupply.Embedder.Pooling;

/// <summary>
/// CLS pooling strategy that uses the first token embedding.
/// Required for BGE models.
/// </summary>
internal sealed class ClsPoolingStrategy : IPoolingStrategy
{
    public void Pool(
        ReadOnlySpan<float> tokenEmbeddings,
        ReadOnlySpan<long> attentionMask,
        Span<float> result,
        int sequenceLength,
        int hiddenDim)
    {
        // Copy first token (CLS) embedding
        tokenEmbeddings[..hiddenDim].CopyTo(result);
    }
}
