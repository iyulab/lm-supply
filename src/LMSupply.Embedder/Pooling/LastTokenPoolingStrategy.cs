namespace LMSupply.Embedder.Pooling;

/// <summary>
/// Last-token pooling: the embedding of the last token the attention mask keeps. Decoder-based embedders (Qwen3-Embedding
/// and others converted with sentence-transformers' <c>pooling_mode_lasttoken</c>) are pooled this way.
/// </summary>
internal sealed class LastTokenPoolingStrategy : IPoolingStrategy
{
    public void Pool(
        ReadOnlySpan<float> tokenEmbeddings,
        ReadOnlySpan<long> attentionMask,
        Span<float> result,
        int sequenceLength,
        int hiddenDim)
    {
        // Padding is on the right, so the last kept token is the last position whose mask is set.
        var last = sequenceLength - 1;
        while (last > 0 && attentionMask[last] == 0)
            last--;

        tokenEmbeddings.Slice(last * hiddenDim, hiddenDim).CopyTo(result);
    }
}
