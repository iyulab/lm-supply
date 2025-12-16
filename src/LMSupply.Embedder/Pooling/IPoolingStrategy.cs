namespace LMSupply.Embedder.Pooling;

/// <summary>
/// Strategy interface for pooling token embeddings into sentence embeddings.
/// </summary>
internal interface IPoolingStrategy
{
    /// <summary>
    /// Pools token embeddings into a single sentence embedding.
    /// </summary>
    /// <param name="tokenEmbeddings">Token embeddings [sequenceLength * hiddenDim].</param>
    /// <param name="attentionMask">Attention mask [sequenceLength].</param>
    /// <param name="result">Output buffer for pooled embedding [hiddenDim].</param>
    /// <param name="sequenceLength">Number of tokens.</param>
    /// <param name="hiddenDim">Embedding dimension.</param>
    void Pool(
        ReadOnlySpan<float> tokenEmbeddings,
        ReadOnlySpan<long> attentionMask,
        Span<float> result,
        int sequenceLength,
        int hiddenDim);
}
