namespace LMSupply.Embedder.Pooling;

/// <summary>
/// Factory for creating pooling strategies.
/// </summary>
internal static class PoolingFactory
{
    public static IPoolingStrategy Create(PoolingMode mode) => mode switch
    {
        PoolingMode.Mean => new MeanPoolingStrategy(),
        PoolingMode.Cls => new ClsPoolingStrategy(),
        PoolingMode.Max => new MaxPoolingStrategy(),
        _ => throw new ArgumentOutOfRangeException(nameof(mode), mode, "Unknown pooling mode")
    };
}
