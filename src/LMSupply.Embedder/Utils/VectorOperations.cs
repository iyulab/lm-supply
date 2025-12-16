using System.Numerics.Tensors;

namespace LMSupply.Embedder.Utils;

/// <summary>
/// SIMD-optimized vector operations using TensorPrimitives.
/// </summary>
internal static class VectorOperations
{
    /// <summary>
    /// Computes cosine similarity between two vectors.
    /// </summary>
    /// <exception cref="ArgumentException">Thrown when vector lengths don't match or are empty.</exception>
    public static float CosineSimilarity(ReadOnlySpan<float> x, ReadOnlySpan<float> y)
    {
        ValidateVectorPair(x, y);
        return TensorPrimitives.CosineSimilarity(x, y);
    }

    /// <summary>
    /// Computes Euclidean distance between two vectors.
    /// </summary>
    /// <exception cref="ArgumentException">Thrown when vector lengths don't match or are empty.</exception>
    public static float EuclideanDistance(ReadOnlySpan<float> x, ReadOnlySpan<float> y)
    {
        ValidateVectorPair(x, y);
        return TensorPrimitives.Distance(x, y);
    }

    /// <summary>
    /// Computes dot product of two vectors.
    /// </summary>
    /// <exception cref="ArgumentException">Thrown when vector lengths don't match or are empty.</exception>
    public static float DotProduct(ReadOnlySpan<float> x, ReadOnlySpan<float> y)
    {
        ValidateVectorPair(x, y);
        return TensorPrimitives.Dot(x, y);
    }

    private static void ValidateVectorPair(ReadOnlySpan<float> x, ReadOnlySpan<float> y)
    {
        if (x.IsEmpty)
            throw new ArgumentException("Vector cannot be empty", nameof(x));
        if (y.IsEmpty)
            throw new ArgumentException("Vector cannot be empty", nameof(y));
        if (x.Length != y.Length)
            throw new ArgumentException($"Vector lengths must match. Got {x.Length} and {y.Length}");
    }

    /// <summary>
    /// Normalizes vector to unit length (L2 normalization) in-place.
    /// </summary>
    public static void NormalizeL2(Span<float> vector)
    {
        var norm = TensorPrimitives.Norm(vector);
        if (norm > 1e-12f)
        {
            TensorPrimitives.Divide(vector, norm, vector);
        }
    }

    /// <summary>
    /// Computes L2 norm of a vector.
    /// </summary>
    public static float Norm(ReadOnlySpan<float> vector)
    {
        return TensorPrimitives.Norm(vector);
    }

    /// <summary>
    /// Adds source vector to destination in-place.
    /// </summary>
    public static void Add(Span<float> destination, ReadOnlySpan<float> source)
    {
        TensorPrimitives.Add(destination, source, destination);
    }

    /// <summary>
    /// Divides all elements by scalar in-place.
    /// </summary>
    public static void Divide(Span<float> vector, float scalar)
    {
        TensorPrimitives.Divide(vector, scalar, vector);
    }
}
