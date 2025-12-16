using System.Buffers;
using Microsoft.ML.OnnxRuntime;

namespace LMSupply.Inference;

/// <summary>
/// Extension methods and utilities for efficient OrtValue-based inference.
/// OrtValue API provides better memory management and zero-copy semantics.
/// </summary>
public static class OrtValueExtensions
{
    /// <summary>
    /// Creates an OrtValue tensor from a float array with specified shape.
    /// Uses the underlying array directly without copying (zero-copy).
    /// </summary>
    /// <param name="data">The float data.</param>
    /// <param name="shape">The tensor shape.</param>
    /// <returns>An OrtValue that wraps the data.</returns>
    public static OrtValue CreateTensorFromArray(float[] data, long[] shape)
    {
        ArgumentNullException.ThrowIfNull(data);
        ArgumentNullException.ThrowIfNull(shape);
        return OrtValue.CreateTensorValueFromMemory<float>(data, shape);
    }

    /// <summary>
    /// Creates an OrtValue tensor from a Memory&lt;float&gt; with specified shape.
    /// </summary>
    public static OrtValue CreateTensorFromMemory(Memory<float> data, long[] shape)
    {
        return OrtValue.CreateTensorValueFromMemory<float>(OrtMemoryInfo.DefaultInstance, data, shape);
    }

    /// <summary>
    /// Creates an OrtValue tensor from a long array with specified shape.
    /// Used for token IDs and attention masks.
    /// </summary>
    public static OrtValue CreateTensorFromArray(long[] data, long[] shape)
    {
        ArgumentNullException.ThrowIfNull(data);
        ArgumentNullException.ThrowIfNull(shape);
        return OrtValue.CreateTensorValueFromMemory<long>(data, shape);
    }

    /// <summary>
    /// Creates an OrtValue tensor from a Memory&lt;long&gt; with specified shape.
    /// </summary>
    public static OrtValue CreateTensorFromMemory(Memory<long> data, long[] shape)
    {
        return OrtValue.CreateTensorValueFromMemory<long>(OrtMemoryInfo.DefaultInstance, data, shape);
    }

    /// <summary>
    /// Creates an OrtValue tensor from an int array, converting to long.
    /// </summary>
    public static OrtValue CreateTensorFromArray(int[] data, long[] shape)
    {
        ArgumentNullException.ThrowIfNull(data);
        var longData = new long[data.Length];
        for (int i = 0; i < data.Length; i++)
        {
            longData[i] = data[i];
        }
        return OrtValue.CreateTensorValueFromMemory<long>(longData, shape);
    }

    /// <summary>
    /// Extracts float array from an OrtValue output tensor.
    /// </summary>
    public static float[] ToFloatArray(this OrtValue ortValue)
    {
        return ortValue.GetTensorDataAsSpan<float>().ToArray();
    }

    /// <summary>
    /// Extracts long array from an OrtValue output tensor.
    /// </summary>
    public static long[] ToLongArray(this OrtValue ortValue)
    {
        return ortValue.GetTensorDataAsSpan<long>().ToArray();
    }

    /// <summary>
    /// Gets the tensor shape from an OrtValue.
    /// </summary>
    public static long[] GetShape(this OrtValue ortValue)
    {
        return ortValue.GetTensorTypeAndShape().Shape;
    }

    /// <summary>
    /// Creates a standard image input tensor in NCHW format.
    /// </summary>
    /// <param name="imageData">Preprocessed image data in NCHW order.</param>
    /// <param name="batchSize">Batch size (default 1).</param>
    /// <param name="channels">Number of channels (default 3).</param>
    /// <param name="height">Image height.</param>
    /// <param name="width">Image width.</param>
    /// <returns>OrtValue tensor ready for inference.</returns>
    public static OrtValue CreateImageTensor(
        float[] imageData,
        int batchSize,
        int channels,
        int height,
        int width)
    {
        return CreateTensorFromArray(imageData, [batchSize, channels, height, width]);
    }

    /// <summary>
    /// Creates token input tensors for transformer models.
    /// </summary>
    /// <param name="tokenIds">Token IDs.</param>
    /// <param name="batchSize">Batch size (default 1).</param>
    /// <returns>OrtValue tensor for input_ids.</returns>
    public static OrtValue CreateTokenTensor(long[] tokenIds, int batchSize = 1)
    {
        int seqLen = tokenIds.Length / batchSize;
        return CreateTensorFromArray(tokenIds, [batchSize, seqLen]);
    }

    /// <summary>
    /// Creates an attention mask tensor (all ones).
    /// </summary>
    /// <param name="sequenceLength">Sequence length.</param>
    /// <param name="batchSize">Batch size (default 1).</param>
    /// <returns>OrtValue tensor for attention_mask.</returns>
    public static OrtValue CreateAttentionMask(int sequenceLength, int batchSize = 1)
    {
        var mask = new long[batchSize * sequenceLength];
        Array.Fill(mask, 1L);
        return CreateTensorFromArray(mask, [batchSize, sequenceLength]);
    }

    /// <summary>
    /// Creates a position IDs tensor.
    /// </summary>
    /// <param name="sequenceLength">Sequence length.</param>
    /// <param name="batchSize">Batch size (default 1).</param>
    /// <returns>OrtValue tensor for position_ids.</returns>
    public static OrtValue CreatePositionIds(int sequenceLength, int batchSize = 1)
    {
        var posIds = new long[batchSize * sequenceLength];
        for (int b = 0; b < batchSize; b++)
        {
            for (int i = 0; i < sequenceLength; i++)
            {
                posIds[b * sequenceLength + i] = i;
            }
        }
        return CreateTensorFromArray(posIds, [batchSize, sequenceLength]);
    }
}

/// <summary>
/// A reusable buffer pool for OrtValue-based inference to reduce allocations.
/// </summary>
public sealed class InferenceBufferPool : IDisposable
{
    private readonly ArrayPool<float> _floatPool = ArrayPool<float>.Shared;
    private readonly ArrayPool<long> _longPool = ArrayPool<long>.Shared;
    private readonly List<float[]> _rentedFloatArrays = [];
    private readonly List<long[]> _rentedLongArrays = [];
    private bool _disposed;

    /// <summary>
    /// Rents a float array from the pool.
    /// </summary>
    public float[] RentFloat(int minimumLength)
    {
        var array = _floatPool.Rent(minimumLength);
        _rentedFloatArrays.Add(array);
        return array;
    }

    /// <summary>
    /// Rents a long array from the pool.
    /// </summary>
    public long[] RentLong(int minimumLength)
    {
        var array = _longPool.Rent(minimumLength);
        _rentedLongArrays.Add(array);
        return array;
    }

    /// <summary>
    /// Creates an OrtValue image tensor using pooled memory.
    /// </summary>
    public OrtValue CreateImageTensor(int batchSize, int channels, int height, int width)
    {
        int size = batchSize * channels * height * width;
        var data = RentFloat(size);
        return OrtValue.CreateTensorValueFromMemory<float>(
            OrtMemoryInfo.DefaultInstance,
            data.AsMemory(0, size),
            [batchSize, channels, height, width]);
    }

    /// <summary>
    /// Creates an OrtValue token tensor using pooled memory.
    /// </summary>
    public OrtValue CreateTokenTensor(int batchSize, int sequenceLength)
    {
        int size = batchSize * sequenceLength;
        var data = RentLong(size);
        return OrtValue.CreateTensorValueFromMemory<long>(
            OrtMemoryInfo.DefaultInstance,
            data.AsMemory(0, size),
            [batchSize, sequenceLength]);
    }

    /// <summary>
    /// Returns all rented arrays to the pool.
    /// </summary>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        foreach (var array in _rentedFloatArrays)
        {
            _floatPool.Return(array);
        }
        _rentedFloatArrays.Clear();

        foreach (var array in _rentedLongArrays)
        {
            _longPool.Return(array);
        }
        _rentedLongArrays.Clear();
    }
}

/// <summary>
/// Named OrtValue input for inference sessions.
/// </summary>
public readonly struct OrtValueInput : IDisposable
{
    /// <summary>
    /// The input name.
    /// </summary>
    public string Name { get; }

    /// <summary>
    /// The OrtValue.
    /// </summary>
    public OrtValue Value { get; }

    /// <summary>
    /// Creates a new OrtValueInput.
    /// </summary>
    public OrtValueInput(string name, OrtValue value)
    {
        Name = name;
        Value = value;
    }

    /// <summary>
    /// Disposes the OrtValue.
    /// </summary>
    public void Dispose()
    {
        Value?.Dispose();
    }

    /// <summary>
    /// Creates a float tensor input.
    /// </summary>
    public static OrtValueInput CreateFloat(string name, float[] data, long[] shape)
    {
        return new OrtValueInput(name, OrtValueExtensions.CreateTensorFromArray(data, shape));
    }

    /// <summary>
    /// Creates a long tensor input.
    /// </summary>
    public static OrtValueInput CreateLong(string name, long[] data, long[] shape)
    {
        return new OrtValueInput(name, OrtValueExtensions.CreateTensorFromArray(data, shape));
    }

    /// <summary>
    /// Creates an image tensor input in NCHW format.
    /// </summary>
    public static OrtValueInput CreateImage(string name, float[] data, int batch, int channels, int height, int width)
    {
        return new OrtValueInput(name, OrtValueExtensions.CreateImageTensor(data, batch, channels, height, width));
    }

    /// <summary>
    /// Creates a token IDs input.
    /// </summary>
    public static OrtValueInput CreateTokenIds(string name, long[] tokenIds, int batchSize = 1)
    {
        return new OrtValueInput(name, OrtValueExtensions.CreateTokenTensor(tokenIds, batchSize));
    }

    /// <summary>
    /// Creates an attention mask input.
    /// </summary>
    public static OrtValueInput CreateAttentionMask(string name, int seqLength, int batchSize = 1)
    {
        return new OrtValueInput(name, OrtValueExtensions.CreateAttentionMask(seqLength, batchSize));
    }
}
