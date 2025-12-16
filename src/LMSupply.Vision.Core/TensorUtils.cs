using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace LMSupply.Vision;

/// <summary>
/// Utility methods for working with tensors and ONNX Runtime.
/// </summary>
public static class TensorUtils
{
    /// <summary>
    /// Creates a DenseTensor from a preprocessed float array in NCHW format.
    /// </summary>
    /// <param name="data">Preprocessed image data.</param>
    /// <param name="width">Image width.</param>
    /// <param name="height">Image height.</param>
    /// <param name="channels">Number of channels (default 3 for RGB).</param>
    /// <param name="batchSize">Batch size (default 1).</param>
    /// <returns>DenseTensor in NCHW format [batch, channels, height, width].</returns>
    public static DenseTensor<float> CreateImageTensor(
        float[] data,
        int width,
        int height,
        int channels = 3,
        int batchSize = 1)
    {
        ArgumentNullException.ThrowIfNull(data);

        var expectedLength = batchSize * channels * height * width;
        if (data.Length != expectedLength)
        {
            throw new ArgumentException(
                $"Data length {data.Length} does not match expected length {expectedLength} " +
                $"for dimensions [{batchSize}, {channels}, {height}, {width}]",
                nameof(data));
        }

        return new DenseTensor<float>(data, [batchSize, channels, height, width]);
    }

    /// <summary>
    /// Creates a DenseTensor from a preprocessed float array using a PreprocessProfile.
    /// </summary>
    /// <param name="data">Preprocessed image data.</param>
    /// <param name="profile">Preprocessing profile containing dimensions.</param>
    /// <param name="batchSize">Batch size (default 1).</param>
    /// <returns>DenseTensor in NCHW format [batch, channels, height, width].</returns>
    public static DenseTensor<float> CreateImageTensor(
        float[] data,
        PreprocessProfile profile,
        int batchSize = 1)
    {
        ArgumentNullException.ThrowIfNull(profile);
        return CreateImageTensor(data, profile.Width, profile.Height, channels: 3, batchSize);
    }

    /// <summary>
    /// Creates a NamedOnnxValue for image input tensor.
    /// </summary>
    /// <param name="name">Input tensor name (e.g., "pixel_values").</param>
    /// <param name="data">Preprocessed image data.</param>
    /// <param name="profile">Preprocessing profile containing dimensions.</param>
    /// <returns>NamedOnnxValue ready for inference.</returns>
    public static NamedOnnxValue CreateImageInput(
        string name,
        float[] data,
        PreprocessProfile profile)
    {
        var tensor = CreateImageTensor(data, profile);
        return NamedOnnxValue.CreateFromTensor(name, tensor);
    }

    /// <summary>
    /// Creates a DenseTensor of token IDs for text input.
    /// </summary>
    /// <param name="tokenIds">Array of token IDs.</param>
    /// <param name="batchSize">Batch size (default 1).</param>
    /// <returns>DenseTensor in shape [batch, sequence_length].</returns>
    public static DenseTensor<long> CreateTokenTensor(int[] tokenIds, int batchSize = 1)
    {
        ArgumentNullException.ThrowIfNull(tokenIds);

        var longIds = new long[tokenIds.Length];
        for (int i = 0; i < tokenIds.Length; i++)
        {
            longIds[i] = tokenIds[i];
        }

        return new DenseTensor<long>(longIds, [batchSize, tokenIds.Length]);
    }

    /// <summary>
    /// Creates a DenseTensor of token IDs from long array.
    /// </summary>
    /// <param name="tokenIds">Array of token IDs.</param>
    /// <param name="batchSize">Batch size (default 1).</param>
    /// <returns>DenseTensor in shape [batch, sequence_length].</returns>
    public static DenseTensor<long> CreateTokenTensor(long[] tokenIds, int batchSize = 1)
    {
        ArgumentNullException.ThrowIfNull(tokenIds);
        return new DenseTensor<long>(tokenIds, [batchSize, tokenIds.Length]);
    }

    /// <summary>
    /// Creates attention mask tensor (all 1s for valid tokens).
    /// </summary>
    /// <param name="sequenceLength">Length of the sequence.</param>
    /// <param name="batchSize">Batch size (default 1).</param>
    /// <returns>DenseTensor of attention mask.</returns>
    public static DenseTensor<long> CreateAttentionMask(int sequenceLength, int batchSize = 1)
    {
        var mask = new long[batchSize * sequenceLength];
        Array.Fill(mask, 1L);
        return new DenseTensor<long>(mask, [batchSize, sequenceLength]);
    }

    /// <summary>
    /// Extracts logits from decoder output and returns the argmax token ID.
    /// </summary>
    /// <param name="logits">Logits tensor from decoder.</param>
    /// <returns>Token ID with highest probability.</returns>
    public static int ArgMax(ReadOnlySpan<float> logits)
    {
        if (logits.IsEmpty)
            throw new ArgumentException("Logits cannot be empty", nameof(logits));

        int maxIndex = 0;
        float maxValue = logits[0];

        for (int i = 1; i < logits.Length; i++)
        {
            if (logits[i] > maxValue)
            {
                maxValue = logits[i];
                maxIndex = i;
            }
        }

        return maxIndex;
    }

    /// <summary>
    /// Applies softmax to logits and returns probabilities.
    /// </summary>
    /// <param name="logits">Input logits.</param>
    /// <param name="temperature">Temperature for scaling (1.0 = no scaling).</param>
    /// <returns>Probability distribution.</returns>
    public static float[] Softmax(ReadOnlySpan<float> logits, float temperature = 1.0f)
    {
        if (logits.IsEmpty)
            throw new ArgumentException("Logits cannot be empty", nameof(logits));

        var result = new float[logits.Length];

        // Find max for numerical stability
        float maxLogit = float.MinValue;
        foreach (var logit in logits)
        {
            if (logit > maxLogit)
                maxLogit = logit;
        }

        // Compute exp(logit / temperature - max) and sum
        float sum = 0f;
        for (int i = 0; i < logits.Length; i++)
        {
            result[i] = MathF.Exp((logits[i] / temperature) - (maxLogit / temperature));
            sum += result[i];
        }

        // Normalize
        for (int i = 0; i < result.Length; i++)
        {
            result[i] /= sum;
        }

        return result;
    }

    /// <summary>
    /// Samples a token ID from a probability distribution.
    /// </summary>
    /// <param name="probabilities">Probability distribution.</param>
    /// <param name="random">Random number generator (optional).</param>
    /// <returns>Sampled token ID.</returns>
    public static int SampleFromDistribution(float[] probabilities, Random? random = null)
    {
        ArgumentNullException.ThrowIfNull(probabilities);

        random ??= Random.Shared;
        float sample = (float)random.NextDouble();
        float cumulative = 0f;

        for (int i = 0; i < probabilities.Length; i++)
        {
            cumulative += probabilities[i];
            if (sample < cumulative)
            {
                return i;
            }
        }

        return probabilities.Length - 1;
    }
}
