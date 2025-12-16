namespace LMSupply.Text;

/// <summary>
/// Represents an encoded text sequence with attention mask.
/// </summary>
public readonly struct EncodedSequence
{
    /// <summary>
    /// Token IDs for the sequence.
    /// </summary>
    public long[] InputIds { get; }

    /// <summary>
    /// Attention mask (1 for real tokens, 0 for padding).
    /// </summary>
    public long[] AttentionMask { get; }

    /// <summary>
    /// Actual sequence length before padding.
    /// </summary>
    public int Length { get; }

    /// <summary>
    /// Creates an encoded sequence.
    /// </summary>
    public EncodedSequence(long[] inputIds, long[] attentionMask, int length)
    {
        InputIds = inputIds;
        AttentionMask = attentionMask;
        Length = length;
    }

    /// <summary>
    /// Creates an encoded sequence from int arrays (converts to long).
    /// </summary>
    public static EncodedSequence FromInts(int[] inputIds, int[] attentionMask, int length)
    {
        var longIds = new long[inputIds.Length];
        var longMask = new long[attentionMask.Length];

        for (int i = 0; i < inputIds.Length; i++)
        {
            longIds[i] = inputIds[i];
            longMask[i] = attentionMask[i];
        }

        return new EncodedSequence(longIds, longMask, length);
    }
}

/// <summary>
/// Represents an encoded text pair with token type IDs.
/// </summary>
public readonly struct EncodedPair
{
    /// <summary>
    /// Token IDs for the pair.
    /// </summary>
    public long[] InputIds { get; }

    /// <summary>
    /// Attention mask (1 for real tokens, 0 for padding).
    /// </summary>
    public long[] AttentionMask { get; }

    /// <summary>
    /// Token type IDs (0 for first text, 1 for second text).
    /// </summary>
    public long[] TokenTypeIds { get; }

    /// <summary>
    /// Actual sequence length before padding.
    /// </summary>
    public int Length { get; }

    /// <summary>
    /// Creates an encoded pair.
    /// </summary>
    public EncodedPair(long[] inputIds, long[] attentionMask, long[] tokenTypeIds, int length)
    {
        InputIds = inputIds;
        AttentionMask = attentionMask;
        TokenTypeIds = tokenTypeIds;
        Length = length;
    }
}

/// <summary>
/// Represents a batch of encoded sequences.
/// </summary>
public sealed class EncodedBatch
{
    /// <summary>
    /// Batched input IDs [batch_size, sequence_length].
    /// </summary>
    public long[,] InputIds { get; }

    /// <summary>
    /// Batched attention masks [batch_size, sequence_length].
    /// </summary>
    public long[,] AttentionMask { get; }

    /// <summary>
    /// Batch size.
    /// </summary>
    public int BatchSize { get; }

    /// <summary>
    /// Sequence length.
    /// </summary>
    public int SequenceLength { get; }

    /// <summary>
    /// Creates an empty batch.
    /// </summary>
    public EncodedBatch(int batchSize, int sequenceLength)
    {
        BatchSize = batchSize;
        SequenceLength = sequenceLength;
        InputIds = new long[batchSize, sequenceLength];
        AttentionMask = new long[batchSize, sequenceLength];
    }

    /// <summary>
    /// Sets a sequence at the specified index.
    /// </summary>
    public void SetSequence(int index, EncodedSequence sequence)
    {
        for (int i = 0; i < SequenceLength; i++)
        {
            InputIds[index, i] = sequence.InputIds[i];
            AttentionMask[index, i] = sequence.AttentionMask[i];
        }
    }

    /// <summary>
    /// Gets flattened input IDs for ONNX Runtime.
    /// </summary>
    public long[] GetFlatInputIds()
    {
        var flat = new long[BatchSize * SequenceLength];
        Buffer.BlockCopy(InputIds, 0, flat, 0, flat.Length * sizeof(long));
        return flat;
    }

    /// <summary>
    /// Gets flattened attention mask for ONNX Runtime.
    /// </summary>
    public long[] GetFlatAttentionMask()
    {
        var flat = new long[BatchSize * SequenceLength];
        Buffer.BlockCopy(AttentionMask, 0, flat, 0, flat.Length * sizeof(long));
        return flat;
    }

    /// <summary>
    /// Gets input IDs as jagged array for parallel processing.
    /// </summary>
    public long[][] GetInputIdsJagged()
    {
        var result = new long[BatchSize][];
        for (int i = 0; i < BatchSize; i++)
        {
            result[i] = new long[SequenceLength];
            for (int j = 0; j < SequenceLength; j++)
            {
                result[i][j] = InputIds[i, j];
            }
        }
        return result;
    }

    /// <summary>
    /// Gets attention masks as jagged array for parallel processing.
    /// </summary>
    public long[][] GetAttentionMasksJagged()
    {
        var result = new long[BatchSize][];
        for (int i = 0; i < BatchSize; i++)
        {
            result[i] = new long[SequenceLength];
            for (int j = 0; j < SequenceLength; j++)
            {
                result[i][j] = AttentionMask[i, j];
            }
        }
        return result;
    }
}

/// <summary>
/// Represents a batch of encoded pairs.
/// </summary>
public sealed class EncodedPairBatch
{
    /// <summary>
    /// Batched input IDs [batch_size, sequence_length].
    /// </summary>
    public long[,] InputIds { get; }

    /// <summary>
    /// Batched attention masks [batch_size, sequence_length].
    /// </summary>
    public long[,] AttentionMask { get; }

    /// <summary>
    /// Batched token type IDs [batch_size, sequence_length].
    /// </summary>
    public long[,] TokenTypeIds { get; }

    /// <summary>
    /// Batch size.
    /// </summary>
    public int BatchSize { get; }

    /// <summary>
    /// Sequence length.
    /// </summary>
    public int SequenceLength { get; }

    /// <summary>
    /// Creates an empty pair batch.
    /// </summary>
    public EncodedPairBatch(int batchSize, int sequenceLength)
    {
        BatchSize = batchSize;
        SequenceLength = sequenceLength;
        InputIds = new long[batchSize, sequenceLength];
        AttentionMask = new long[batchSize, sequenceLength];
        TokenTypeIds = new long[batchSize, sequenceLength];
    }

    /// <summary>
    /// Sets a pair at the specified index.
    /// </summary>
    public void SetPair(int index, EncodedPair pair)
    {
        for (int i = 0; i < SequenceLength; i++)
        {
            InputIds[index, i] = pair.InputIds[i];
            AttentionMask[index, i] = pair.AttentionMask[i];
            TokenTypeIds[index, i] = pair.TokenTypeIds[i];
        }
    }

    /// <summary>
    /// Gets flattened input IDs for ONNX Runtime.
    /// </summary>
    public long[] GetFlatInputIds()
    {
        var flat = new long[BatchSize * SequenceLength];
        Buffer.BlockCopy(InputIds, 0, flat, 0, flat.Length * sizeof(long));
        return flat;
    }

    /// <summary>
    /// Gets flattened attention mask for ONNX Runtime.
    /// </summary>
    public long[] GetFlatAttentionMask()
    {
        var flat = new long[BatchSize * SequenceLength];
        Buffer.BlockCopy(AttentionMask, 0, flat, 0, flat.Length * sizeof(long));
        return flat;
    }

    /// <summary>
    /// Gets flattened token type IDs for ONNX Runtime.
    /// </summary>
    public long[] GetFlatTokenTypeIds()
    {
        var flat = new long[BatchSize * SequenceLength];
        Buffer.BlockCopy(TokenTypeIds, 0, flat, 0, flat.Length * sizeof(long));
        return flat;
    }
}
