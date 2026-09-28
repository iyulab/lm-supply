using System.Buffers.Binary;
using System.Diagnostics;
using System.Text;

namespace LMSupply.Generator.Internal.Llama;

/// <summary>
/// Reads metadata from GGUF model files.
/// </summary>
/// <remarks>
/// GGUF format specification: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
/// </remarks>
public static class GgufMetadataReader
{
    private const uint GgufMagic = 0x46554747; // "GGUF" in little-endian

    /// <summary>
    /// Reads metadata from a GGUF file.
    /// </summary>
    /// <param name="filePath">Path to the GGUF file.</param>
    /// <param name="includeRawMetadata">Whether to include all raw metadata key-values.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Parsed metadata, or null if the file is not a valid GGUF.</returns>
    public static async Task<GgufMetadata?> ReadAsync(
        string filePath,
        bool includeRawMetadata = false,
        CancellationToken cancellationToken = default)
    {
        if (!File.Exists(filePath))
            return null;

        await using var stream = new FileStream(
            filePath,
            FileMode.Open,
            FileAccess.Read,
            FileShare.Read,
            bufferSize: 64 * 1024,
            useAsync: true);

        return await ReadFromStreamAsync(stream, includeRawMetadata, cancellationToken);
    }

    /// <summary>
    /// Reads metadata from a GGUF stream.
    /// </summary>
    public static async Task<GgufMetadata?> ReadFromStreamAsync(
        Stream stream,
        bool includeRawMetadata = false,
        CancellationToken cancellationToken = default)
    {
        var reader = new GgufReader(stream);

        // Read and verify magic number
        var magic = await reader.ReadUInt32Async(cancellationToken);
        if (magic != GgufMagic)
            return null;

        // Read version
        var version = await reader.ReadUInt32Async(cancellationToken);
        if (version < 2 || version > 3)
            return null; // Only support GGUF v2 and v3

        // Read counts
        var tensorCount = await reader.ReadUInt64Async(cancellationToken);
        var metadataKvCount = await reader.ReadUInt64Async(cancellationToken);

        // Read metadata key-value pairs
        var rawMetadata = includeRawMetadata ? new Dictionary<string, object?>() : null;
        string? architecture = null;
        string? name = null;
        int? layerCount = null;
        int? contextLength = null;
        int? embeddingLength = null;
        int? headCount = null;
        int? headCountKv = null;
        int? vocabSize = null;
        int? ffnLength = null;
        float? ropeFreqBase = null;
        string? quantizationType = null;
        int? fileType = null;
        int? expertCount = null;
        int? expertUsedCount = null;
        int[]? headCountKvPerLayer = null;
        int? keyLength = null;
        int? valueLength = null;
        int? keyLengthSwa = null;
        int? valueLengthSwa = null;
        int? slidingWindow = null;
        bool[]? slidingWindowPattern = null;
        int? sharedKvLayers = null;
        int? fullAttentionInterval = null;

        for (ulong i = 0; i < metadataKvCount; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var key = await reader.ReadStringAsync(cancellationToken);
            var valueType = await reader.ReadUInt32Async(cancellationToken);
            var value = await reader.ReadValueAsync(valueType, cancellationToken);

            rawMetadata?.TryAdd(key, value);

            // Extract known metadata fields
            switch (key.ToLowerInvariant())
            {
                case "general.architecture":
                    architecture = value?.ToString();
                    break;
                case "general.name":
                    name = value?.ToString();
                    break;
                case "general.file_type":
                    fileType = value is int ft ? ft : (value is uint uft ? (int)uft : null);
                    break;
                // Note: general.quantization_version is a version number, not quantization type
                // Quantization type is inferred from general.file_type
                default:
                    // Architecture-specific keys
                    if (key.EndsWith(".block_count", StringComparison.Ordinal))
                        layerCount = ConvertToInt(value);
                    else if (key.EndsWith(".context_length", StringComparison.Ordinal))
                        contextLength = ConvertToInt(value);
                    else if (key.EndsWith(".embedding_length", StringComparison.Ordinal))
                        embeddingLength = ConvertToInt(value);
                    else if (key.EndsWith(".attention.head_count", StringComparison.Ordinal))
                        headCount = ConvertToInt(value);
                    else if (key.EndsWith(".attention.head_count_kv", StringComparison.Ordinal))
                    {
                        // A scalar for most models; a per-layer array where layers differ (0 = no KV).
                        if (value is object?[] perLayer)
                        {
                            headCountKvPerLayer = perLayer.Select(v => ConvertToInt(v) ?? 0).ToArray();
                            headCountKv = headCountKvPerLayer.Length > 0 ? headCountKvPerLayer.Max() : null;
                        }
                        else
                        {
                            headCountKv = ConvertToInt(value);
                        }
                    }
                    else if (key.EndsWith(".attention.key_length", StringComparison.Ordinal))
                        keyLength = ConvertToInt(value);
                    else if (key.EndsWith(".attention.value_length", StringComparison.Ordinal))
                        valueLength = ConvertToInt(value);
                    else if (key.EndsWith(".attention.key_length_swa", StringComparison.Ordinal))
                        keyLengthSwa = ConvertToInt(value);
                    else if (key.EndsWith(".attention.value_length_swa", StringComparison.Ordinal))
                        valueLengthSwa = ConvertToInt(value);
                    else if (key.EndsWith(".attention.sliding_window", StringComparison.Ordinal))
                        slidingWindow = ConvertToInt(value);
                    else if (key.EndsWith(".attention.sliding_window_pattern", StringComparison.Ordinal))
                        slidingWindowPattern = value is object?[] pattern ? pattern.Select(v => v is true).ToArray() : null;
                    else if (key.EndsWith(".attention.shared_kv_layers", StringComparison.Ordinal))
                        sharedKvLayers = ConvertToInt(value);
                    else if (key.EndsWith(".full_attention_interval", StringComparison.Ordinal))
                        fullAttentionInterval = ConvertToInt(value);
                    else if (key.EndsWith(".feed_forward_length", StringComparison.Ordinal))
                        ffnLength = ConvertToInt(value);
                    else if (key.EndsWith(".rope.freq_base", StringComparison.Ordinal))
                        ropeFreqBase = ConvertToFloat(value);
                    else if (key.EndsWith(".vocab_size", StringComparison.Ordinal) || key == "tokenizer.ggml.tokens")
                        vocabSize ??= value is string[] arr ? arr.Length : ConvertToInt(value);
                    else if (key.EndsWith(".expert_count", StringComparison.Ordinal))
                        expertCount = ConvertToInt(value);
                    else if (key.EndsWith(".expert_used_count", StringComparison.Ordinal))
                        expertUsedCount = ConvertToInt(value);
                    break;
            }
        }

        // Infer quantization type from file type if not explicitly set
        if (string.IsNullOrEmpty(quantizationType) && fileType.HasValue)
        {
            quantizationType = InferQuantizationFromFileType(fileType.Value);
        }

        return new GgufMetadata
        {
            Version = version,
            Architecture = architecture,
            Name = name,
            LayerCount = layerCount,
            ContextLength = contextLength,
            EmbeddingLength = embeddingLength,
            HeadCount = headCount,
            HeadCountKv = headCountKv,
            VocabSize = vocabSize,
            FeedForwardLength = ffnLength,
            RopeFreqBase = ropeFreqBase,
            QuantizationType = quantizationType,
            FileType = fileType,
            TensorCount = (long)tensorCount,
            ExpertCount = expertCount,
            ExpertUsedCount = expertUsedCount,
            HeadCountKvPerLayer = headCountKvPerLayer,
            KeyLength = keyLength,
            ValueLength = valueLength,
            KeyLengthSwa = keyLengthSwa,
            ValueLengthSwa = valueLengthSwa,
            SlidingWindow = slidingWindow,
            SlidingWindowPattern = slidingWindowPattern,
            SharedKvLayers = sharedKvLayers,
            FullAttentionInterval = fullAttentionInterval,
            RawMetadata = rawMetadata
        };
    }

    /// <summary>
    /// Quickly checks if a file is a valid GGUF file.
    /// </summary>
    public static async Task<bool> IsGgufFileAsync(string filePath, CancellationToken cancellationToken = default)
    {
        if (!File.Exists(filePath))
            return false;

        try
        {
            await using var stream = new FileStream(
                filePath,
                FileMode.Open,
                FileAccess.Read,
                FileShare.Read,
                bufferSize: 8);

            var buffer = new byte[4];
            var bytesRead = await stream.ReadAsync(buffer, cancellationToken);

            if (bytesRead < 4)
                return false;

            var magic = BinaryPrimitives.ReadUInt32LittleEndian(buffer);
            return magic == GgufMagic;
        }
        catch (Exception ex)
        {
            Trace.TraceInformation($"[GgufMetadataReader] GGUF magic check failed: {ex.Message}");
            return false;
        }
    }

    private static int? ConvertToInt(object? value)
    {
        return value switch
        {
            int i => i,
            uint u => (int)u,
            long l => (int)l,
            ulong ul => (int)ul,
            float f => (int)f,
            double d => (int)d,
            _ => null
        };
    }

    private static float? ConvertToFloat(object? value)
    {
        return value switch
        {
            float f => f,
            double d => (float)d,
            int i => i,
            long l => l,
            _ => null
        };
    }

    private static string InferQuantizationFromFileType(int fileType)
    {
        // GGUF file type enum values from llama.cpp
        return fileType switch
        {
            0 => "F32",
            1 => "F16",
            2 => "Q4_0",
            3 => "Q4_1",
            6 => "Q5_0",
            7 => "Q5_1",
            8 => "Q8_0",
            9 => "Q8_1",
            10 => "Q2_K",
            11 => "Q3_K_S",
            12 => "Q3_K_M",
            13 => "Q3_K_L",
            14 => "Q4_K_S",
            15 => "Q4_K_M",
            16 => "Q5_K_S",
            17 => "Q5_K_M",
            18 => "Q6_K",
            19 => "IQ2_XXS",
            20 => "IQ2_XS",
            21 => "IQ3_XXS",
            22 => "IQ1_S",
            23 => "IQ4_NL",
            24 => "IQ3_S",
            25 => "IQ2_S",
            26 => "IQ4_XS",
            27 => "IQ1_M",
            28 => "BF16",
            _ => $"UNKNOWN_{fileType}"
        };
    }

    /// <summary>
    /// Binary reader for GGUF format.
    /// </summary>
    private sealed class GgufReader
    {
        private readonly Stream _stream;
        private const ulong MaxReadNumericArrayLength = 4096;
        private readonly byte[] _buffer = new byte[8];

        public GgufReader(Stream stream)
        {
            _stream = stream;
        }

        public async Task<uint> ReadUInt32Async(CancellationToken ct)
        {
            await _stream.ReadExactlyAsync(_buffer.AsMemory(0, 4), ct);
            return BinaryPrimitives.ReadUInt32LittleEndian(_buffer);
        }

        public async Task<ulong> ReadUInt64Async(CancellationToken ct)
        {
            await _stream.ReadExactlyAsync(_buffer.AsMemory(0, 8), ct);
            return BinaryPrimitives.ReadUInt64LittleEndian(_buffer);
        }

        public async Task<long> ReadInt64Async(CancellationToken ct)
        {
            await _stream.ReadExactlyAsync(_buffer.AsMemory(0, 8), ct);
            return BinaryPrimitives.ReadInt64LittleEndian(_buffer);
        }

        public async Task<int> ReadInt32Async(CancellationToken ct)
        {
            await _stream.ReadExactlyAsync(_buffer.AsMemory(0, 4), ct);
            return BinaryPrimitives.ReadInt32LittleEndian(_buffer);
        }

        public async Task<float> ReadFloat32Async(CancellationToken ct)
        {
            await _stream.ReadExactlyAsync(_buffer.AsMemory(0, 4), ct);
            return BinaryPrimitives.ReadSingleLittleEndian(_buffer);
        }

        public async Task<double> ReadFloat64Async(CancellationToken ct)
        {
            await _stream.ReadExactlyAsync(_buffer.AsMemory(0, 8), ct);
            return BinaryPrimitives.ReadDoubleLittleEndian(_buffer);
        }

        public async Task<byte> ReadByteAsync(CancellationToken ct)
        {
            await _stream.ReadExactlyAsync(_buffer.AsMemory(0, 1), ct);
            return _buffer[0];
        }

        public async Task<ushort> ReadUInt16Async(CancellationToken ct)
        {
            await _stream.ReadExactlyAsync(_buffer.AsMemory(0, 2), ct);
            return BinaryPrimitives.ReadUInt16LittleEndian(_buffer);
        }

        public async Task<bool> ReadBoolAsync(CancellationToken ct)
        {
            await _stream.ReadExactlyAsync(_buffer.AsMemory(0, 1), ct);
            return _buffer[0] != 0;
        }

        public async Task<string> ReadStringAsync(CancellationToken ct)
        {
            var length = await ReadUInt64Async(ct);
            if (length == 0)
                return string.Empty;

            // Limit string length to prevent memory issues
            if (length > 1024 * 1024)
                throw new InvalidDataException($"String length {length} exceeds maximum allowed.");

            var bytes = new byte[length];
            await _stream.ReadExactlyAsync(bytes, ct);
            return Encoding.UTF8.GetString(bytes);
        }

        public async Task<object?> ReadValueAsync(uint valueType, CancellationToken ct)
        {
            // GGUF value types
            return valueType switch
            {
                0 => (int)await ReadByteAsync(ct),           // GGUF_TYPE_UINT8
                1 => (int)(sbyte)await ReadByteAsync(ct),    // GGUF_TYPE_INT8
                2 => (int)await ReadUInt16Async(ct),         // GGUF_TYPE_UINT16
                3 => (int)(short)await ReadUInt16Async(ct),  // GGUF_TYPE_INT16
                4 => await ReadUInt32Async(ct),              // GGUF_TYPE_UINT32
                5 => await ReadInt32Async(ct),               // GGUF_TYPE_INT32
                6 => await ReadFloat32Async(ct),             // GGUF_TYPE_FLOAT32
                7 => await ReadBoolAsync(ct),                // GGUF_TYPE_BOOL
                8 => await ReadStringAsync(ct),              // GGUF_TYPE_STRING
                9 => await ReadArrayAsync(ct),               // GGUF_TYPE_ARRAY
                10 => await ReadUInt64Async(ct),             // GGUF_TYPE_UINT64
                11 => await ReadInt64Async(ct),              // GGUF_TYPE_INT64
                12 => await ReadFloat64Async(ct),            // GGUF_TYPE_FLOAT64
                _ => throw new InvalidDataException($"Unknown GGUF value type: {valueType}")
            };
        }

        private async Task<object?> ReadArrayAsync(CancellationToken ct)
        {
            var elementType = await ReadUInt32Async(ct);
            var length = await ReadUInt64Async(ct);

            // Limit array size
            if (length > 1024 * 1024)
            {
                // Skip large arrays
                return null;
            }

            // For string arrays (common for vocabulary)
            if (elementType == 8) // GGUF_TYPE_STRING
            {
                var strings = new string[length];
                for (ulong i = 0; i < length; i++)
                {
                    strings[i] = await ReadStringAsync(ct);
                }
                return strings;
            }

            // Short numeric arrays carry per-layer architecture (e.g. which layers use sliding-window
            // attention, or a per-layer KV head count) — read them. Long ones (token types, scores) are
            // vocabulary-sized and are skipped.
            if (length <= MaxReadNumericArrayLength)
            {
                var values = new object?[length];
                for (ulong i = 0; i < length; i++)
                    values[i] = await ReadValueAsync(elementType, ct);
                return values;
            }

            var elementSize = GetElementSize(elementType);
            if (elementSize > 0)
            {
                var bytesToSkip = (long)(length * (ulong)elementSize);
                _stream.Seek(bytesToSkip, SeekOrigin.Current);
            }

            return null;
        }

        private static int GetElementSize(uint elementType)
        {
            return elementType switch
            {
                0 or 1 => 1,      // uint8/int8
                2 or 3 => 2,      // uint16/int16
                4 or 5 or 6 => 4, // uint32/int32/float32
                7 => 1,           // bool
                10 or 11 or 12 => 8, // uint64/int64/float64
                _ => 0            // Unknown
            };
        }
    }
}
