namespace LMSupply.Generator.Gguf;

/// <summary>
/// The generator's view of GGUF metadata: the memory configuration the loader and the memory estimator plan with. The
/// metadata itself (<see cref="GgufMetadata"/>) lives in LMSupply.Llama, shared with the GGUF embedder and reranker.
/// </summary>
public static class GgufMetadataMemoryExtensions
{
    /// <summary>
    /// Creates a ModelMemoryConfig from GGUF metadata.
    /// </summary>
    /// <param name="metadata">The metadata read from the file.</param>
    /// <param name="contextLength">Override context length (uses metadata value if null).</param>
    /// <returns>Memory configuration, or null if insufficient metadata.</returns>
    public static ModelMemoryConfig? ToMemoryConfig(this GgufMetadata metadata, int? contextLength = null)
    {
        if (!metadata.LayerCount.HasValue || !metadata.EmbeddingLength.HasValue)
            return null;

        // Estimate parameter count if not available
        var paramCount = metadata.ParameterCountOrEstimate;

        return new ModelMemoryConfig
        {
            ParameterCount = paramCount,
            NumLayers = metadata.LayerCount.Value,
            HiddenSize = metadata.EmbeddingLength.Value,
            ContextLength = contextLength ?? metadata.ContextLength ?? 4096,
            Quantization = metadata.InferQuantization()
        };
    }

    /// <summary>
    /// Infers quantization type from file type or quantization string.
    /// </summary>
    private static Quantization InferQuantization(this GgufMetadata metadata)
    {
        // Infer from quantization type string
        if (!string.IsNullOrEmpty(metadata.QuantizationType))
        {
            var qt = metadata.QuantizationType.ToUpperInvariant();
            if (qt.Contains("Q4") || qt.Contains("INT4"))
                return Quantization.Quant4;
            if (qt.Contains("Q8") || qt.Contains("INT8"))
                return Quantization.Quant8;
            if (qt.Contains("F16") || qt.Contains("FP16"))
                return Quantization.FP16;
            if (qt.Contains("F32") || qt.Contains("FP32"))
                return Quantization.FP32;
        }

        // Infer from GGUF file type enum
        if (metadata.FileType.HasValue)
        {
            // GGUF file types (from llama.cpp/ggml.h):
            // 0: F32, 1: F16, 2-6: Q4_X, 7-8: Q8_X, etc.
            return metadata.FileType.Value switch
            {
                0 => Quantization.FP32,
                1 => Quantization.FP16,
                >= 2 and <= 6 => Quantization.Quant4,
                >= 7 and <= 8 => Quantization.Quant8,
                _ => Quantization.Quant4 // Default to Quant4 for quantized models
            };
        }

        return Quantization.Quant4; // Most GGUF files are quantized
    }
}
