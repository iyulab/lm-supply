namespace LMSupply.Generator.Models;

/// <summary>
/// Specifies the model file format for text generation.
/// </summary>
public enum ModelFormat
{
    /// <summary>
    /// ONNX Runtime GenAI format (.onnx files with genai_config.json).
    /// Used by Microsoft's ONNX-converted models.
    /// </summary>
    Onnx,

    /// <summary>
    /// GGUF format (.gguf files) used by llama.cpp.
    /// Popular for quantized models on HuggingFace.
    /// </summary>
    Gguf,

    /// <summary>
    /// Unknown or unsupported format.
    /// </summary>
    Unknown
}
