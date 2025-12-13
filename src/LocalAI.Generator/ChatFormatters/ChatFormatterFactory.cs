using LocalAI.Generator.Abstractions;

namespace LocalAI.Generator.ChatFormatters;

/// <summary>
/// Factory for creating chat formatters based on model name.
/// </summary>
public static class ChatFormatterFactory
{
    /// <summary>
    /// Creates a chat formatter based on the model name.
    /// </summary>
    /// <param name="modelName">The model name or identifier.</param>
    /// <returns>An appropriate chat formatter for the model.</returns>
    public static IChatFormatter Create(string modelName)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(modelName);

        var lowerName = modelName.ToLowerInvariant();

        return lowerName switch
        {
            // Phi models
            var n when n.Contains("phi-4") => new Phi3ChatFormatter(),
            var n when n.Contains("phi-3") => new Phi3ChatFormatter(),
            var n when n.Contains("phi3") => new Phi3ChatFormatter(),

            // Llama models
            var n when n.Contains("llama-3") => new Llama3ChatFormatter(),
            var n when n.Contains("llama3") => new Llama3ChatFormatter(),
            var n when n.Contains("llama-2") => new Llama3ChatFormatter(), // Similar format

            // Qwen models (use ChatML)
            var n when n.Contains("qwen") => new ChatMLFormatter(),

            // Mistral models using ChatML
            var n when n.Contains("mistral") && n.Contains("instruct") => new ChatMLFormatter(),

            // Default to Phi3 format (most common for ONNX models)
            _ => new Phi3ChatFormatter()
        };
    }

    /// <summary>
    /// Creates a chat formatter by format name.
    /// </summary>
    /// <param name="formatName">The format name (phi3, llama3, chatml).</param>
    /// <returns>The corresponding chat formatter.</returns>
    public static IChatFormatter CreateByFormat(string formatName)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(formatName);

        return formatName.ToLowerInvariant() switch
        {
            "phi3" or "phi-3" or "phi" => new Phi3ChatFormatter(),
            "llama3" or "llama-3" or "llama" => new Llama3ChatFormatter(),
            "chatml" or "qwen" => new ChatMLFormatter(),
            _ => throw new ArgumentException($"Unknown chat format: {formatName}", nameof(formatName))
        };
    }
}
