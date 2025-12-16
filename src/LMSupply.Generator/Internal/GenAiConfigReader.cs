using System.Text.Json;

namespace LMSupply.Generator.Internal;

/// <summary>
/// Reads configuration from ONNX GenAI model config files.
/// </summary>
internal static class GenAiConfigReader
{
    private const string ConfigFileName = "genai_config.json";
    private const int DefaultMaxContextLength = 4096;

    /// <summary>
    /// Reads the maximum context length from the model's genai_config.json.
    /// </summary>
    /// <param name="modelPath">Path to the model directory.</param>
    /// <returns>The maximum context length, or default if not found.</returns>
    public static int ReadMaxContextLength(string modelPath)
    {
        var configPath = Path.Combine(modelPath, ConfigFileName);
        if (!File.Exists(configPath))
        {
            return DefaultMaxContextLength;
        }

        try
        {
            var json = File.ReadAllText(configPath);
            using var doc = JsonDocument.Parse(json);
            var root = doc.RootElement;

            // Try different possible locations for context length
            // 1. model.context_length (common location)
            if (root.TryGetProperty("model", out var modelSection))
            {
                if (modelSection.TryGetProperty("context_length", out var ctxLen))
                {
                    return ctxLen.GetInt32();
                }

                // Some models use max_position_embeddings
                if (modelSection.TryGetProperty("max_position_embeddings", out var maxPos))
                {
                    return maxPos.GetInt32();
                }
            }

            // 2. search.max_length (GenAI specific)
            if (root.TryGetProperty("search", out var searchSection))
            {
                if (searchSection.TryGetProperty("max_length", out var maxLen))
                {
                    return maxLen.GetInt32();
                }
            }

            // 3. Direct context_length at root
            if (root.TryGetProperty("context_length", out var rootCtxLen))
            {
                return rootCtxLen.GetInt32();
            }

            return DefaultMaxContextLength;
        }
        catch (JsonException)
        {
            return DefaultMaxContextLength;
        }
    }

    /// <summary>
    /// Reads the model type/architecture from the config.
    /// </summary>
    /// <param name="modelPath">Path to the model directory.</param>
    /// <returns>The model type, or null if not found.</returns>
    public static string? ReadModelType(string modelPath)
    {
        var configPath = Path.Combine(modelPath, ConfigFileName);
        if (!File.Exists(configPath))
        {
            return null;
        }

        try
        {
            var json = File.ReadAllText(configPath);
            using var doc = JsonDocument.Parse(json);
            var root = doc.RootElement;

            if (root.TryGetProperty("model", out var modelSection))
            {
                if (modelSection.TryGetProperty("type", out var modelType))
                {
                    return modelType.GetString();
                }
            }

            return null;
        }
        catch (JsonException)
        {
            return null;
        }
    }

    /// <summary>
    /// Reads the vocabulary size from the config.
    /// </summary>
    /// <param name="modelPath">Path to the model directory.</param>
    /// <returns>The vocabulary size, or null if not found.</returns>
    public static int? ReadVocabSize(string modelPath)
    {
        var configPath = Path.Combine(modelPath, ConfigFileName);
        if (!File.Exists(configPath))
        {
            return null;
        }

        try
        {
            var json = File.ReadAllText(configPath);
            using var doc = JsonDocument.Parse(json);
            var root = doc.RootElement;

            if (root.TryGetProperty("model", out var modelSection))
            {
                if (modelSection.TryGetProperty("vocab_size", out var vocabSize))
                {
                    return vocabSize.GetInt32();
                }
            }

            return null;
        }
        catch (JsonException)
        {
            return null;
        }
    }
}
