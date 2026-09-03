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
    /// Resolves the path to genai_config.json by searching the given directories in order.
    /// </summary>
    /// <param name="searchPaths">Directories to search, in priority order.</param>
    /// <returns>The full path to the config file, or null if not found.</returns>
    internal static string? ResolveConfigPath(params string[] searchPaths)
    {
        foreach (var dir in searchPaths)
        {
            var candidate = Path.Combine(dir, ConfigFileName);
            if (File.Exists(candidate))
                return candidate;
        }
        return null;
    }

    /// <summary>
    /// Reads the maximum context length from the model's genai_config.json.
    /// Prioritizes model architecture limits (context_length, max_position_embeddings)
    /// over generation defaults (search.max_length) which are often set to small values.
    /// </summary>
    /// <param name="modelPath">Path to the model directory (ONNX files location).</param>
    /// <param name="configBasePath">Optional base directory to search for genai_config.json if not found in modelPath.</param>
    /// <returns>The maximum context length, or default if not found.</returns>
    public static int ReadMaxContextLength(string modelPath, string? configBasePath = null)
    {
        var configPath = configBasePath != null
            ? ResolveConfigPath(modelPath, configBasePath)
            : ResolveConfigPath(modelPath);
        if (configPath == null)
        {
            return DefaultMaxContextLength;
        }

        try
        {
            var json = File.ReadAllText(configPath);
            using var doc = JsonDocument.Parse(json);
            var root = doc.RootElement;

            // Priority 1: model.context_length (model architecture limit)
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

            // Priority 2: Direct context_length at root
            if (root.TryGetProperty("context_length", out var rootCtxLen))
            {
                return rootCtxLen.GetInt32();
            }

            // Priority 3: search.max_length (GenAI default, often too small)
            // Only use if >= 1024, otherwise it's likely a conservative default
            if (root.TryGetProperty("search", out var searchSection))
            {
                if (searchSection.TryGetProperty("max_length", out var maxLen))
                {
                    var searchMaxLength = maxLen.GetInt32();
                    if (searchMaxLength >= 1024)
                    {
                        return searchMaxLength;
                    }
                    // Ignore small search.max_length values (e.g., 512 default)
                    // These are generation defaults, not model limits
                }
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
    /// <param name="modelPath">Path to the model directory (ONNX files location).</param>
    /// <param name="configBasePath">Optional base directory to search for genai_config.json if not found in modelPath.</param>
    /// <returns>The model type, or null if not found.</returns>
    public static string? ReadModelType(string modelPath, string? configBasePath = null)
    {
        var configPath = configBasePath != null
            ? ResolveConfigPath(modelPath, configBasePath)
            : ResolveConfigPath(modelPath);
        if (configPath == null)
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
    /// <param name="modelPath">Path to the model directory (ONNX files location).</param>
    /// <param name="configBasePath">Optional base directory to search for genai_config.json if not found in modelPath.</param>
    /// <returns>The vocabulary size, or null if not found.</returns>
    public static int? ReadVocabSize(string modelPath, string? configBasePath = null)
    {
        var configPath = configBasePath != null
            ? ResolveConfigPath(modelPath, configBasePath)
            : ResolveConfigPath(modelPath);
        if (configPath == null)
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


    /// <summary>
    /// Reads the EOS (End-of-Sequence) token IDs from the config.
    /// ONNX Runtime GenAI supports both single int and array formats for eos_token_id.
    /// </summary>
    /// <param name="modelPath">Path to the model directory (ONNX files location).</param>
    /// <param name="configBasePath">Optional base directory to search for genai_config.json if not found in modelPath.</param>
    /// <returns>Array of EOS token IDs, or empty array if not found.</returns>
    public static int[] ReadEosTokenIds(string modelPath, string? configBasePath = null)
    {
        var configPath = configBasePath != null
            ? ResolveConfigPath(modelPath, configBasePath)
            : ResolveConfigPath(modelPath);
        if (configPath == null)
        {
            return [];
        }

        try
        {
            var json = File.ReadAllText(configPath);
            using var doc = JsonDocument.Parse(json);
            var root = doc.RootElement;

            // Try model.eos_token_id first (common location)
            if (root.TryGetProperty("model", out var modelSection))
            {
                if (modelSection.TryGetProperty("eos_token_id", out var eosToken))
                {
                    return ParseEosTokenId(eosToken);
                }
            }

            // Try search.eos_token_id (GenAI specific)
            if (root.TryGetProperty("search", out var searchSection))
            {
                if (searchSection.TryGetProperty("eos_token_id", out var eosToken))
                {
                    return ParseEosTokenId(eosToken);
                }
            }

            return [];
        }
        catch (JsonException)
        {
            return [];
        }
    }

    /// <summary>
    /// Reads stop sequences from the config if defined.
    /// </summary>
    /// <param name="modelPath">Path to the model directory (ONNX files location).</param>
    /// <param name="configBasePath">Optional base directory to search for genai_config.json if not found in modelPath.</param>
    /// <returns>List of stop sequences, or empty list if not found.</returns>
    public static IReadOnlyList<string> ReadStopSequences(string modelPath, string? configBasePath = null)
    {
        var configPath = configBasePath != null
            ? ResolveConfigPath(modelPath, configBasePath)
            : ResolveConfigPath(modelPath);
        if (configPath == null)
        {
            return [];
        }

        try
        {
            var json = File.ReadAllText(configPath);
            using var doc = JsonDocument.Parse(json);
            var root = doc.RootElement;

            // Try search.stop_strings (GenAI specific)
            if (root.TryGetProperty("search", out var searchSection))
            {
                if (searchSection.TryGetProperty("stop_strings", out var stopStrings) && 
                    stopStrings.ValueKind == JsonValueKind.Array)
                {
                    var result = new List<string>();
                    foreach (var item in stopStrings.EnumerateArray())
                    {
                        var str = item.GetString();
                        if (!string.IsNullOrEmpty(str))
                        {
                            result.Add(str);
                        }
                    }
                    return result;
                }
            }

            return [];
        }
        catch (JsonException)
        {
            return [];
        }
    }

    private static int[] ParseEosTokenId(JsonElement element)
    {
        if (element.ValueKind == JsonValueKind.Number)
        {
            // Single EOS token ID
            return [element.GetInt32()];
        }
        else if (element.ValueKind == JsonValueKind.Array)
        {
            // Array of EOS token IDs (e.g., Phi-3 uses [32007, 32001, 32000])
            var result = new List<int>();
            foreach (var item in element.EnumerateArray())
            {
                if (item.ValueKind == JsonValueKind.Number)
                {
                    result.Add(item.GetInt32());
                }
            }
            return result.ToArray();
        }
        return [];
    }
}
