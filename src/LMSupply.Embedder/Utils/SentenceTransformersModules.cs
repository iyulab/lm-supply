using System.Text.Json;

namespace LMSupply.Embedder.Utils;

/// <summary>
/// What a sentence-transformers model declares about its own pipeline beyond the tokenizer: how the
/// token embeddings are pooled (<c>1_Pooling/config.json</c>) and, when the repository says so, the
/// prompts a query and a passage are prefixed with (<c>config_sentence_transformers.json</c>).
/// </summary>
/// <remarks>
/// A model loaded by repository id has no catalog entry to take these from. Without reading them the
/// loader pooled every such model with <see cref="PoolingMode.Mean"/> — a CLS-pooling model loaded by
/// id produced vectors in a different space from the same model loaded by alias — and applied no
/// prefixes at all.
/// </remarks>
internal static class SentenceTransformersModules
{
    public const string PoolingConfigPath = "1_Pooling/config.json";
    public const string SentenceTransformersConfigFileName = "config_sentence_transformers.json";

    /// <summary>
    /// The pooling the model declares, from the first of <paramref name="rootDirectories"/> holding
    /// <c>1_Pooling/config.json</c>; <see langword="null"/> when none does, the file is unreadable, or
    /// the declared pooling is one this library does not implement (weighted mean, last token).
    /// </summary>
    public static PoolingMode? TryReadPoolingMode(params string?[] rootDirectories)
    {
        var root = ReadJson(rootDirectories, PoolingConfigPath);
        if (root is not { ValueKind: JsonValueKind.Object } config)
        {
            return null;
        }

        // Exactly one of these is true in a well-formed file. An unsupported mode that is set — or a
        // file where several are set — is "unknown", not a guess.
        var cls = Flag(config, "pooling_mode_cls_token");
        var mean = Flag(config, "pooling_mode_mean_tokens");
        var max = Flag(config, "pooling_mode_max_tokens");
        var meanSqrt = Flag(config, "pooling_mode_mean_sqrt_len_tokens");
        var lastToken = Flag(config, "pooling_mode_lasttoken");
        var weightedMean = Flag(config, "pooling_mode_weightedmean_tokens");

        if (meanSqrt || lastToken || weightedMean)
        {
            return null;
        }

        return (cls, mean, max) switch
        {
            (true, false, false) => PoolingMode.Cls,
            (false, true, false) => PoolingMode.Mean,
            (false, false, true) => PoolingMode.Max,
            _ => null
        };
    }

    /// <summary>
    /// The query and passage prompts the model declares under <c>prompts</c> in
    /// <c>config_sentence_transformers.json</c> (<c>query</c>, and <c>passage</c> or <c>document</c>).
    /// Most repositories declare none; the result is then <c>(null, null)</c>.
    /// </summary>
    public static (string? QueryPrefix, string? PassagePrefix) TryReadPrompts(params string?[] rootDirectories)
    {
        var root = ReadJson(rootDirectories, SentenceTransformersConfigFileName);
        if (root is not { ValueKind: JsonValueKind.Object } config
            || !config.TryGetProperty("prompts", out var prompts)
            || prompts.ValueKind != JsonValueKind.Object)
        {
            return (null, null);
        }

        return (Prompt(prompts, "query"), Prompt(prompts, "passage") ?? Prompt(prompts, "document"));
    }

    private static bool Flag(JsonElement config, string name) =>
        config.TryGetProperty(name, out var value) && value.ValueKind == JsonValueKind.True;

    private static string? Prompt(JsonElement prompts, string name) =>
        prompts.TryGetProperty(name, out var value) && value.ValueKind == JsonValueKind.String && value.GetString() is { Length: > 0 } s
            ? s
            : null;

    private static JsonElement? ReadJson(string?[] rootDirectories, string relativePath)
    {
        foreach (var directory in rootDirectories)
        {
            if (string.IsNullOrEmpty(directory))
            {
                continue;
            }

            var path = Path.Combine(directory, relativePath.Replace('/', Path.DirectorySeparatorChar));
            if (!File.Exists(path))
            {
                continue;
            }

            try
            {
                using var document = JsonDocument.Parse(File.ReadAllText(path));
                return document.RootElement.Clone();
            }
            catch (Exception ex) when (ex is JsonException or IOException or UnauthorizedAccessException)
            {
                // An unreadable optional file is the same as an absent one.
                return null;
            }
        }

        return null;
    }
}
