using System.Text.Json;

namespace LMSupply.Embedder.Utils;

/// <summary>
/// What a sentence-transformers model declares about itself in <c>sentence_bert_config.json</c>.
/// </summary>
internal static class SentenceBertConfig
{
    public const string FileName = "sentence_bert_config.json";

    /// <summary>
    /// The model's <c>max_seq_length</c> from the first of <paramref name="directories"/> that holds
    /// the file, or <c>null</c> when none does or the value is missing or not a positive integer.
    /// </summary>
    public static int? TryReadMaxSequenceLength(params string?[] directories)
    {
        foreach (var directory in directories)
        {
            if (string.IsNullOrEmpty(directory)) continue;

            var path = Path.Combine(directory, FileName);
            if (!File.Exists(path)) continue;

            try
            {
                using var document = JsonDocument.Parse(File.ReadAllText(path));
                if (document.RootElement.ValueKind == JsonValueKind.Object
                    && document.RootElement.TryGetProperty("max_seq_length", out var value)
                    && value.ValueKind == JsonValueKind.Number
                    && value.TryGetInt32(out var length)
                    && length > 0)
                {
                    return length;
                }
            }
            catch (Exception ex) when (ex is JsonException or IOException or UnauthorizedAccessException)
            {
                // An unreadable optional file is the same as an absent one.
            }

            return null;
        }

        return null;
    }

    /// <summary>
    /// The sequence length to tokenize with. An explicit caller value wins; otherwise what the model
    /// declares about itself, then what the catalog knows, then <see cref="EmbedderOptions.DefaultMaxSequenceLength"/>.
    /// </summary>
    public static int ResolveMaxSequenceLength(int? callerValue, int? declaredByModel, int? catalogValue) =>
        callerValue ?? declaredByModel ?? catalogValue ?? EmbedderOptions.DefaultMaxSequenceLength;
}
