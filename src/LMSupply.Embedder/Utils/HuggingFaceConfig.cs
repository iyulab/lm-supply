using System.Text.Json;

namespace LMSupply.Embedder.Utils;

/// <summary>
/// What a HuggingFace transformers model declares about itself in <c>config.json</c>.
/// </summary>
internal static class HuggingFaceConfig
{
    public const string FileName = "config.json";

    /// <summary>
    /// The model's <c>hidden_size</c> from the first of <paramref name="directories"/> that holds the
    /// file, or <c>null</c> when none does or the value is missing or not a positive integer. This is the
    /// width of the token embeddings the encoder produces — the vector dimension after pooling for the
    /// sentence-transformers models this library loads.
    /// </summary>
    public static int? TryReadHiddenSize(params string?[] directories)
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
                    && document.RootElement.TryGetProperty("hidden_size", out var value)
                    && value.ValueKind == JsonValueKind.Number
                    && value.TryGetInt32(out var size)
                    && size > 0)
                {
                    return size;
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
}
