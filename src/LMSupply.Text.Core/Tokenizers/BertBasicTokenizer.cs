using System.Globalization;

namespace LMSupply.Text;

/// <summary>
/// What a BERT-family model expects to happen to text before WordPiece sees it.
/// </summary>
/// <param name="CleanText">Drop control characters and turn every whitespace character into a space.</param>
/// <param name="HandleChineseChars">Make every CJK ideograph a word of its own.</param>
/// <param name="StripAccents">Remove combining marks; <see langword="null"/> follows <paramref name="Lowercase"/>.</param>
/// <param name="Lowercase">Lowercase the text (uncased vocabularies hold no uppercase pieces).</param>
/// <param name="SplitPunctuation">Make every punctuation character a word of its own.</param>
internal sealed record BertNormalization(
    bool CleanText = true,
    bool HandleChineseChars = true,
    bool? StripAccents = null,
    bool Lowercase = true,
    bool SplitPunctuation = true)
{
    /// <summary>The uncased BERT convention: what a model means when it says nothing else.</summary>
    public static BertNormalization Uncased { get; } = new();

    /// <summary>
    /// The convention as a short ASCII string, with <see cref="StripAccents"/> resolved to the value
    /// actually applied. Part of <see cref="ISequenceTokenizer.Signature"/>.
    /// </summary>
    public string Signature =>
        $"clean={Flag(CleanText)};chinese={Flag(HandleChineseChars)};accents={Flag(StripAccents ?? Lowercase)};" +
        $"lower={Flag(Lowercase)};punct={Flag(SplitPunctuation)}";

    private static char Flag(bool value) => value ? '1' : '0';

    /// <summary>
    /// Reads the convention a model declares. <c>tokenizer.json</c> is the authority (its
    /// <c>normalizer</c> and <c>pre_tokenizer</c> are what the reference implementation executes), then
    /// <c>tokenizer_config.json</c>. A model that ships neither is judged by its vocabulary: an uncased
    /// vocabulary holds no uppercase pieces, so one that does is cased.
    /// </summary>
    public static BertNormalization Load(string modelDir, Func<IEnumerable<string>>? vocabulary = null)
    {
        var tokenizerJson = Path.Combine(modelDir, "tokenizer.json");
        if (File.Exists(tokenizerJson) && TryFromTokenizerJson(tokenizerJson, out var fromJson))
            return fromJson;

        var configJson = Path.Combine(modelDir, "tokenizer_config.json");
        if (File.Exists(configJson) && TryFromTokenizerConfig(configJson, out var fromConfig))
            return fromConfig;

        if (vocabulary != null && LooksCased(vocabulary()))
            return Uncased with { Lowercase = false, StripAccents = false };

        return Uncased;
    }

    private static bool TryFromTokenizerJson(string path, out BertNormalization result)
    {
        result = Uncased;
        try
        {
            using var doc = JsonDocument.Parse(File.ReadAllText(path));
            var root = doc.RootElement;

            var hasNormalizer = root.TryGetProperty("normalizer", out var normalizer);
            var hasPreTokenizer = root.TryGetProperty("pre_tokenizer", out var preTokenizer);
            if (!hasNormalizer && !hasPreTokenizer)
                return false;

            // No normalizer declared means none is applied.
            var clean = false;
            var chinese = false;
            var lower = false;
            bool? strip = false;

            foreach (var step in Flatten(normalizer, hasNormalizer))
            {
                switch (TypeOf(step))
                {
                    case "BertNormalizer":
                        clean = Flag(step, "clean_text", true);
                        chinese = Flag(step, "handle_chinese_chars", true);
                        lower = Flag(step, "lowercase", true);
                        strip = step.TryGetProperty("strip_accents", out var s) && s.ValueKind is JsonValueKind.True or JsonValueKind.False
                            ? s.GetBoolean()
                            : null;
                        break;
                    case "Lowercase":
                        lower = true;
                        break;
                    case "StripAccents":
                        strip = true;
                        break;
                }
            }

            var splitPunctuation = false;
            foreach (var step in Flatten(preTokenizer, hasPreTokenizer))
            {
                if (TypeOf(step) is "BertPreTokenizer" or "Punctuation")
                    splitPunctuation = true;
            }

            result = new BertNormalization(clean, chinese, strip, lower, splitPunctuation);
            return true;
        }
        catch (JsonException)
        {
            return false;
        }
    }

    private static bool TryFromTokenizerConfig(string path, out BertNormalization result)
    {
        result = Uncased;
        try
        {
            using var doc = JsonDocument.Parse(File.ReadAllText(path));
            var root = doc.RootElement;
            if (!root.TryGetProperty("do_lower_case", out _) && !root.TryGetProperty("tokenize_chinese_chars", out _)
                && !root.TryGetProperty("strip_accents", out _))
            {
                return false;
            }

            result = new BertNormalization(
                CleanText: true,
                HandleChineseChars: Flag(root, "tokenize_chinese_chars", true),
                StripAccents: root.TryGetProperty("strip_accents", out var s) && s.ValueKind is JsonValueKind.True or JsonValueKind.False
                    ? s.GetBoolean()
                    : null,
                Lowercase: Flag(root, "do_lower_case", true),
                SplitPunctuation: true);
            return true;
        }
        catch (JsonException)
        {
            return false;
        }
    }

    private static IEnumerable<JsonElement> Flatten(JsonElement element, bool present)
    {
        if (!present || element.ValueKind != JsonValueKind.Object)
            yield break;

        if (TypeOf(element) == "Sequence")
        {
            var listName = element.TryGetProperty("normalizers", out _) ? "normalizers" : "pretokenizers";
            if (element.TryGetProperty(listName, out var list) && list.ValueKind == JsonValueKind.Array)
            {
                foreach (var item in list.EnumerateArray())
                    yield return item;
            }

            yield break;
        }

        yield return element;
    }

    private static string? TypeOf(JsonElement element)
        => element.ValueKind == JsonValueKind.Object && element.TryGetProperty("type", out var t) ? t.GetString() : null;

    private static bool Flag(JsonElement element, string name, bool fallback)
        => element.TryGetProperty(name, out var v) && v.ValueKind is JsonValueKind.True or JsonValueKind.False
            ? v.GetBoolean()
            : fallback;

    private static bool LooksCased(IEnumerable<string> vocabulary)
    {
        foreach (var piece in vocabulary)
        {
            // [CLS], [unused12] and the like are uppercase in every BERT vocabulary.
            if (piece.Length == 0 || piece[0] == '[')
                continue;

            foreach (var c in piece)
            {
                if (char.IsUpper(c))
                    return true;
            }
        }

        return false;
    }
}

/// <summary>
/// BERT's basic tokenization: the step between raw text and WordPiece. WordPiece only looks a
/// whitespace-separated word up in the vocabulary, piece by piece; it is this step that lowercases,
/// strips accents, and separates punctuation and CJK ideographs into words. Without it, on an uncased
/// vocabulary, every capitalized word and every word with punctuation attached is <c>[UNK]</c>.
/// </summary>
/// <remarks>
/// Follows the reference normalizer and pre-tokenizer (clean text, space out CJK ideographs, NFD and
/// drop combining marks, lowercase, isolate punctuation), in that order. The result is the words joined
/// by single spaces, which is the input a whitespace-splitting WordPiece tokenizer expects.
/// </remarks>
internal sealed class BertBasicTokenizer
{
    private readonly BertNormalization _options;

    public BertBasicTokenizer(BertNormalization options) => _options = options;

    public BertNormalization Options => _options;

    public string Normalize(string text)
    {
        if (string.IsNullOrEmpty(text))
            return string.Empty;

        var stripAccents = _options.StripAccents ?? _options.Lowercase;
        var source = stripAccents ? text.Normalize(NormalizationForm.FormD) : text;

        var builder = new StringBuilder(source.Length + 16);
        foreach (var rune in source.EnumerateRunes())
        {
            var category = Rune.GetUnicodeCategory(rune);

            if (_options.CleanText)
            {
                if (rune.Value is 0 or 0xFFFD || IsControl(rune, category))
                    continue;
            }

            if (IsWhitespace(rune, category))
            {
                builder.Append(' ');
                continue;
            }

            if (stripAccents && category == UnicodeCategory.NonSpacingMark)
                continue;

            var isolate = (_options.HandleChineseChars && IsCjkIdeograph(rune.Value))
                          || (_options.SplitPunctuation && IsPunctuation(rune, category));

            if (isolate)
                builder.Append(' ');

            if (_options.Lowercase)
                builder.Append(rune.ToString().ToLowerInvariant());
            else
                builder.Append(rune.ToString());

            if (isolate)
                builder.Append(' ');
        }

        return CollapseSpaces(builder);
    }

    private static string CollapseSpaces(StringBuilder builder)
    {
        var result = new StringBuilder(builder.Length);
        var pendingSpace = false;
        for (var i = 0; i < builder.Length; i++)
        {
            var c = builder[i];
            if (c == ' ')
            {
                pendingSpace = result.Length > 0;
                continue;
            }

            if (pendingSpace)
            {
                result.Append(' ');
                pendingSpace = false;
            }

            result.Append(c);
        }

        return result.ToString();
    }

    // Tab, line feed and carriage return are whitespace here, not control characters.
    private static bool IsWhitespace(Rune rune, UnicodeCategory category)
        => rune.Value is ' ' or '\t' or '\n' or '\r'
           || category is UnicodeCategory.SpaceSeparator or UnicodeCategory.LineSeparator or UnicodeCategory.ParagraphSeparator;

    private static bool IsControl(Rune rune, UnicodeCategory category)
    {
        if (rune.Value is '\t' or '\n' or '\r')
            return false;

        return category is UnicodeCategory.Control or UnicodeCategory.Format
            or UnicodeCategory.PrivateUse or UnicodeCategory.Surrogate or UnicodeCategory.OtherNotAssigned;
    }

    // BERT counts every non-alphanumeric ASCII character as punctuation — "$", "+", "=", "^", "`", "|",
    // "~", "<" and ">" included, although Unicode files those under symbols.
    private static bool IsPunctuation(Rune rune, UnicodeCategory category)
    {
        var v = rune.Value;
        if (v is >= 33 and <= 47 or >= 58 and <= 64 or >= 91 and <= 96 or >= 123 and <= 126)
            return true;

        return category is UnicodeCategory.ConnectorPunctuation or UnicodeCategory.DashPunctuation
            or UnicodeCategory.OpenPunctuation or UnicodeCategory.ClosePunctuation
            or UnicodeCategory.InitialQuotePunctuation or UnicodeCategory.FinalQuotePunctuation
            or UnicodeCategory.OtherPunctuation;
    }

    // The CJK Unified Ideographs blocks. Hiragana, Katakana and Hangul are written with spaces or are
    // handled by WordPiece itself, and are deliberately not in this list.
    private static bool IsCjkIdeograph(int v)
        => v is >= 0x4E00 and <= 0x9FFF
            or >= 0x3400 and <= 0x4DBF
            or >= 0x20000 and <= 0x2A6DF
            or >= 0x2A700 and <= 0x2B73F
            or >= 0x2B740 and <= 0x2B81F
            or >= 0x2B820 and <= 0x2CEAF
            or >= 0xF900 and <= 0xFAFF
            or >= 0x2F800 and <= 0x2FA1F;
}
