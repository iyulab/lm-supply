// This file is copied verbatim into every package that estimates tokens from characters, so that each of them gives the same
// answer for the same text. Change every copy together.
namespace TokenEstimation;

/// <summary>
/// Characters per token for each kind of character. A text's estimate is the sum over its characters of 1 / ratio of their kind.
/// </summary>
/// <remarks>
/// <see cref="O200k"/> is fitted to OpenAI's o200k_base over English chat turns, Korean passages, mixed Korean/English prose,
/// C# source and JSON (2026-10-16): with it each set's estimate is within 2 % of o200k on the sum. Other tokenizers differ, most
/// on Korean, so an estimate stays approximate whatever the model.
/// </remarks>
internal readonly record struct CharTokenRatios(
    double Latin, double Korean, double Japanese, double Chinese, double Digit, double Punctuation, double WhitespaceRun, double Other)
{
    /// <summary>Fitted to o200k_base, which OpenAI's models use and other cloud models stay close to.</summary>
    internal static readonly CharTokenRatios O200k = new(
        Latin: 7.0, Korean: 1.35, Japanese: 1.5, Chinese: 1.2, Digit: 1.3, Punctuation: 1.25, WhitespaceRun: 2.75, Other: 3.5);

    /// <summary>Fitted to Gemma 4's tokenizer; Japanese and Chinese keep the o200k values (not measured).</summary>
    internal static readonly CharTokenRatios Gemma = new(
        Latin: 6.23, Korean: 1.46, Japanese: 1.5, Chinese: 1.2, Digit: 0.93, Punctuation: 0.91, WhitespaceRun: 3.80, Other: 3.86);

    /// <summary>Fitted to Qwen 3.5's tokenizer; Japanese and Chinese keep the o200k values (not measured).</summary>
    internal static readonly CharTokenRatios Qwen = new(
        Latin: 6.84, Korean: 1.65, Japanese: 1.5, Chinese: 1.2, Digit: 0.93, Punctuation: 1.03, WhitespaceRun: 2.93, Other: 3.80);
}

/// <summary>The kind a character is charged as.</summary>
internal enum CharKind
{
    /// <summary>Not charged: a control character, or whitespace that continues a run.</summary>
    None,
    Latin,
    Korean,
    Japanese,
    Chinese,
    Digit,
    Punctuation,
    WhitespaceRun,
    Other,
}

/// <summary>How many characters of each kind a text holds (a run of whitespace counts once).</summary>
internal readonly record struct CharKindCounts(
    int Latin, int Korean, int Japanese, int Chinese, int Digits, int Punctuation, int WhitespaceRuns, int Other)
{
    public int Total => Latin + Korean + Japanese + Chinese + Digits + Punctuation + WhitespaceRuns + Other;
}

/// <summary>Estimates a text's token count from the kinds of its characters.</summary>
internal static class CharTokenEstimator
{
    /// <summary>The estimated token count of <paramref name="text"/>: 0 for an empty text, otherwise the cost rounded up.</summary>
    public static int Count(string? text, in CharTokenRatios ratios)
    {
        if (string.IsNullOrEmpty(text))
            return 0;

        var c = Classify(text);
        if (c.Total == 0)
            return 0;

        double tokens = 0;
        tokens += c.Latin / ratios.Latin;
        tokens += c.Korean / ratios.Korean;
        tokens += c.Japanese / ratios.Japanese;
        tokens += c.Chinese / ratios.Chinese;
        tokens += c.Digits / ratios.Digit;
        tokens += c.Punctuation / ratios.Punctuation;
        tokens += c.WhitespaceRuns / ratios.WhitespaceRun;
        tokens += c.Other / ratios.Other;

        return (int)Math.Ceiling(tokens);
    }

    /// <summary>Counts the characters of each kind in <paramref name="text"/>.</summary>
    public static CharKindCounts Classify(string text)
    {
        int latin = 0, korean = 0, japanese = 0, chinese = 0, digits = 0, punctuation = 0, whitespaceRuns = 0, other = 0;
        var inWhitespace = false;

        foreach (var ch in text)
        {
            switch (KindOf(ch, ref inWhitespace))
            {
                case CharKind.Latin: latin++; break;
                case CharKind.Korean: korean++; break;
                case CharKind.Japanese: japanese++; break;
                case CharKind.Chinese: chinese++; break;
                case CharKind.Digit: digits++; break;
                case CharKind.Punctuation: punctuation++; break;
                case CharKind.WhitespaceRun: whitespaceRuns++; break;
                case CharKind.Other: other++; break;
            }
        }

        return new CharKindCounts(latin, korean, japanese, chinese, digits, punctuation, whitespaceRuns, other);
    }

    /// <summary>
    /// The kind <paramref name="ch"/> is charged as, given whether the character before it was whitespace (updated). Walking a
    /// text with this and summing <see cref="Cost"/> gives the same total as <see cref="Count"/> before rounding.
    /// </summary>
    public static CharKind KindOf(char ch, ref bool inWhitespace)
    {
        if (char.IsWhiteSpace(ch))
        {
            // A run of whitespace counts once: a tokenizer merges indentation and line breaks
            if (inWhitespace)
                return CharKind.None;
            inWhitespace = true;
            return CharKind.WhitespaceRun;
        }

        inWhitespace = false;
        if (char.IsControl(ch))
            return CharKind.None;
        if (IsKorean(ch))
            return CharKind.Korean;
        if (IsJapanese(ch))
            return CharKind.Japanese;
        if (IsChinese(ch))
            return CharKind.Chinese;
        if (ch <= 'ɏ' && char.IsLetter(ch))
            return CharKind.Latin;
        if (char.IsDigit(ch))
            return CharKind.Digit;
        if (char.IsPunctuation(ch) || char.IsSymbol(ch))
            return CharKind.Punctuation;
        return CharKind.Other;
    }

    /// <summary>The tokens one character of <paramref name="kind"/> costs.</summary>
    public static double Cost(CharKind kind, in CharTokenRatios ratios) => kind switch
    {
        CharKind.Latin => 1 / ratios.Latin,
        CharKind.Korean => 1 / ratios.Korean,
        CharKind.Japanese => 1 / ratios.Japanese,
        CharKind.Chinese => 1 / ratios.Chinese,
        CharKind.Digit => 1 / ratios.Digit,
        CharKind.Punctuation => 1 / ratios.Punctuation,
        CharKind.WhitespaceRun => 1 / ratios.WhitespaceRun,
        CharKind.Other => 1 / ratios.Other,
        _ => 0,
    };

    // Hangul syllables, Hangul Jamo, Hangul compatibility Jamo
    private static bool IsKorean(char c) =>
        c is (>= '가' and <= '힯') or (>= 'ᄀ' and <= 'ᇿ') or (>= '㄰' and <= '㆏');

    // Hiragana, Katakana (Kanji counts as Chinese)
    private static bool IsJapanese(char c) => c is >= '぀' and <= 'ヿ';

    // CJK unified ideographs, extension A
    private static bool IsChinese(char c) => c is (>= '一' and <= '鿿') or (>= '㐀' and <= '䶿');
}
