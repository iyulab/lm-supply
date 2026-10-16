namespace LMSupply.Synthesizer.Core.English;

/// <summary>
/// Numbers as the words that say them, in the American style (no «and» after «hundred»).
/// </summary>
internal static class NumberWords
{
    // Past this many digits a number is read digit by digit: nobody names the magnitude.
    public const int MaxCardinalDigits = 15;

    private static readonly string[] Ones =
    [
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
        "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"
    ];

    private static readonly string[] Tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"];

    private static readonly string[] Thousands = ["", "thousand", "million", "billion", "trillion"];

    private static readonly Dictionary<string, string> IrregularOrdinals = new(StringComparer.Ordinal)
    {
        ["one"] = "first", ["two"] = "second", ["three"] = "third", ["five"] = "fifth", ["eight"] = "eighth",
        ["nine"] = "ninth", ["twelve"] = "twelfth",
    };

    /// <summary>Appends the words of a whole number below 10^15.</summary>
    public static void AppendCardinal(long value, List<string> words)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(value);
        if (value == 0)
        {
            words.Add(Ones[0]);
            return;
        }

        // The groups of three digits, highest first.
        Span<int> groups = stackalloc int[Thousands.Length];
        var count = 0;
        for (; value > 0; value /= 1000)
            groups[count++] = (int)(value % 1000);

        for (var i = count - 1; i >= 0; i--)
        {
            if (groups[i] == 0)
                continue;

            AppendBelowThousand(groups[i], words);
            if (i > 0)
                words.Add(Thousands[i]);
        }
    }

    /// <summary>Appends each digit's name: how a number with leading zeros, or one too long to name, is read.</summary>
    public static void AppendDigits(ReadOnlySpan<char> digits, List<string> words)
    {
        foreach (var digit in digits)
            words.Add(Ones[digit - '0']);
    }

    /// <summary>
    /// Appends a number from 1100 to 2099 the way a year is read: in pairs (<c>1999</c> «nineteen ninety nine»,
    /// <c>1905</c> «nineteen oh five», <c>1900</c> «nineteen hundred»), and <c>2000</c>–<c>2009</c> as a cardinal.
    /// </summary>
    public static void AppendInPairs(int value, List<string> words)
    {
        var high = value / 100;
        var low = value % 100;
        if (high == 20 && low < 10)
        {
            AppendCardinal(value, words);
            return;
        }

        AppendBelowThousand(high, words);
        if (low == 0)
        {
            words.Add("hundred");
        }
        else
        {
            if (low < 10)
                words.Add("oh");
            AppendBelowThousand(low, words);
        }
    }

    /// <summary>Turns the last word of a cardinal into its ordinal: «twenty one» becomes «twenty first».</summary>
    public static void MakeOrdinal(List<string> words)
    {
        var last = words[^1];
        words[^1] = IrregularOrdinals.TryGetValue(last, out var irregular) ? irregular
            : last.EndsWith('y') ? string.Concat(last.AsSpan(0, last.Length - 1), "ieth")
            : last + "th";
    }

    private static void AppendBelowThousand(int value, List<string> words)
    {
        if (value >= 100)
        {
            words.Add(Ones[value / 100]);
            words.Add("hundred");
            value %= 100;
        }

        if (value >= 20)
        {
            words.Add(Tens[value / 10]);
            value %= 10;
        }

        if (value > 0)
            words.Add(Ones[value]);
    }
}
