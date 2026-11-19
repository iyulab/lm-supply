namespace LMSupply.Ocr.Models;

/// <summary>
/// Turns a language code as callers and operating systems write it into the codes the recognition models are
/// registered under. A code is a BCP 47 tag (<c>ko-KR</c>, <c>sr-Latn-RS</c>, <c>zh-Hant-TW</c>), but a POSIX
/// locale name (<c>ko_KR.UTF-8</c>) arrives just as often; both mean the same language.
/// </summary>
internal static class OcrLanguageCode
{
    // Codes that name one language this library reads under another registered code. A code with two readings
    // (the legacy "sh" is Serbo-Croatian, "mo" is Moldavian) is not listed - it has no single answer, so it is
    // refused like any code without a model.
    private static readonly Dictionary<string, string> Canonical = new(StringComparer.OrdinalIgnoreCase)
    {
        ["cmn"] = "zh", // Mandarin
        ["yue"] = "zh", // Cantonese is written with the same characters
        ["iw"] = "he",  // the former code of Hebrew
    };

    /// <summary>
    /// The registry keys to try for a code, most specific first: the code as written, then language-script-region,
    /// language-script, language-region and the bare language. The script subtag comes before the region because
    /// it decides what is printed (<c>sr-Latn</c> is Latin letters, <c>sr</c> Cyrillic) and the region does not.
    /// </summary>
    public static IReadOnlyList<string> Candidates(string languageCode) => Parse(languageCode).Candidates;

    /// <summary>
    /// The script the code names, if it names one: a four-letter script subtag (<c>sr-Latn</c>) or a POSIX modifier
    /// (<c>sr_RS@latin</c>); <see langword="null"/> when the code leaves the script to its language.
    /// </summary>
    public static string? ScriptOf(string languageCode) => Parse(languageCode).Script;

    // The scripts each built-in recognizer reads (ISO 15924). A model that is not listed - one a caller registered -
    // is not judged: its dictionary is unknown here.
    private static readonly Dictionary<string, string[]> RecognizerScripts = new(StringComparer.OrdinalIgnoreCase)
    {
        ["crnn-en-v5"] = ["Latn"],
        ["crnn-latin-v5"] = ["Latn"],
        ["crnn-korean-v5"] = ["Kore", "Hang"],
        ["crnn-chinese-v5"] = ["Hani", "Hans", "Hant", "Jpan", "Hira", "Kana"],
        ["crnn-arabic-v5"] = ["Arab"],
        ["crnn-cyrillic-v5"] = ["Cyrl"],
        ["crnn-devanagari-v5"] = ["Deva"],
        ["crnn-greek-v5"] = ["Grek"],
        ["crnn-thai-v5"] = ["Thai"],
        ["crnn-tamil-v5"] = ["Taml"],
        ["crnn-telugu-v5"] = ["Telu"],
    };

    /// <summary>
    /// Whether the recognizer reads <paramref name="script"/>. Answers <see langword="true"/> for a recognizer it knows
    /// nothing about, so a caller's own model is never refused on a guess.
    /// </summary>
    public static bool ReadsScript(string recognizerAlias, string script)
        => !RecognizerScripts.TryGetValue(recognizerAlias, out var scripts)
           || scripts.Contains(script, StringComparer.OrdinalIgnoreCase);

    /// <summary>Whether a registry key names the script itself (<c>sr-latn</c>), as opposed to leaving it to the language.</summary>
    public static bool NamesScript(string candidate, string script)
        => candidate.Split('-').Skip(1).Any(part => part.Equals(script, StringComparison.OrdinalIgnoreCase));

    private static (IReadOnlyList<string> Candidates, string? Script) Parse(string languageCode)
    {
        var written = languageCode.Trim();
        var normalized = written;

        // A POSIX locale name carries an encoding or a modifier after the tag: ko_KR.UTF-8, sr_RS@latin. The modifier
        // "latin" or "cyrillic" names the script, and the script is what picks the recognizer.
        string? modifierScript = null;
        var at = normalized.IndexOf('@');
        if (at >= 0)
        {
            modifierScript = normalized[(at + 1)..].ToLowerInvariant() switch
            {
                "latin" => "Latn",
                "cyrillic" => "Cyrl",
                _ => null,
            };
        }

        var cut = normalized.IndexOfAny(['.', '@']);
        if (cut >= 0)
            normalized = normalized[..cut];

        normalized = normalized.Replace('_', '-');

        var candidates = new List<string>();
        void Add(string candidate)
        {
            if (candidate.Length > 0 && !candidates.Contains(candidate, StringComparer.OrdinalIgnoreCase))
                candidates.Add(candidate);
        }

        // As written first: a registered alias or an id such as "crnn-korean-v5" (or a repository id or path,
        // whose dots and underscores are part of the name) resolves as itself.
        Add(written);
        Add(normalized);

        var subtags = normalized.Split('-', StringSplitOptions.RemoveEmptyEntries);
        if (subtags.Length == 0)
            return (candidates, null);

        var language = Canonical.TryGetValue(subtags[0], out var canonical) ? canonical : subtags[0];

        string? script = null;
        string? region = null;
        foreach (var subtag in subtags.Skip(1))
        {
            if (script is null && region is null && subtag.Length == 4 && subtag.All(char.IsAsciiLetter))
                script = subtag;
            else if (region is null && ((subtag.Length == 2 && subtag.All(char.IsAsciiLetter))
                                        || (subtag.Length == 3 && subtag.All(char.IsAsciiDigit))))
                region = subtag;
        }

        script ??= modifierScript;

        if (script is not null && region is not null)
            Add($"{language}-{script}-{region}");
        if (script is not null)
            Add($"{language}-{script}");
        if (region is not null)
            Add($"{language}-{region}");
        Add(language);

        return (candidates, script);
    }
}
