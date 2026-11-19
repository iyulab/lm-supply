using System.Globalization;
using System.Text;

namespace LMSupply.Ocr.Recognition;

/// <summary>
/// Reads the character list of a PaddleOCR <c>inference.yml</c> — the file that ships beside an official PP-OCR
/// recognition model and holds its dictionary as <c>PostProcess.character_dict</c>, a block list with one character per entry.
/// Only what that list uses is read (plain, single-quoted and double-quoted scalars); anything else is refused
/// rather than guessed at, because a misread dictionary reads text wrongly without failing.
/// </summary>
internal static class PaddleInferenceConfig
{
    private const string Key = "character_dict:";

    /// <summary>Reads the <c>character_dict</c> entries of an <c>inference.yml</c>.</summary>
    /// <exception cref="InvalidDataException">The file has no <c>character_dict</c> list, or an entry uses syntax this reader does not support.</exception>
    public static List<string> ReadCharacterDictionary(string path)
    {
        var lines = File.ReadAllLines(path);
        var start = -1;
        var keyIndent = 0;
        for (var i = 0; i < lines.Length; i++)
        {
            var trimmed = lines[i].TrimStart();
            if (trimmed.TrimEnd() == Key)
            {
                start = i + 1;
                keyIndent = lines[i].Length - trimmed.Length;
                break;
            }
        }

        if (start < 0)
            throw new InvalidDataException($"'{Path.GetFileName(path)}' has no 'character_dict' list.");

        var entries = new List<string>();
        for (var i = start; i < lines.Length; i++)
        {
            var line = lines[i].TrimEnd('\r');
            var trimmed = line.TrimStart(' ');
            var indent = line.Length - trimmed.Length;

            // A blank line or a comment line between entries is part of the list, not the end of it.
            if (trimmed.Length == 0 || trimmed[0] == '#')
                continue;

            // A block list may sit at the key's own indentation; the first line that is not an entry of it ends the list.
            if (indent < keyIndent || !trimmed.StartsWith("- ", StringComparison.Ordinal))
                break;

            entries.Add(ParseScalar(trimmed[2..], i + 1, path));
        }

        if (entries.Count == 0)
            throw new InvalidDataException($"'{Path.GetFileName(path)}' has an empty 'character_dict' list.");

        return entries;
    }

    private static string ParseScalar(string text, int lineNumber, string path)
    {
        if (text.Length == 0)
            throw Unsupported(lineNumber, path, "an empty entry");

        // NEL, LS and PS are line breaks in YAML, which folds them inside a quoted scalar; this reader does not.
        if (text.AsSpan().IndexOfAny('\u0085', '\u2028', '\u2029') >= 0)
            throw Unsupported(lineNumber, path, "a raw NEL, LS or PS line separator");

        switch (text[0])
        {
            case '\'':
                return NotEmpty(ParseSingleQuoted(text, lineNumber, path), lineNumber, path);
            case '"':
                return NotEmpty(ParseDoubleQuoted(text, lineNumber, path), lineNumber, path);
            case '|' or '>' or '&' or '*' or '!' or '[' or '{' or '#' or '%' or '@' or '`':
                throw Unsupported(lineNumber, path, $"an entry starting with '{text[0]}'");
            default:
                // A plain scalar runs to the end of the line, minus trailing spaces; one that holds " #" or ": " is not a bare
                // character, and null, ~ and the booleans are not text (a writer quotes a string that reads as one).
                text = text.TrimEnd(' ');
                if (text.Contains(" #", StringComparison.Ordinal) || text.Contains(": ", StringComparison.Ordinal))
                    throw Unsupported(lineNumber, path, "a plain entry with a comment or a mapping");
                if (text is "~" or "null" or "Null" or "NULL" or "true" or "True" or "TRUE" or "false" or "False" or "FALSE")
                    throw Unsupported(lineNumber, path, $"the plain entry '{text}', which YAML reads as a non-text value");
                return text;
        }
    }

    // An entry is a character; an empty one would be skipped by the dictionary and shift every class after it.
    private static string NotEmpty(string value, int lineNumber, string path) =>
        value.Length > 0 ? value : throw Unsupported(lineNumber, path, "an empty entry");

    private static string ParseSingleQuoted(string text, int lineNumber, string path)
    {
        var sb = new StringBuilder();
        for (var i = 1; i < text.Length; i++)
        {
            if (text[i] != '\'')
            {
                sb.Append(text[i]);
                continue;
            }

            // '' is an escaped quote; a lone ' closes the scalar, and nothing may follow it.
            if (i + 1 < text.Length && text[i + 1] == '\'')
            {
                sb.Append('\'');
                i++;
                continue;
            }

            if (i + 1 != text.Length)
                throw Unsupported(lineNumber, path, "text after a closing quote");
            return sb.ToString();
        }

        throw Unsupported(lineNumber, path, "an unterminated single-quoted entry");
    }

    private static string ParseDoubleQuoted(string text, int lineNumber, string path)
    {
        var sb = new StringBuilder();
        for (var i = 1; i < text.Length; i++)
        {
            var c = text[i];
            if (c == '"')
            {
                if (i + 1 != text.Length)
                    throw Unsupported(lineNumber, path, "text after a closing quote");
                return sb.ToString();
            }

            if (c != '\\')
            {
                sb.Append(c);
                continue;
            }

            if (++i >= text.Length)
                break;

            switch (text[i])
            {
                case '\\': sb.Append('\\'); break;
                case '"': sb.Append('"'); break;
                case '/': sb.Append('/'); break;
                case '0': sb.Append('\0'); break;
                case 'n': sb.Append('\n'); break;
                case 'r': sb.Append('\r'); break;
                case 't': sb.Append('\t'); break;
                case ' ': sb.Append(' '); break;
                case 'a': sb.Append('\a'); break;
                case 'b': sb.Append('\b'); break;
                case 'e': sb.Append('\u001b'); break;
                case 'f': sb.Append('\f'); break;
                case 'v': sb.Append('\v'); break;
                case '_': sb.Append('\u00a0'); break;
                case 'N': sb.Append('\u0085'); break;
                case 'L': sb.Append('\u2028'); break;
                case 'P': sb.Append('\u2029'); break;
                case 'x': i = AppendCodePoint(sb, text, i, 2, lineNumber, path); break;
                case 'u': i = AppendCodePoint(sb, text, i, 4, lineNumber, path); break;
                case 'U': i = AppendCodePoint(sb, text, i, 8, lineNumber, path); break;
                default:
                    throw Unsupported(lineNumber, path, $"the escape '\\{text[i]}'");
            }
        }

        throw Unsupported(lineNumber, path, "an unterminated double-quoted entry");
    }

    private static int AppendCodePoint(StringBuilder sb, string text, int escapeIndex, int digits, int lineNumber, string path)
    {
        var from = escapeIndex + 1;
        if (from + digits > text.Length
            || !int.TryParse(text.AsSpan(from, digits), NumberStyles.AllowHexSpecifier, CultureInfo.InvariantCulture, out var value)
            || !Rune.IsValid(value))
            throw Unsupported(lineNumber, path, "a malformed hexadecimal escape");

        sb.Append(char.ConvertFromUtf32(value));
        return from + digits - 1;
    }

    private static InvalidDataException Unsupported(int lineNumber, string path, string what) =>
        new($"'{Path.GetFileName(path)}' line {lineNumber}: {what} is not supported in a character_dict entry.");
}
