using System.Text;

namespace LMSupply.Generator.ChatFormatters;

/// <summary>
/// Converts JS-object-literal style payloads into strict JSON so
/// <see cref="System.Text.Json.JsonDocument"/> can parse them.
/// </summary>
/// <remarks>
/// Gemma 4's native chat template emits tool-call bodies in
/// <c>{key:"value", n:1}</c> shape — unquoted identifier keys,
/// double- or single-quoted string values, and occasional trailing commas
/// before <c>]</c> / <c>}</c>. This is valid JavaScript / JSON5 but rejected
/// by <see cref="System.Text.Json"/>. The normalizer walks the input as a
/// minimal lexer that respects string boundaries, then rewrites:
/// <list type="bullet">
/// <item>bare identifier keys (in object key position) → double-quoted keys</item>
/// <item>single-quoted string literals → double-quoted with proper escaping</item>
/// <item>trailing commas before <c>}</c> or <c>]</c> → removed</item>
/// </list>
/// String contents are copied verbatim (escapes preserved) so colons, braces,
/// and quotes inside string values are never confused with structural tokens.
/// The output is still passed through <see cref="System.Text.Json.JsonDocument"/>
/// for final validation — the normalizer is "best-effort rewrite", not a parser.
///
/// Gemma 4 emits tool-call bodies such as <c>{query:"…"}</c> that strict
/// <c>JsonDocument.Parse</c> rejects; without this rewrite every such tool call would be dropped.
/// </remarks>
internal static class RelaxedJsonNormalizer
{
    public static string? Normalize(string input)
    {
        if (string.IsNullOrEmpty(input))
        {
            return null;
        }

        var sb = new StringBuilder(input.Length + 16);
        var i = 0;
        while (i < input.Length)
        {
            var c = input[i];

            if (c == '"')
            {
                CopyDoubleQuotedString(input, sb, ref i);
                continue;
            }

            if (c == '\'')
            {
                if (!RewriteSingleQuotedString(input, sb, ref i))
                {
                    return null;
                }
                continue;
            }

            if (IsIdentifierStart(c) && IsAtKeyPosition(sb))
            {
                var start = i;
                while (i < input.Length && IsIdentifierContinue(input[i]))
                {
                    i++;
                }
                var ident = input.AsSpan(start, i - start);

                var lookahead = i;
                while (lookahead < input.Length && char.IsWhiteSpace(input[lookahead]))
                {
                    lookahead++;
                }

                if (lookahead < input.Length && input[lookahead] == ':')
                {
                    sb.Append('"').Append(ident).Append('"');
                }
                else
                {
                    sb.Append(ident);
                }
                continue;
            }

            if (c == ',')
            {
                var k = i + 1;
                while (k < input.Length && char.IsWhiteSpace(input[k]))
                {
                    k++;
                }
                if (k < input.Length && (input[k] == '}' || input[k] == ']'))
                {
                    i = k;
                    continue;
                }
            }

            sb.Append(c);
            i++;
        }

        return sb.Length == 0 ? null : sb.ToString();
    }

    private static void CopyDoubleQuotedString(string input, StringBuilder sb, ref int i)
    {
        sb.Append(input[i]);
        i++;
        while (i < input.Length)
        {
            var d = input[i];
            sb.Append(d);
            i++;
            if (d == '\\' && i < input.Length)
            {
                sb.Append(input[i]);
                i++;
                continue;
            }
            if (d == '"')
            {
                return;
            }
        }
    }

    private static bool RewriteSingleQuotedString(string input, StringBuilder sb, ref int i)
    {
        sb.Append('"');
        i++;
        while (i < input.Length)
        {
            var d = input[i];

            if (d == '\\' && i + 1 < input.Length)
            {
                var next = input[i + 1];
                if (next == '\'')
                {
                    sb.Append('\'');
                }
                else
                {
                    sb.Append('\\').Append(next);
                }
                i += 2;
                continue;
            }

            if (d == '\'')
            {
                sb.Append('"');
                i++;
                return true;
            }

            if (d == '"')
            {
                sb.Append('\\').Append('"');
                i++;
                continue;
            }

            sb.Append(d);
            i++;
        }
        return false;
    }

    private static bool IsAtKeyPosition(StringBuilder emitted)
    {
        for (var idx = emitted.Length - 1; idx >= 0; idx--)
        {
            var ch = emitted[idx];
            if (char.IsWhiteSpace(ch))
            {
                continue;
            }
            return ch == '{' || ch == ',';
        }
        return false;
    }

    private static bool IsIdentifierStart(char c)
        => c == '_' || c == '$' || char.IsLetter(c);

    private static bool IsIdentifierContinue(char c)
        => c == '_' || c == '$' || char.IsLetterOrDigit(c);
}
