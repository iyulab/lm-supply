using System.Text.Json;
using System.Text.RegularExpressions;
using LMSupply.Generator.Models;

namespace LMSupply.Generator.Internal;

/// <summary>
/// Parses tool call JSON from ONNX model text output.
/// ONNX models output raw text, so tool calls must be detected and parsed from the response.
/// </summary>
/// <remarks>
/// Supports two formats:
/// <list type="number">
/// <item>
/// OpenAI-style tool_calls array:
/// <code>{"tool_calls": [{"id": "call_123", "type": "function", "function": {"name": "fn", "arguments": "{...}"}}]}</code>
/// </item>
/// <item>
/// Direct function call (single tool):
/// <code>{"name": "fn", "arguments": {"key": "value"}}</code>
/// </item>
/// </list>
/// </remarks>
internal static class ToolCallTextParser
{
    /// <summary>
    /// Attempts to parse tool calls from model output text.
    /// Returns null if the text does not contain recognizable tool call JSON.
    /// </summary>
    /// <param name="text">The raw text output from the model.</param>
    /// <returns>A list of parsed tool calls, or null if no tool calls were found.</returns>
    public static IReadOnlyList<ChatToolCall>? TryParse(string? text)
    {
        if (string.IsNullOrWhiteSpace(text))
            return null;

        var trimmed = text.Trim();
        if (!trimmed.StartsWith('{'))
            return null;

        // Phi-4-mini and similar small ONNX models occasionally emit trailing
        // garbage tokens (extra '}' or stray whitespace) after the closing brace
        // of the tool-call object — strict JsonDocument.Parse rejects those.
        // Walk the string honoring quoted strings + escapes, grab the substring
        // that ends at the first balanced top-level closing brace, and parse that.
        var candidate = TryExtractBalancedJsonObject(trimmed) ?? trimmed;

        try
        {
            using var doc = JsonDocument.Parse(candidate);
            var root = doc.RootElement;

            // Format 1: {"tool_calls": [...]}
            if (root.TryGetProperty("tool_calls", out var toolCallsElement)
                && toolCallsElement.ValueKind == JsonValueKind.Array)
            {
                return ParseToolCallsArray(toolCallsElement);
            }

            // Format 2: {"name": "fn", "arguments": {...}}
            if (root.TryGetProperty("name", out var nameElement)
                && nameElement.ValueKind == JsonValueKind.String
                && root.TryGetProperty("arguments", out var argsElement))
            {
                return ParseDirectFunctionCall(nameElement, argsElement);
            }

            return null;
        }
        catch (JsonException)
        {
            // Strict + recovery passes both failed. Phi-4-mini occasionally emits
            // the right semantic content (function name + arguments) wrapped in
            // structurally invalid JSON — e.g. mixing `]` / `}` closers or padding
            // an extra `]]` instead of the outer `}` (Sprint-RR1 RR-M evidence,
            // 2026-05-02). Fall through to a tolerant regex-based extractor that
            // surfaces the legitimate tool call instead of dropping it.
            return TryExtractByRegex(trimmed);
        }
    }

    // Tolerant fallback for malformed envelopes: locate `"name": "..."` and the
    // adjacent `"arguments":` value (string literal or object literal) and emit
    // the corresponding ChatToolCall directly, bypassing the strict structural
    // walk. Used only after the strict pass has already rejected the input.
    private static IReadOnlyList<ChatToolCall>? TryExtractByRegex(string text)
    {
        var nameMatch = Regex.Match(text, """"name"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"""", RegexOptions.Singleline);
        if (!nameMatch.Success)
            return null;
        var name = nameMatch.Groups[1].Value;
        if (string.IsNullOrEmpty(name))
            return null;

        var argsStart = text.IndexOf("\"arguments\"", nameMatch.Index, StringComparison.Ordinal);
        string arguments = "{}";
        if (argsStart >= 0)
        {
            // Skip past `"arguments"`, optional whitespace, and the colon.
            var cursor = argsStart + "\"arguments\"".Length;
            while (cursor < text.Length && (text[cursor] == ' ' || text[cursor] == '\t' || text[cursor] == ':'))
                cursor++;
            if (cursor < text.Length)
            {
                if (text[cursor] == '{')
                {
                    var span = ExtractBalancedSpan(text, cursor, '{', '}');
                    if (span is not null)
                        arguments = span;
                }
                else if (text[cursor] == '"')
                {
                    var span = ExtractStringLiteral(text, cursor);
                    if (span is not null)
                    {
                        // span holds the raw inter-quote characters with JSON
                        // escapes intact (e.g. `{\"query\":\"x\"}`). Decode by
                        // re-wrapping in quotes and asking the JSON serializer
                        // to interpret the escape sequences; the strict path's
                        // SerializeArguments returns the same decoded form.
                        arguments = JsonDecodeStringValue(span);
                    }
                }
            }
        }

        var idMatch = Regex.Match(text, """"id"\s*:\s*"([^"]+)"""", RegexOptions.Singleline);
        var id = idMatch.Success ? idMatch.Groups[1].Value : GenerateCallId();

        return [new ChatToolCall(id, name, arguments)];
    }

    private static string? ExtractBalancedSpan(string text, int start, char open, char close)
    {
        if (start >= text.Length || text[start] != open) return null;
        var depth = 0;
        var inString = false;
        var escapeNext = false;
        for (var i = start; i < text.Length; i++)
        {
            var c = text[i];
            if (escapeNext) { escapeNext = false; continue; }
            if (inString)
            {
                if (c == '\\') escapeNext = true;
                else if (c == '"') inString = false;
                continue;
            }
            if (c == '"') { inString = true; continue; }
            if (c == open) depth++;
            else if (c == close)
            {
                depth--;
                if (depth == 0) return text[start..(i + 1)];
            }
        }
        return null;
    }

    private static string? ExtractStringLiteral(string text, int start)
    {
        if (start >= text.Length || text[start] != '"') return null;
        var escapeNext = false;
        for (var i = start + 1; i < text.Length; i++)
        {
            var c = text[i];
            if (escapeNext) { escapeNext = false; continue; }
            if (c == '\\') { escapeNext = true; continue; }
            if (c == '"') return text[(start + 1)..i];
        }
        return null;
    }

    private static string JsonDecodeStringValue(string rawWithEscapes)
    {
        try
        {
            // JsonDocument, not JsonSerializer.Deserialize<string>: no reflection metadata needed (trimmed/AOT hosts).
            using var doc = JsonDocument.Parse("\"" + rawWithEscapes + "\"");
            return doc.RootElement.GetString() ?? "{}";
        }
        catch (JsonException)
        {
            return "{}";
        }
    }

    /// <summary>
    /// Returns the substring of <paramref name="text"/> that ends at the first
    /// balanced top-level closing brace, or <c>null</c> if no balanced object
    /// is present. Honours quoted strings and standard JSON escapes so that
    /// braces inside string literals are not counted.
    /// </summary>
    /// <remarks>
    /// Phi-4-mini and similar small instruction-tuned ONNX models occasionally
    /// emit a tool-call envelope that ends with one extra <c>\"</c> escape
    /// sequence before the legitimate closing quote of an inner string value
    /// (Sprint-RR1 RR-K evidence, 2026-05-02 — Kafka prompt produced
    /// <c>"arguments": "{...}\""</c> instead of the canonical
    /// <c>"arguments": "{...}"</c>). When the strict pass leaves the parser stuck
    /// inside an unterminated string, this helper performs a best-effort
    /// recovery: scan from end-of-input for the last unescaped quote candidate,
    /// remove the immediately-preceding stray <c>\</c>, and re-run the strict
    /// pass on the cleaned input. The recovery is bounded — it tries at most
    /// once and bails out if the cleaned variant also fails — so a genuinely
    /// truncated payload still returns <c>null</c>.
    /// </remarks>
    private static string? TryExtractBalancedJsonObject(string text)
    {
        if (text.Length == 0 || text[0] != '{')
            return null;

        var strict = TryWalkBalanced(text);
        if (strict is not null)
            return strict;

        // Recovery 1: stray <c>\</c> before the closing quote of an inner string
        // value (Phi-4-mini RR-K pattern). Remove the first such escape from the
        // tail forward and retry once.
        for (var idx = text.Length - 1; idx > 1; idx--)
        {
            if (text[idx] != '"' || text[idx - 1] != '\\')
                continue;

            var rebuilt = string.Concat(text.AsSpan(0, idx - 1), text.AsSpan(idx));
            var recovered = TryWalkBalanced(rebuilt);
            if (recovered is not null)
                return recovered;

            break;
        }

        // Recovery 2: missing trailing <c>}</c> closures (Phi-4-mini RR-M
        // pattern — the model emitted <c>...]]</c> without closing the outer
        // object). Pad the input with up to a small number of synthetic
        // <c>}</c> closes and re-walk; bail out if even the padded variant
        // fails so genuinely truncated payloads still return null.
        const int MaxSyntheticBraces = 6;
        for (var pad = 1; pad <= MaxSyntheticBraces; pad++)
        {
            var padded = text + new string('}', pad);
            var recovered = TryWalkBalanced(padded);
            if (recovered is not null)
                return recovered;
        }

        return null;
    }

    private static string? TryWalkBalanced(string text)
    {
        var depth = 0;
        var inString = false;
        var escapeNext = false;

        for (var i = 0; i < text.Length; i++)
        {
            var c = text[i];

            if (escapeNext)
            {
                escapeNext = false;
                continue;
            }

            if (inString)
            {
                if (c == '\\')
                {
                    escapeNext = true;
                }
                else if (c == '"')
                {
                    inString = false;
                }
                continue;
            }

            switch (c)
            {
                case '"':
                    inString = true;
                    break;
                case '{':
                    depth++;
                    break;
                case '}':
                    depth--;
                    if (depth == 0)
                    {
                        return text[..(i + 1)];
                    }
                    break;
            }
        }

        return null;
    }

    private static List<ChatToolCall>? ParseToolCallsArray(JsonElement toolCallsElement)
    {
        var calls = new List<ChatToolCall>();

        foreach (var item in toolCallsElement.EnumerateArray())
        {
            if (!item.TryGetProperty("function", out var fn))
                continue;

            var id = item.TryGetProperty("id", out var idEl) && idEl.ValueKind == JsonValueKind.String
                ? idEl.GetString() ?? GenerateCallId()
                : GenerateCallId();

            var name = fn.TryGetProperty("name", out var nm) && nm.ValueKind == JsonValueKind.String
                ? nm.GetString() ?? string.Empty
                : string.Empty;

            var arguments = fn.TryGetProperty("arguments", out var args)
                ? SerializeArguments(args)
                : "{}";

            if (string.IsNullOrEmpty(name))
                continue;

            calls.Add(new ChatToolCall(id, name, arguments));
        }

        return calls.Count > 0 ? calls : null;
    }

    private static List<ChatToolCall>? ParseDirectFunctionCall(
        JsonElement nameElement, JsonElement argsElement)
    {
        var name = nameElement.GetString();
        if (string.IsNullOrEmpty(name))
            return null;

        var arguments = SerializeArguments(argsElement);

        return [new ChatToolCall(GenerateCallId(), name, arguments)];
    }

    /// <summary>
    /// Serializes arguments to a JSON string.
    /// If the arguments are already a string, returns them directly.
    /// If they are an object/array, serializes to JSON.
    /// </summary>
    private static string SerializeArguments(JsonElement args)
    {
        return args.ValueKind switch
        {
            JsonValueKind.String => args.GetString() ?? "{}",
            JsonValueKind.Object or JsonValueKind.Array => args.GetRawText(),
            _ => "{}"
        };
    }

    private static string GenerateCallId()
    {
        return $"call_{Guid.NewGuid():N}"[..24];
    }
}
