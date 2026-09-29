using System.Text;
using System.Text.Json;
using LMSupply.Generator.Abstractions;
using LMSupply.Generator.Models;

namespace LMSupply.Generator.ChatFormatters;

/// <summary>
/// Streaming parser for Qwen/ChatML's native tool-call wrapper:
/// <c>&lt;tool_call&gt;{"name":"...","arguments":{...}}&lt;/tool_call&gt;</c>.
/// </summary>
/// <remarks>
/// State machine mirrors <see cref="Gemma4ToolCallStreamParser"/> (TEXT/BODY buffering with
/// partial-marker holdback), but the wrapper markers and body shape differ: ChatML's body is a
/// single strict-JSON object with <c>name</c> and <c>arguments</c> keys, not Gemma 4's
/// <c>call:NAME{ARGS}</c> custom shape. Registered only as a fallback for the minority of turns
/// where the server's grammar-constrained channel does not produce a delta — see
/// <c>IChatFormatter.SuppressServerToolCallsWhenParserActive</c> and
/// <c>ToolCallStreamCoexistence</c> for how the generator decides which source wins per chunk.
///
/// A grammar-unconstrained Qwen response can leak its native <c>&lt;tool_call&gt;</c> wrapper as
/// plain text instead of invoking through llama-server's structured channel; this parser recovers
/// the call from the text in that case.
/// </remarks>
internal sealed class ChatMLToolCallStreamParser : IToolCallStreamParser
{
    private const string OpenMarker = "<tool_call>";
    private const string CloseMarker = "</tool_call>";
    private const int PreviewMaxLength = 256;

    private readonly StringBuilder _buffer = new();
    private bool _inToolCall;
    private int _toolCallIndex;

    public ToolCallStreamResult Feed(string textDelta)
    {
        if (string.IsNullOrEmpty(textDelta))
        {
            return ToolCallStreamResult.Empty;
        }

        if (ToolCallStreamEventSource.Log.IsEnabled())
        {
            ToolCallStreamEventSource.Log.Feed(_inToolCall ? "BODY" : "TEXT", Truncate(textDelta));
        }

        _buffer.Append(textDelta);
        var result = Drain(flush: false);
        EmitDiagnostics(result);
        return result;
    }

    public ToolCallStreamResult Flush()
    {
        if (ToolCallStreamEventSource.Log.IsEnabled())
        {
            ToolCallStreamEventSource.Log.Flush(_inToolCall ? "BODY" : "TEXT", Truncate(_buffer.ToString()));
        }
        var result = Drain(flush: true);
        EmitDiagnostics(result);
        return result;
    }

    private ToolCallStreamResult Drain(bool flush)
    {
        var emittedText = new StringBuilder();
        List<ChatToolCallDelta>? emittedCalls = null;

        while (_buffer.Length > 0)
        {
            var bufStr = _buffer.ToString();

            if (!_inToolCall)
            {
                var openIdx = bufStr.IndexOf(OpenMarker, StringComparison.Ordinal);
                if (openIdx >= 0)
                {
                    if (openIdx > 0)
                    {
                        emittedText.Append(bufStr, 0, openIdx);
                    }
                    _buffer.Clear();
                    var remainStart = openIdx + OpenMarker.Length;
                    if (remainStart < bufStr.Length)
                    {
                        _buffer.Append(bufStr, remainStart, bufStr.Length - remainStart);
                    }
                    _inToolCall = true;
                    continue;
                }

                if (flush)
                {
                    emittedText.Append(bufStr);
                    _buffer.Clear();
                    break;
                }

                var holdLen = LongestPrefixMatchAtEnd(bufStr, OpenMarker);
                if (holdLen > 0 && holdLen < bufStr.Length)
                {
                    emittedText.Append(bufStr, 0, bufStr.Length - holdLen);
                    _buffer.Clear();
                    _buffer.Append(bufStr, bufStr.Length - holdLen, holdLen);
                }
                else if (holdLen == 0)
                {
                    emittedText.Append(bufStr);
                    _buffer.Clear();
                }
                break;
            }
            else
            {
                var closeIdx = bufStr.IndexOf(CloseMarker, StringComparison.Ordinal);
                if (closeIdx >= 0)
                {
                    var body = bufStr.Substring(0, closeIdx);
                    var delta = TryParseToolCallBody(body, _toolCallIndex);
                    if (delta is not null)
                    {
                        (emittedCalls ??= new List<ChatToolCallDelta>()).Add(delta);
                        _toolCallIndex++;
                    }

                    _buffer.Clear();
                    var remainStart = closeIdx + CloseMarker.Length;
                    if (remainStart < bufStr.Length)
                    {
                        _buffer.Append(bufStr, remainStart, bufStr.Length - remainStart);
                    }
                    _inToolCall = false;
                    continue;
                }

                if (flush)
                {
                    _buffer.Clear();
                    _inToolCall = false;
                    break;
                }

                break;
            }
        }

        if (emittedText.Length == 0 && emittedCalls is null)
        {
            return ToolCallStreamResult.Empty;
        }

        return new ToolCallStreamResult
        {
            Text = emittedText.Length > 0 ? emittedText.ToString() : null,
            ToolCalls = emittedCalls
        };
    }

    private static int LongestPrefixMatchAtEnd(string text, string marker)
    {
        var maxLen = Math.Min(text.Length, marker.Length - 1);
        for (var len = maxLen; len > 0; len--)
        {
            if (marker.AsSpan(0, len).SequenceEqual(text.AsSpan(text.Length - len, len)))
            {
                return len;
            }
        }
        return 0;
    }

    private static ChatToolCallDelta? TryParseToolCallBody(string body, int index)
    {
        if (string.IsNullOrWhiteSpace(body))
        {
            return null;
        }

        var (name, argumentsJson) = TryParseNameAndArguments(body.Trim());
        if (name is null)
        {
            if (ToolCallStreamEventSource.Log.IsEnabled())
            {
                ToolCallStreamEventSource.Log.BodyParse("rejected", Truncate(body));
            }
            return null;
        }

        if (ToolCallStreamEventSource.Log.IsEnabled())
        {
            ToolCallStreamEventSource.Log.BodyParse("accepted", Truncate(body));
        }

        return new ChatToolCallDelta
        {
            Index = index,
            Id = $"call_cm_{Guid.NewGuid():N}"[..24],
            Name = name,
            Arguments = argumentsJson
        };
    }

    private static (string? Name, string? ArgumentsJson) TryParseNameAndArguments(string body)
    {
        var name = TryExtract(body, out var argumentsJson);
        if (name is not null)
        {
            return (name, argumentsJson);
        }

        var relaxed = RelaxedJsonNormalizer.Normalize(body);
        return relaxed is null ? (null, null) : (TryExtract(relaxed, out argumentsJson), argumentsJson);
    }

    private static string? TryExtract(string json, out string? argumentsJson)
    {
        argumentsJson = null;
        try
        {
            using var doc = JsonDocument.Parse(json);
            if (doc.RootElement.ValueKind != JsonValueKind.Object)
            {
                return null;
            }

            if (!doc.RootElement.TryGetProperty("name", out var nameElement) ||
                nameElement.ValueKind != JsonValueKind.String)
            {
                return null;
            }

            var name = nameElement.GetString();
            if (string.IsNullOrEmpty(name))
            {
                return null;
            }

            argumentsJson = doc.RootElement.TryGetProperty("arguments", out var argsElement)
                ? argsElement.GetRawText()
                : "{}";

            return name;
        }
        catch (JsonException)
        {
            return null;
        }
    }

    private static void EmitDiagnostics(ToolCallStreamResult result)
    {
        if (!ToolCallStreamEventSource.Log.IsEnabled())
        {
            return;
        }

        if (!string.IsNullOrEmpty(result.Text))
        {
            ToolCallStreamEventSource.Log.EmitText(Truncate(result.Text!));
        }

        if (result.ToolCalls is { Count: > 0 } calls)
        {
            foreach (var call in calls)
            {
                ToolCallStreamEventSource.Log.EmitToolCall(
                    call.Name ?? string.Empty,
                    Truncate(call.Arguments ?? string.Empty));
            }
        }
    }

    private static string Truncate(string value)
    {
        if (value.Length <= PreviewMaxLength)
        {
            return value;
        }
        return string.Concat(value.AsSpan(0, PreviewMaxLength), "...");
    }
}
