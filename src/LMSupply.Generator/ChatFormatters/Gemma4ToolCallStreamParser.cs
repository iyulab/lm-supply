using System.Text;
using System.Text.Json;
using LMSupply.Generator.Abstractions;
using LMSupply.Generator.Models;

namespace LMSupply.Generator.ChatFormatters;

/// <summary>
/// Streaming parser for Gemma 4's native tool-call wrapper:
/// <c>&lt;|tool_call&gt;call:NAME{ARGS_JSON}&lt;tool_call|&gt;</c>.
/// </summary>
/// <remarks>
/// State machine:
/// <list type="bullet">
/// <item>TEXT mode (default): forward incoming deltas to the text channel; on
/// detecting <c>&lt;|tool_call&gt;</c> flush text-before-marker and switch to BODY mode.</item>
/// <item>BODY mode: buffer incoming deltas without emitting any text; on detecting
/// <c>&lt;tool_call|&gt;</c> parse the body into a <see cref="ChatToolCallDelta"/>
/// (name + JSON arguments) and switch back to TEXT mode.</item>
/// </list>
/// Partial wrapper prefixes at the tail of a delta are retained in the buffer so
/// that wrappers split across deltas are recognized correctly. Incomplete bodies
/// (closing token never arrives) are discarded on flush — emitting a half-formed
/// delta would leak a non-invokable tool call upstream.
///
/// Bodies are parsed leniently to accept Gemma 4's native JS-object-literal shape
/// (<c>{key:"value"}</c> — unquoted identifier keys, single-quoted strings,
/// trailing commas) by routing through <see cref="RelaxedJsonNormalizer"/>
/// while still rejecting non-object roots so phantom invocations stay impossible.
/// An opt-in <see cref="ToolCallStreamEventSource"/> stream-tap lets parser correctness be
/// validated independently of model emit behavior under <c>dotnet-trace</c>; it adds no
/// overhead when no listener is attached.
/// </remarks>
internal sealed class Gemma4ToolCallStreamParser : IToolCallStreamParser
{
    private const string OpenMarker = "<|tool_call>";
    private const string CloseMarker = "<tool_call|>";
    private const string CallPrefix = "call:";
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

        var trimmed = body.TrimStart();
        if (!trimmed.StartsWith(CallPrefix, StringComparison.Ordinal))
        {
            return null;
        }

        var afterCall = trimmed.AsSpan(CallPrefix.Length);
        var braceIdx = afterCall.IndexOf('{');
        if (braceIdx < 0)
        {
            return null;
        }

        var name = afterCall[..braceIdx].Trim().ToString();
        if (string.IsNullOrEmpty(name))
        {
            return null;
        }

        var argsRaw = afterCall[braceIdx..].ToString().TrimEnd();
        var argsJson = TryNormalizeArgsBody(argsRaw);
        if (argsJson is null)
        {
            if (ToolCallStreamEventSource.Log.IsEnabled())
            {
                ToolCallStreamEventSource.Log.BodyParse("rejected", Truncate(argsRaw));
            }
            return null;
        }

        if (ToolCallStreamEventSource.Log.IsEnabled())
        {
            ToolCallStreamEventSource.Log.BodyParse("accepted", Truncate(argsRaw));
        }

        return new ChatToolCallDelta
        {
            Index = index,
            Id = $"call_g4_{Guid.NewGuid():N}"[..24],
            Name = name,
            Arguments = argsJson
        };
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

    private static string? TryNormalizeArgsBody(string argsRaw)
    {
        if (TryParseStrict(argsRaw) is { } strict)
        {
            return strict;
        }

        var relaxed = RelaxedJsonNormalizer.Normalize(argsRaw);
        if (relaxed is null)
        {
            return null;
        }

        return TryParseStrict(relaxed);
    }

    private static string? TryParseStrict(string json)
    {
        try
        {
            using var doc = JsonDocument.Parse(json);
            if (doc.RootElement.ValueKind != JsonValueKind.Object)
            {
                return null;
            }
            return doc.RootElement.GetRawText();
        }
        catch (JsonException)
        {
            return null;
        }
    }
}
