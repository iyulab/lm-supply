using LMSupply.Generator.Abstractions;
using LMSupply.Generator.Models;

namespace LMSupply.Generator.ChatFormatters;

/// <summary>
/// Resolves, per streaming chunk, which tool-call source is authoritative when a formatter's
/// <see cref="IChatFormatter.CreateToolCallStreamParser"/> parser coexists with llama-server's own
/// grammar-constrained tool-call channel (<see cref="IChatFormatter.SuppressServerToolCallsWhenParserActive"/>
/// == <c>false</c>).
/// </summary>
/// <remarks>
/// Unlike Gemma 4 (whose grammar channel never produces a usable delta, so the parser can own the
/// channel unconditionally), a coexisting formatter's grammar channel usually works — the parser
/// exists only to catch the model's rarer native-wrapper fallback. Server deltas therefore win
/// whenever present for a chunk; the parser is fed regardless (so any incidental wrapper text is
/// still stripped from the text channel rather than leaked), but its extracted tool calls are used
/// only when the server gave none this chunk.
///
/// A naive "parser present therefore suppress server deltas" mirror of the Gemma 4 policy would
/// regress Qwen from 5 of 7 working turns to 0 of 7 in a measured run, because
/// it discards the server's structured deltas on every turn, not just the ones where they are
/// actually absent.
/// </remarks>
internal static class ToolCallStreamCoexistence
{
    public static (string? Text, IReadOnlyList<ChatToolCallDelta>? ToolCalls) Resolve(
        string? text,
        IReadOnlyList<ChatToolCallDelta>? serverToolCallDeltas,
        IToolCallStreamParser parser)
    {
        if (serverToolCallDeltas is { Count: > 0 })
        {
            var strippedText = text is not null ? parser.Feed(text).Text : text;
            return (strippedText, serverToolCallDeltas);
        }

        if (text is null)
        {
            return (null, null);
        }

        var parsed = parser.Feed(text);
        return (parsed.Text, parsed.ToolCalls);
    }
}
