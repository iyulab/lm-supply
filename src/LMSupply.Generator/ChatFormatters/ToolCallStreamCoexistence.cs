using LMSupply.Generator.Abstractions;
using LMSupply.Generator.Models;

namespace LMSupply.Generator.ChatFormatters;

/// <summary>
/// Resolves, per streaming chunk, which tool-call source is authoritative when a formatter's
/// <see cref="IChatFormatter.CreateToolCallStreamParser"/> parser coexists with llama-server's own
/// tool-call channel.
/// </summary>
/// <remarks>
/// The server's channel usually works — the parser exists to catch a native wrapper the model or the
/// server build leaves in the text. Server deltas therefore win whenever present for a chunk; the
/// parser is fed regardless (so any incidental wrapper text is still stripped from the text channel
/// rather than leaked), but its extracted tool calls are used only when the server gave none this chunk.
///
/// Suppressing the server's deltas whenever a parser is present discards working calls: for Qwen it
/// measured 0 of 7 working turns against 5 of 7, and for Gemma 4 on a build that parses the wrapper
/// itself it lost every streamed call.
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
