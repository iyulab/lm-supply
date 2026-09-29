using LMSupply.Generator.Models;

namespace LMSupply.Generator.Abstractions;

/// <summary>
/// Stateful parser that converts model-native tool-call wrapper tokens emitted in a
/// streaming text channel into structured <see cref="ChatToolCallDelta"/> events.
/// </summary>
/// <remarks>
/// Some local models emit tool calls using their native chat-template wrappers
/// (e.g. Gemma 4's <c>&lt;|tool_call&gt;...&lt;tool_call|&gt;</c>). llama-server does not always
/// recognize these wrappers and forwards them verbatim in the text channel, so the
/// chunks never become invokable tool calls upstream. A formatter that knows its own
/// wrapper format can supply an <see cref="IToolCallStreamParser"/> via
/// <see cref="IChatFormatter.CreateToolCallStreamParser"/>; the active GGUF generator
/// then routes every <c>TextDelta</c> through the parser and treats its emitted
/// tool-call deltas as the authoritative source for that turn.
/// </remarks>
public interface IToolCallStreamParser
{
    /// <summary>
    /// Feeds a streaming text delta into the parser. Returns text that should be
    /// forwarded to the downstream text channel and any complete tool-call deltas
    /// extracted from the buffered stream so far.
    /// </summary>
    /// <remarks>
    /// Parser implementations MUST hold partial wrapper prefixes/suffixes in their
    /// internal state so that wrapper tokens split across multiple deltas are
    /// recognized. Text up to (but not including) any partial wrapper match should
    /// be released; text inside an unclosed wrapper body must NOT be released into
    /// the text channel.
    /// </remarks>
    ToolCallStreamResult Feed(string textDelta);

    /// <summary>
    /// Flushes any remaining buffered state at end-of-stream. Implementations
    /// SHOULD discard incomplete tool-call bodies (rather than emit half-formed
    /// deltas that would never invoke); buffered text outside a wrapper body MAY
    /// be released as final text-channel output.
    /// </summary>
    ToolCallStreamResult Flush();
}

/// <summary>
/// Result of one <see cref="IToolCallStreamParser.Feed"/> or
/// <see cref="IToolCallStreamParser.Flush"/> call.
/// </summary>
public sealed record ToolCallStreamResult
{
    /// <summary>
    /// Text that should be forwarded to the downstream text channel for this step,
    /// or <c>null</c> if the parser produced no releasable text on this step.
    /// </summary>
    public string? Text { get; init; }

    /// <summary>
    /// Tool-call deltas completed during this step, or <c>null</c> if none.
    /// Each delta is fully formed (name + JSON arguments) — partial deltas are
    /// retained in parser state until the closing wrapper token arrives.
    /// </summary>
    public IReadOnlyList<ChatToolCallDelta>? ToolCalls { get; init; }

    /// <summary>An empty result (no text, no tool calls).</summary>
    public static ToolCallStreamResult Empty { get; } = new();
}
