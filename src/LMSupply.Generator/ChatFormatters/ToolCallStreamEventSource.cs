using System.Diagnostics.Tracing;

namespace LMSupply.Generator.ChatFormatters;

/// <summary>
/// EventSource exposing Gemma 4 tool-call stream parser internals as ETW/EventPipe
/// events so parser correctness can be validated independently of what the model
/// emits. Emits raw input deltas, body-parse outcomes, and
/// emitted text/tool-call payloads so a trace can map model emit 1:1 to parser emission.
/// </summary>
/// <remarks>
/// Capture (dev/diagnostic only):
/// <code>dotnet-trace collect --providers LMSupply.Generator.ChatFormatters</code>
/// Production overhead is zero when no listener is attached — every emit site is gated
/// by <see cref="EventSource.IsEnabled()"/>. The parser remains internal-stable; this
/// type is a pure-additive diagnostic seam, no <see cref="Abstractions.IToolCallStreamParser"/>
/// or <see cref="Abstractions.IChatFormatter"/> contract change.
/// </remarks>
[EventSource(Name = "LMSupply.Generator.ChatFormatters")]
internal sealed class ToolCallStreamEventSource : EventSource
{
    /// <summary>Singleton event source instance; access via this field.</summary>
    public static readonly ToolCallStreamEventSource Log = new();

    private ToolCallStreamEventSource()
    {
    }

    /// <summary>Raw input delta received via Feed. <paramref name="state"/> is "TEXT" or "BODY".</summary>
    [Event(1, Level = EventLevel.Informational, Message = "Feed: state={0} delta={1}")]
    public void Feed(string state, string deltaPreview)
    {
        if (IsEnabled())
        {
            WriteEvent(1, state, deltaPreview);
        }
    }

    /// <summary>Text emitted to the downstream text channel from one Feed/Flush step.</summary>
    [Event(2, Level = EventLevel.Informational, Message = "EmitText: text={0}")]
    public void EmitText(string textPreview)
    {
        if (IsEnabled())
        {
            WriteEvent(2, textPreview);
        }
    }

    /// <summary>Complete tool-call delta emitted upstream (name + normalized args JSON).</summary>
    [Event(3, Level = EventLevel.Informational, Message = "EmitToolCall: name={0} args={1}")]
    public void EmitToolCall(string name, string argsPreview)
    {
        if (IsEnabled())
        {
            WriteEvent(3, name, argsPreview);
        }
    }

    /// <summary>
    /// Tool-call wrapper body parse outcome. <paramref name="outcome"/> is "accepted"
    /// (strict or relaxed JSON normalized successfully) or "rejected" (body neither
    /// strict-JSON nor relaxed-JSON object form).
    /// </summary>
    /// <remarks>
    /// Correlating rejected events with the raw model emit in a trace shows whether a
    /// body shape is one the relaxed parse already covers or a new one.
    /// </remarks>
    [Event(4, Level = EventLevel.Informational, Message = "BodyParse: outcome={0} body={1}")]
    public void BodyParse(string outcome, string bodyPreview)
    {
        if (IsEnabled())
        {
            WriteEvent(4, outcome, bodyPreview);
        }
    }

    /// <summary>Flush at end-of-stream. <paramref name="state"/> is "TEXT" or "BODY".</summary>
    [Event(5, Level = EventLevel.Informational, Message = "Flush: state={0} buffer={1}")]
    public void Flush(string state, string bufferPreview)
    {
        if (IsEnabled())
        {
            WriteEvent(5, state, bufferPreview);
        }
    }
}
