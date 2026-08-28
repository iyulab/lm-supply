using System.Diagnostics.Tracing;
using AwesomeAssertions;
using LMSupply.Generator.ChatFormatters;

namespace LMSupply.Generator.Tests;

// ──────────────────────────────────────────────────────────────────────────
// 2026-05-01 ecosystem ISSUE Option D-7 (cycle-704) — stream-tap diagnostic
// for Gemma 4 tool-call wrapper parser. Filer F-10 ask: enable independent
// validation of Surface A (parser correctness) under Surface B (model emit
// floor) masking via dotnet-trace capture of EventSource provider
// "LMSupply.Generator.ChatFormatters". The diagnostic surface must add zero
// overhead when no listener is attached and must not regress the D-5/D-6
// fixture suite (Gemma4ToolCallStreamParserTests).
// ──────────────────────────────────────────────────────────────────────────
public class Gemma4ToolCallStreamParserDiagnosticsTests
{
    [Fact]
    public void Feed_TextOnly_EmitsFeedAndEmitTextEvents()
    {
        using var listener = new TestListener();
        var parser = new Gemma4ChatFormatter().CreateToolCallStreamParser()!;

        parser.Feed("__d7_text_only__");
        parser.Flush();

        var feedEvents = listener.Events
            .Where(e => e.EventName == "Feed" && PayloadString(e, 0) == "TEXT" && PayloadString(e, 1) == "__d7_text_only__")
            .ToList();
        feedEvents.Should().NotBeEmpty(because: "Feed event must carry parser state and the raw delta payload");

        var emitTextEvents = listener.Events
            .Where(e => e.EventName == "EmitText" && PayloadString(e, 0) == "__d7_text_only__")
            .ToList();
        emitTextEvents.Should().NotBeEmpty(because: "EmitText event must mirror what Drain forwards downstream");
    }

    [Fact]
    public void Feed_StrictJsonBody_EmitsAcceptedBodyParseAndEmitToolCall()
    {
        using var listener = new TestListener();
        var parser = new Gemma4ChatFormatter().CreateToolCallStreamParser()!;

        parser.Feed("<|tool_call>call:search_knowledge{\"query\":\"d7-strict\"}<tool_call|>");

        var bodyParseEvents = listener.Events
            .Where(e => e.EventName == "BodyParse"
                && PayloadString(e, 0) == "accepted"
                && PayloadContains(e, 1, "d7-strict"))
            .ToList();
        bodyParseEvents.Should().NotBeEmpty(because: "strict-JSON body must trace as accepted with the raw body preview");

        var emitToolCallEvents = listener.Events
            .Where(e => e.EventName == "EmitToolCall"
                && PayloadString(e, 0) == "search_knowledge"
                && PayloadContains(e, 1, "d7-strict"))
            .ToList();
        emitToolCallEvents.Should().NotBeEmpty(because: "EmitToolCall must surface the parsed name and normalized args");
    }

    [Fact]
    public void Feed_RelaxedJsonBody_EmitsAcceptedBodyParse()
    {
        using var listener = new TestListener();
        var parser = new Gemma4ChatFormatter().CreateToolCallStreamParser()!;

        parser.Feed("<|tool_call>call:search_knowledge{query:\"d7-relaxed\"}<tool_call|>");

        var bodyParseEvents = listener.Events
            .Where(e => e.EventName == "BodyParse"
                && PayloadString(e, 0) == "accepted"
                && PayloadContains(e, 1, "d7-relaxed"))
            .ToList();
        bodyParseEvents.Should().NotBeEmpty(because: "D-6 relaxed normalizer success path must still emit accepted outcome");
    }

    [Fact]
    public void Feed_MalformedBody_EmitsRejectedBodyParse()
    {
        using var listener = new TestListener();
        var parser = new Gemma4ChatFormatter().CreateToolCallStreamParser()!;

        parser.Feed("<|tool_call>call:foo{__d7_malformed__ not json}<tool_call|>");

        var rejectedEvents = listener.Events
            .Where(e => e.EventName == "BodyParse"
                && PayloadString(e, 0) == "rejected"
                && PayloadContains(e, 1, "__d7_malformed__"))
            .ToList();
        rejectedEvents.Should().NotBeEmpty(because: "rejected outcome distinguishes residual body shapes for cycle-705+ triage");
    }

    [Fact]
    public void Feed_AcrossDeltas_FlushEventReportsBufferState()
    {
        using var listener = new TestListener();
        var parser = new Gemma4ChatFormatter().CreateToolCallStreamParser()!;

        parser.Feed("<|tool_call>call:slow{\"q\":");
        parser.Flush();

        var flushEvents = listener.Events
            .Where(e => e.EventName == "Flush" && PayloadString(e, 0) == "BODY")
            .ToList();
        flushEvents.Should().NotBeEmpty(because: "Flush event must record that an unclosed wrapper body was buffered at end-of-stream");
    }

    [Fact]
    public void NoListener_ParserBehaviorUnchanged()
    {
        var parser = new Gemma4ChatFormatter().CreateToolCallStreamParser()!;

        var result = parser.Feed("<|tool_call>call:search_knowledge{\"query\":\"x\"}<tool_call|>");

        result.ToolCalls.Should().NotBeNull().And.HaveCount(1,
            because: "the diagnostic seam is pure-additive; Feed contract must not change when no listener is attached");
        result.ToolCalls![0].Name.Should().Be("search_knowledge");
        result.ToolCalls[0].Arguments.Should().Contain("\"query\"");
    }

    private static string? PayloadString(EventWrittenEventArgs ev, int index)
    {
        if (ev.Payload is null || index < 0 || index >= ev.Payload.Count)
        {
            return null;
        }
        return ev.Payload[index] as string;
    }

    private static bool PayloadContains(EventWrittenEventArgs ev, int index, string substring)
    {
        var s = PayloadString(ev, index);
        return s is not null && s.Contains(substring);
    }

    private sealed class TestListener : EventListener
    {
        public List<EventWrittenEventArgs> Events { get; } = new();

        protected override void OnEventSourceCreated(EventSource eventSource)
        {
            if (eventSource.Name == "LMSupply.Generator.ChatFormatters")
            {
                EnableEvents(eventSource, EventLevel.Informational);
            }
        }

        protected override void OnEventWritten(EventWrittenEventArgs eventData)
        {
            Events.Add(eventData);
        }
    }
}
