using AwesomeAssertions;
using LMSupply.Generator.Abstractions;
using LMSupply.Generator.ChatFormatters;
using LMSupply.Generator.Models;
using NSubstitute;

namespace LMSupply.Generator.Tests;

// ──────────────────────────────────────────────────────────────────────────
// 2026-08-17 ecosystem ISSUE Option D-8 — per-chunk source resolution for
// formatters whose grammar-constrained channel usually works (ChatML/Qwen),
// unlike Gemma 4 whose channel never does. A naive "parser registered
// therefore suppress server deltas unconditionally" mirror of the Gemma 4
// policy regressed Qwen from 5/7 working turns to 0/7 during triage — this
// is the regression guard for that exact failure mode.
// ──────────────────────────────────────────────────────────────────────────
public class ToolCallStreamCoexistenceTests
{
    private static readonly IReadOnlyList<ChatToolCallDelta> ServerDeltas =
    [
        new ChatToolCallDelta { Index = 0, Id = "call_1", Name = "search_knowledge", Arguments = "{\"q\":\"x\"}" }
    ];

    [Fact]
    public void Coexist_ServerDeltasPresent_ParserAbsent_ReturnsServerDeltas()
    {
        // The exact regression this coexistence resolver exists to prevent: the server's
        // grammar-constrained channel already produced a complete, valid delta for this chunk
        // (the majority Qwen path), and the chunk's plain text contains no wrapper for the
        // parser to find. The server's delta must win — not be silently discarded just because
        // a parser happens to be registered for the turn.
        var parser = Substitute.For<IToolCallStreamParser>();
        parser.Feed(Arg.Any<string>()).Returns(new ToolCallStreamResult { Text = "plain text", ToolCalls = null });

        var (text, calls) = ToolCallStreamCoexistence.Resolve("plain text", ServerDeltas, parser);

        calls.Should().BeSameAs(ServerDeltas,
            because: "server deltas take priority whenever present for the chunk");
        text.Should().Be("plain text");
    }

    [Fact]
    public void Coexist_ServerDeltasAbsent_ParserExtractsCall_ReturnsParserResult()
    {
        // The minority path this parser exists for: the server gave nothing this chunk (grammar
        // channel failed to constrain), and the raw text carries the model's native wrapper.
        var parser = Substitute.For<IToolCallStreamParser>();
        var parserCalls = new List<ChatToolCallDelta>
        {
            new() { Index = 0, Id = "call_cm_1", Name = "search_knowledge", Arguments = "{\"q\":\"y\"}" }
        };
        parser.Feed("<tool_call>...</tool_call>").Returns(new ToolCallStreamResult { Text = null, ToolCalls = parserCalls });

        var (text, calls) = ToolCallStreamCoexistence.Resolve("<tool_call>...</tool_call>", serverToolCallDeltas: null, parser);

        calls.Should().BeSameAs(parserCalls);
        text.Should().BeNull();
    }

    [Fact]
    public void Coexist_ServerDeltasAbsent_ParserFindsNothing_ReturnsTextUnchanged()
    {
        var parser = Substitute.For<IToolCallStreamParser>();
        parser.Feed("just talking").Returns(new ToolCallStreamResult { Text = "just talking", ToolCalls = null });

        var (text, calls) = ToolCallStreamCoexistence.Resolve("just talking", serverToolCallDeltas: null, parser);

        text.Should().Be("just talking");
        calls.Should().BeNull();
    }

    [Fact]
    public void Coexist_ServerDeltasPresent_StillFeedsTextThroughParser_ToStripIncidentalWrapperLeakage()
    {
        // Even when the server's structured channel wins, any wrapper-looking text in this same
        // chunk must still be stripped rather than shown to the user verbatim.
        var parser = Substitute.For<IToolCallStreamParser>();
        parser.Feed("prefix <tool_call>leftover").Returns(new ToolCallStreamResult { Text = "prefix ", ToolCalls = null });

        var (text, calls) = ToolCallStreamCoexistence.Resolve("prefix <tool_call>leftover", ServerDeltas, parser);

        calls.Should().BeSameAs(ServerDeltas);
        text.Should().Be("prefix ",
            because: "the parser's text output (with wrapper markers stripped) is used even when its ToolCalls are discarded in favor of the server's");
        parser.Received(1).Feed("prefix <tool_call>leftover");
    }

    [Fact]
    public void Coexist_NullTextAndNoServerDeltas_ReturnsEmptyWithoutCallingParser()
    {
        var parser = Substitute.For<IToolCallStreamParser>();

        var (text, calls) = ToolCallStreamCoexistence.Resolve(text: null, serverToolCallDeltas: null, parser);

        text.Should().BeNull();
        calls.Should().BeNull();
        parser.DidNotReceive().Feed(Arg.Any<string>());
    }

    [Fact]
    public void Coexist_NullTextButServerDeltasPresent_ReturnsServerDeltasWithNullText()
    {
        var parser = Substitute.For<IToolCallStreamParser>();

        var (text, calls) = ToolCallStreamCoexistence.Resolve(text: null, ServerDeltas, parser);

        calls.Should().BeSameAs(ServerDeltas);
        text.Should().BeNull();
        parser.DidNotReceive().Feed(Arg.Any<string>());
    }
}
