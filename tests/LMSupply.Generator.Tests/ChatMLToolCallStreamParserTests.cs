using AwesomeAssertions;
using LMSupply.Generator.Abstractions;
using LMSupply.Generator.ChatFormatters;
using LMSupply.Generator.Models;

namespace LMSupply.Generator.Tests;

// ──────────────────────────────────────────────────────────────────────────
// 2026-08-17 ecosystem ISSUE Option D-8 — ChatML/Qwen native tool-call wrapper
// extraction. Filer observed that on 2/7 turns llama-server forwards Qwen's
// native `<tool_call>{"name":...,"arguments":{...}}</tool_call>` wrapper
// verbatim in the streaming text channel instead of invoking it through the
// structured channel. Unlike Gemma 4 (whose structured channel never works),
// Qwen's structured channel is the primary path (5/7) — see
// ToolCallStreamCoexistenceTests for the per-chunk source-selection coverage
// that keeps this parser from regressing the working majority.
// ──────────────────────────────────────────────────────────────────────────
public class ChatMLToolCallStreamParserTests
{
    private static IToolCallStreamParser CreateParser()
    {
        var formatter = new ChatMLFormatter();
        var parser = formatter.CreateToolCallStreamParser();
        parser.Should().NotBeNull(
            because: "ChatMLFormatter must opt in to wrapper extraction to catch the native-wrapper fallback path");
        return parser!;
    }

    [Fact]
    public void Feed_SingleDelta_WithCompleteWrapper_EmitsToolCallAndStripsTokensFromText()
    {
        var parser = CreateParser();

        var result = parser.Feed(
            "I'll search memory for this.\n" +
            "<tool_call>{\"name\":\"search_knowledge\",\"arguments\":{\"query\":\"Python use cases\"}}</tool_call>\n");

        result.ToolCalls.Should().NotBeNull().And.HaveCount(1);
        var call = result.ToolCalls![0];
        call.Name.Should().Be("search_knowledge");
        call.Arguments.Should().NotBeNull();
        call.Arguments!.Should().Contain("\"query\"").And.Contain("Python use cases");

        result.Text.Should().NotBeNull();
        result.Text!.Should()
            .NotContain("<tool_call>",
                because: "wrapper tokens must never leak into the text channel")
            .And.NotContain("</tool_call>")
            .And.Contain("I'll search memory");
    }

    [Fact]
    public void Feed_KoreanQuery_ExtractedAsArgumentsValue()
    {
        var parser = CreateParser();

        var result = parser.Feed(
            "<tool_call>{\"name\":\"search_knowledge\",\"arguments\":{\"query\":\"Python의 주요 활용 분야\"}}</tool_call>");

        result.ToolCalls.Should().NotBeNull().And.HaveCount(1);
        result.ToolCalls![0].Name.Should().Be("search_knowledge");
        result.ToolCalls[0].Arguments.Should().Contain("Python의 주요 활용 분야",
            because: "Korean Unicode characters in JSON string values must round-trip through System.Text.Json");
    }

    [Fact]
    public void Feed_WrapperSplitAcrossDeltas_StillEmitsSingleToolCall()
    {
        var parser = CreateParser();

        var r1 = parser.Feed("Reasoning... <tool_");
        r1.ToolCalls.Should().BeNull(
            because: "no complete wrapper yet — must not emit a partial chunk");

        var r2 = parser.Feed("call>{\"name\":\"search_kn");
        r2.ToolCalls.Should().BeNull();

        var r3 = parser.Feed("owledge\",\"arguments\":{\"q\":\"test\"}}</tool_");
        r3.ToolCalls.Should().BeNull(
            because: "closing token incomplete — body must not be released yet");

        var r4 = parser.Feed("call>tail.");

        r4.ToolCalls.Should().NotBeNull().And.HaveCount(1);
        r4.ToolCalls![0].Name.Should().Be("search_knowledge");
        r4.ToolCalls[0].Arguments.Should().Contain("\"q\"").And.Contain("test");
        r4.Text.Should().Contain("tail.");

        var assembledText = (r1.Text ?? "") + (r2.Text ?? "") + (r3.Text ?? "") + (r4.Text ?? "");
        assembledText.Should()
            .Contain("Reasoning...")
            .And.NotContain("<tool_call>")
            .And.NotContain("</tool_call>");
    }

    [Fact]
    public void Feed_MultipleWrappersInOneDelta_EmitsTwoSeparateCallsWithDistinctIndexes()
    {
        var parser = CreateParser();

        var result = parser.Feed(
            "<tool_call>{\"name\":\"read_file\",\"arguments\":{\"path\":\"a.md\"}}</tool_call>" +
            "<tool_call>{\"name\":\"read_file\",\"arguments\":{\"path\":\"b.md\"}}</tool_call>");

        result.ToolCalls.Should().NotBeNull().And.HaveCount(2);
        result.ToolCalls![0].Index.Should().Be(0);
        result.ToolCalls[1].Index.Should().Be(1,
            because: "each emitted delta must carry a distinct index so downstream accumulation distinguishes them");
        result.ToolCalls[0].Arguments.Should().Contain("a.md");
        result.ToolCalls[1].Arguments.Should().Contain("b.md");
    }

    [Fact]
    public void Feed_MissingArgumentsProperty_DefaultsToEmptyObject()
    {
        var parser = CreateParser();

        var result = parser.Feed("<tool_call>{\"name\":\"list_files\"}</tool_call>");

        result.ToolCalls.Should().NotBeNull().And.HaveCount(1);
        result.ToolCalls![0].Arguments.Should().Be("{}",
            because: "a tool with no parameters legitimately omits the arguments key — it must not drop the whole call");
    }

    [Fact]
    public void Feed_MissingNameProperty_DropsCall()
    {
        var parser = CreateParser();

        var result = parser.Feed("<tool_call>{\"arguments\":{\"q\":1}}</tool_call>tail");

        result.ToolCalls.Should().BeNull(
            because: "a wrapper body without a name is not a valid tool call");
        result.Text.Should().Be("tail");
    }

    [Fact]
    public void Feed_MalformedJsonInsideWrapper_DropsCallAndDoesNotLeakWrapperTokens()
    {
        var parser = CreateParser();

        var result = parser.Feed("<tool_call>not json at all</tool_call>after");

        result.ToolCalls.Should().BeNull(
            because: "the parser MUST NOT emit a half-formed delta when the body is not JSON-parseable");
        result.Text.Should().NotBeNull();
        result.Text!.Should()
            .NotContain("<tool_call>")
            .And.NotContain("</tool_call>")
            .And.Contain("after");
    }

    [Fact]
    public void Feed_NonObjectRootArrayBody_DropsCall()
    {
        var parser = CreateParser();

        var result = parser.Feed("<tool_call>[1,2,3]</tool_call>");

        result.ToolCalls.Should().BeNull(
            because: "tool-call bodies are property bags — non-object roots must drop to preserve the phantom-invocation guarantee");
    }

    [Fact]
    public void Flush_OpenWrapperWithoutClosingToken_DiscardsBodyAndResets()
    {
        var parser = CreateParser();

        var feed = parser.Feed("prefix <tool_call>{\"name\":\"search_knowledge\"");
        feed.ToolCalls.Should().BeNull(
            because: "no closing wrapper yet → body must be retained, not emitted");

        var flush = parser.Flush();

        flush.ToolCalls.Should().BeNull(
            because: "incomplete wrapper at end-of-stream must NOT produce a delta — the model never finished the call");
    }

    [Fact]
    public void Feed_EmptyOrNullDelta_ReturnsEmpty()
    {
        var parser = CreateParser();

        parser.Feed(string.Empty).Should().BeSameAs(ToolCallStreamResult.Empty);
        parser.Feed(" ").ToolCalls.Should().BeNull();
    }

    [Fact]
    public void Feed_TextOnlyDelta_PassesThroughUnmodified()
    {
        var parser = CreateParser();

        var result = parser.Feed("Hello, how can I help you today?");

        result.Text.Should().Be("Hello, how can I help you today?");
        result.ToolCalls.Should().BeNull();
    }

    [Fact]
    public void ChatMLFormatter_CreateToolCallStreamParser_ReturnsFreshInstance()
    {
        var formatter = new ChatMLFormatter();

        var first = formatter.CreateToolCallStreamParser();
        var second = formatter.CreateToolCallStreamParser();

        first.Should().NotBeNull();
        second.Should().NotBeNull();
        first.Should().NotBeSameAs(second,
            because: "each turn must get its own stateful parser — sharing state across turns would corrupt buffer/tool-index counters");
    }

    [Fact]
    public void ChatMLFormatter_SuppressServerToolCallsWhenParserActive_IsFalse()
    {
        // The defining difference from Gemma 4: Qwen's grammar channel usually works, so the
        // generator must NOT discard server deltas just because this parser is registered.
        var formatter = new ChatMLFormatter();

        formatter.SuppressServerToolCallsWhenParserActive.Should().BeFalse(
            because: "unconditional suppression would regress Qwen's working majority path (5/7) to fix the minority native-wrapper leak (2/7)");
    }
}
