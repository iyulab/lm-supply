using AwesomeAssertions;
using LMSupply.Generator.Abstractions;
using LMSupply.Generator.ChatFormatters;
using LMSupply.Generator.Models;

namespace LMSupply.Generator.Tests;

// ──────────────────────────────────────────────────────────────────────────
// 2026-05-01 ecosystem ISSUE Option D-5 — Gemma 4 native tool-call wrapper
// extraction. Filer cycle-701 surfaced that llama-server forwards Gemma 4's
// `<|tool_call>call:NAME{ARGS_JSON}<tool_call|>` wrapper verbatim in the
// streaming text channel; without this parser the wrapper tokens leak as
// plain text and chunks never invoke (chat-rag flow 0% success on
// gguf:default Gemma 4 E4B).
// ──────────────────────────────────────────────────────────────────────────
public class Gemma4ToolCallStreamParserTests
{
    private static IToolCallStreamParser CreateParser()
    {
        var formatter = new Gemma4ChatFormatter();
        var parser = formatter.CreateToolCallStreamParser();
        parser.Should().NotBeNull(
            because: "Gemma 4 must opt in to wrapper extraction so the chunks reach the invoke layer");
        return parser!;
    }

    [Fact]
    public void Feed_SingleDelta_WithCompleteWrapper_EmitsToolCallAndStripsTokensFromText()
    {
        var parser = CreateParser();

        var result = parser.Feed(
            "I'll search memory for this.\n" +
            "<|tool_call>call:search_knowledge{\"query\":\"Python use cases\"}<tool_call|>\n");

        result.ToolCalls.Should().NotBeNull().And.HaveCount(1);
        var call = result.ToolCalls![0];
        call.Name.Should().Be("search_knowledge");
        call.Arguments.Should().NotBeNull();
        call.Arguments!.Should().Contain("\"query\"").And.Contain("Python use cases");

        result.Text.Should().NotBeNull();
        result.Text!.Should()
            .NotContain("<|tool_call>",
                because: "wrapper tokens must never leak into the text channel — that was the cycle-701 root cause")
            .And.NotContain("<tool_call|>")
            .And.Contain("I'll search memory");
    }

    [Fact]
    public void Feed_KoreanQuery_ExtractedAsArgumentsValue()
    {
        var parser = CreateParser();

        var result = parser.Feed(
            "<|tool_call>call:search_knowledge{\"query\":\"Python의 주요 활용 분야\"}<tool_call|>");

        result.ToolCalls.Should().NotBeNull().And.HaveCount(1);
        result.ToolCalls![0].Name.Should().Be("search_knowledge");
        result.ToolCalls[0].Arguments.Should().Contain("Python의 주요 활용 분야",
            because: "Korean Unicode characters in JSON string values must round-trip through System.Text.Json");
    }

    [Fact]
    public void Feed_WrapperSplitAcrossDeltas_StillEmitsSingleToolCall()
    {
        var parser = CreateParser();

        var r1 = parser.Feed("Reasoning... <|tool_");
        r1.ToolCalls.Should().BeNull(
            because: "no complete wrapper yet — must not emit a partial chunk");

        var r2 = parser.Feed("call>call:search_kn");
        r2.ToolCalls.Should().BeNull();

        var r3 = parser.Feed("owledge{\"q\":\"test\"}<tool_");
        r3.ToolCalls.Should().BeNull(
            because: "closing token incomplete — body must not be released yet");

        var r4 = parser.Feed("call|>tail.");

        r4.ToolCalls.Should().NotBeNull().And.HaveCount(1);
        r4.ToolCalls![0].Name.Should().Be("search_knowledge");
        r4.ToolCalls[0].Arguments.Should().Contain("\"q\"").And.Contain("test");
        r4.Text.Should().Contain("tail.");

        var assembledText = (r1.Text ?? "") + (r2.Text ?? "") + (r3.Text ?? "") + (r4.Text ?? "");
        assembledText.Should()
            .Contain("Reasoning...")
            .And.NotContain("<|tool_call>")
            .And.NotContain("<tool_call|>");
    }

    [Fact]
    public void Feed_MultipleWrappersInOneDelta_EmitsTwoSeparateCallsWithDistinctIndexes()
    {
        var parser = CreateParser();

        var result = parser.Feed(
            "<|tool_call>call:read_file{\"path\":\"a.md\"}<tool_call|>" +
            "<|tool_call>call:read_file{\"path\":\"b.md\"}<tool_call|>");

        result.ToolCalls.Should().NotBeNull().And.HaveCount(2);
        result.ToolCalls![0].Index.Should().Be(0);
        result.ToolCalls[1].Index.Should().Be(1,
            because: "each emitted delta must carry a distinct index so downstream accumulation distinguishes them");
        result.ToolCalls[0].Arguments.Should().Contain("a.md");
        result.ToolCalls[1].Arguments.Should().Contain("b.md");
    }

    [Fact]
    public void Feed_ReasoningTextMentionsToolNamesWithoutWrapper_DoesNotProduceToolCallChunk()
    {
        // §6.2 negative regression: reasoning text that names tools (e.g.
        // "GlobFiles, ReadFile 등") must NOT be emitted as tool calls. Earlier
        // cycles produced spurious half-formed name-only chunks for exactly this
        // pattern, leaving u-tool-block stuck in pending UI state.
        var parser = CreateParser();

        var result = parser.Feed(
            "사용자가 메모라이즈된 문서를 참고하여 답변을 요청했으므로, 파일 시스템 탐색 도구" +
            "(GlobFiles, ReadFile 등) 대신 지식 기반 검색 도구인 search_knowledge를 사용해야 합니다.");

        result.ToolCalls.Should().BeNull(
            because: "tool names mentioned in reasoning text must never produce ChatToolCallDelta — only wrapper-delimited bodies count");
        result.Text.Should().NotBeNull();
        result.Text!.Should().Contain("search_knowledge",
            because: "reasoning text must pass through unmodified for the user to see");
    }

    [Fact]
    public void Feed_MalformedJsonInsideWrapper_DropsCallAndDoesNotLeakWrapperTokens()
    {
        var parser = CreateParser();

        var result = parser.Feed("<|tool_call>call:foo{not json}<tool_call|>after");

        result.ToolCalls.Should().BeNull(
            because: "parser MUST NOT emit a half-formed delta when the args body is not JSON-parseable — that would leak as a non-invokable tool call upstream (the exact cycle-701 failure mode)");
        result.Text.Should().NotBeNull();
        result.Text!.Should()
            .NotContain("<|tool_call>")
            .And.NotContain("<tool_call|>")
            .And.Contain("after");
    }

    [Fact]
    public void Feed_MissingCallPrefix_DropsBody()
    {
        var parser = CreateParser();

        var result = parser.Feed("<|tool_call>{\"x\":1}<tool_call|>tail");

        result.ToolCalls.Should().BeNull(
            because: "the wrapper body must start with 'call:NAME' — bare JSON is not a valid Gemma 4 tool-call body");
        result.Text.Should().Be("tail");
    }

    [Fact]
    public void Flush_OpenWrapperWithoutClosingToken_DiscardsBodyAndResets()
    {
        var parser = CreateParser();

        var feed = parser.Feed("prefix <|tool_call>call:search_knowledge{\"q\":\"oops\"}");
        feed.ToolCalls.Should().BeNull(
            because: "no closing wrapper yet → body must be retained, not emitted");

        var flush = parser.Flush();

        flush.ToolCalls.Should().BeNull(
            because: "incomplete wrapper at end-of-stream must NOT produce a delta — the model never finished the call, so emitting one would create a phantom invocation");
    }

    [Fact]
    public void Flush_PendingPartialOpeningPrefix_EmittedAsText()
    {
        var parser = CreateParser();

        var feed = parser.Feed("normal text <|tool");
        feed.Text.Should().Be("normal text ",
            because: "trailing prefix that could be a wrapper opening must be retained until we know whether it is");

        var flush = parser.Flush();

        flush.Text.Should().Be("<|tool",
            because: "on flush, retained prefix is no longer ambiguous — release as text so the user sees what the model actually emitted");
    }

    [Fact]
    public void Feed_EmptyOrNullDelta_ReturnsEmpty()
    {
        var parser = CreateParser();

        parser.Feed(string.Empty).Should().BeSameAs(ToolCallStreamResult.Empty);
        parser.Feed(" ").ToolCalls.Should().BeNull();
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
    public void Gemma4ChatFormatter_CreateToolCallStreamParser_ReturnsFreshInstance()
    {
        var formatter = new Gemma4ChatFormatter();

        var first = formatter.CreateToolCallStreamParser();
        var second = formatter.CreateToolCallStreamParser();

        first.Should().NotBeNull();
        second.Should().NotBeNull();
        first.Should().NotBeSameAs(second,
            because: "each turn must get its own stateful parser — sharing state across turns would corrupt buffer/tool-index counters");
    }

    [Theory]
    [InlineData(typeof(Phi3ChatFormatter))]
    [InlineData(typeof(Llama3ChatFormatter))]
    [InlineData(typeof(GemmaChatFormatter))]
    [InlineData(typeof(ExaoneChatFormatter))]
    [InlineData(typeof(DeepSeekChatFormatter))]
    [InlineData(typeof(MistralChatFormatter))]
    public void NonGemma4Formatters_CreateToolCallStreamParser_ReturnsNullByDefault(Type formatterType)
    {
        // ChatMLFormatter is deliberately excluded here since ecosystem ISSUE Option D-8
        // (2026-08-17): it opts in to its own ChatMLToolCallStreamParser (coexist mode) —
        // see ChatMLToolCallStreamParserTests / ChatFormatterTests for its coverage.
        var formatter = (IChatFormatter)Activator.CreateInstance(formatterType)!;

        formatter.CreateToolCallStreamParser().Should().BeNull(
            because: $"{formatterType.Name} did not opt in to wrapper extraction; the active GGUF generator must keep server-side tool-call deltas as the source of truth for it");
    }

    // ──────────────────────────────────────────────────────────────────────
    // 2026-05-01 ecosystem ISSUE Option D-6 (cycle-703) — Filer cycle-702 ran
    // 0.32.4 against three Korean RAG prompts and got 0/3 non-empty responses.
    // Root cause: Gemma 4's native body shape `{query:"…"}` is JS-object-literal,
    // not strict JSON, so D-5's `JsonDocument.Parse` rejected real tool calls
    // and silent_llm_empty fired. D-6 routes failed strict parses through
    // `RelaxedJsonNormalizer` (unquoted identifier keys, single-quoted string
    // values, trailing commas) before re-validating with strict JSON. The
    // object-only guard preserves D-5's phantom-invocation prevention.
    // ──────────────────────────────────────────────────────────────────────

    [Fact]
    public void Feed_Gemma4NativeUnquotedKeyBody_ExtractsToolCallWithStrictJsonArguments()
    {
        var parser = CreateParser();

        var result = parser.Feed(
            "<|tool_call>call:search_knowledge{query:\"Python use cases\"}<tool_call|>");

        result.ToolCalls.Should().NotBeNull().And.HaveCount(1,
            because: "Gemma 4's native unquoted-key body must round-trip — this exact shape was the cycle-702 silent-empty regression");
        var call = result.ToolCalls![0];
        call.Name.Should().Be("search_knowledge");
        call.Arguments.Should().Be("{\"query\":\"Python use cases\"}",
            because: "the relaxed normalizer must rewrite the body into strict JSON so downstream JsonDocument consumers see canonical output");
    }

    [Fact]
    public void Feed_Gemma4NativeBodyWithKoreanValue_RoundTripsAsStrictJson()
    {
        var parser = CreateParser();

        var result = parser.Feed(
            "<|tool_call>call:search_knowledge{query:\"Python의 주요 활용 분야\"}<tool_call|>");

        result.ToolCalls.Should().NotBeNull().And.HaveCount(1,
            because: "this is the verbatim cycle-702 production emit — D-6 must accept it");
        result.ToolCalls![0].Arguments.Should()
            .StartWith("{").And.EndWith("}")
            .And.Contain("\"query\"")
            .And.Contain("Python의 주요 활용 분야",
                because: "Korean Unicode in string values must survive normalization without re-encoding");
    }

    [Fact]
    public void Feed_SingleQuotedStringValue_NormalizedToDoubleQuoted()
    {
        var parser = CreateParser();

        var result = parser.Feed(
            "<|tool_call>call:read_file{path:'docs/intro.md'}<tool_call|>");

        result.ToolCalls.Should().NotBeNull().And.HaveCount(1);
        result.ToolCalls![0].Arguments.Should().Be("{\"path\":\"docs/intro.md\"}",
            because: "JS-style single-quoted string values must be rewritten to JSON double-quoted form");
    }

    [Fact]
    public void Feed_TrailingCommaBeforeClosingBrace_Stripped()
    {
        var parser = CreateParser();

        var result = parser.Feed(
            "<|tool_call>call:foo{a:1,b:2,}<tool_call|>");

        result.ToolCalls.Should().NotBeNull().And.HaveCount(1);
        result.ToolCalls![0].Arguments.Should().Be("{\"a\":1,\"b\":2}",
            because: "trailing commas before the closing brace are valid JSON5 / JS but must be stripped for strict JSON output");
    }

    [Fact]
    public void Feed_NestedObjectWithUnquotedKeys_NormalizedRecursively()
    {
        var parser = CreateParser();

        var result = parser.Feed(
            "<|tool_call>call:configure{opts:{deep:true,limit:5}}<tool_call|>");

        result.ToolCalls.Should().NotBeNull().And.HaveCount(1);
        result.ToolCalls![0].Arguments.Should().Contain("\"opts\"")
            .And.Contain("\"deep\":true")
            .And.Contain("\"limit\":5",
                because: "the normalizer walks the whole input as a single pass — nested object keys must also be quoted");
    }

    [Fact]
    public void Feed_StringValueContainsBraceColonAndComma_PreservedVerbatim()
    {
        var parser = CreateParser();

        var result = parser.Feed(
            "<|tool_call>call:lookup{q:\"key:value, {brace}\"}<tool_call|>");

        result.ToolCalls.Should().NotBeNull().And.HaveCount(1,
            because: "structural-looking characters inside string values must not trip the normalizer's lexer");
        result.ToolCalls![0].Arguments.Should().Contain("key:value, {brace}");
    }

    [Fact]
    public void Feed_NumericBooleanAndNullValues_PassThroughUnmodified()
    {
        var parser = CreateParser();

        var result = parser.Feed(
            "<|tool_call>call:foo{n:42,b:true,nu:null,f:1.5}<tool_call|>");

        result.ToolCalls.Should().NotBeNull().And.HaveCount(1);
        result.ToolCalls![0].Arguments.Should().Be(
            "{\"n\":42,\"b\":true,\"nu\":null,\"f\":1.5}",
            because: "JSON primitives in value position must not be wrapped in quotes; only bare keys are quoted");
    }

    [Fact]
    public void Feed_StrictJsonBody_PassesThroughWithoutNormalization()
    {
        var parser = CreateParser();

        var result = parser.Feed(
            "<|tool_call>call:search_knowledge{\"query\":\"already strict\"}<tool_call|>");

        result.ToolCalls.Should().NotBeNull().And.HaveCount(1,
            because: "D-6 must not regress D-5's strict-JSON round-trip — strict path is tried first and must keep working");
        result.ToolCalls![0].Arguments.Should().Be("{\"query\":\"already strict\"}");
    }

    [Fact]
    public void Feed_NonObjectRootArrayBody_DropsCall()
    {
        var parser = CreateParser();

        var result = parser.Feed(
            "<|tool_call>call:foo[1,2,3]<tool_call|>");

        result.ToolCalls.Should().BeNull(
            because: "Gemma 4 tool-call bodies are property bags — non-object roots (arrays, scalars) must drop to preserve the phantom-invocation guarantee");
    }

    [Fact]
    public void Feed_TotallyGarbageBody_DropsCallWithoutLeakingWrapperTokens()
    {
        var parser = CreateParser();

        var result = parser.Feed("<|tool_call>call:foo{not even close}<tool_call|>after");

        result.ToolCalls.Should().BeNull(
            because: "even after relaxed normalization, syntactically broken bodies must be dropped — D-5 phantom-invocation guard is preserved");
        result.Text.Should().NotBeNull();
        result.Text!.Should()
            .NotContain("<|tool_call>")
            .And.NotContain("<tool_call|>")
            .And.Contain("after",
                because: "wrapper tokens must still be stripped from text channel even when the body is unparseable");
    }

    [Fact]
    public void Feed_MixedQuotedAndUnquotedKeysInOneBody_AllNormalized()
    {
        var parser = CreateParser();

        var result = parser.Feed(
            "<|tool_call>call:foo{\"already\":\"quoted\",bare:\"key\"}<tool_call|>");

        result.ToolCalls.Should().NotBeNull().And.HaveCount(1);
        result.ToolCalls![0].Arguments.Should()
            .Contain("\"already\":\"quoted\"")
            .And.Contain("\"bare\":\"key\"",
                because: "the normalizer must coexist with already-quoted keys — partial-strict bodies are common when models drift mid-stream");
    }
}
