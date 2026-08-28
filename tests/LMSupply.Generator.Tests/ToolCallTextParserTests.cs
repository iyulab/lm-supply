using AwesomeAssertions;
using LMSupply.Generator.Internal;

namespace LMSupply.Generator.Tests;

public class ToolCallTextParserTests
{
    [Fact]
    public void TryParse_NormalText_ReturnsNull()
    {
        var result = ToolCallTextParser.TryParse("Hello, how can I help you today?");

        result.Should().BeNull();
    }

    [Fact]
    public void TryParse_ValidToolCallsJson_ParsesCorrectly()
    {
        var json = """
            {"tool_calls": [{"id": "call_123", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Seoul\"}"}}]}
            """;

        var result = ToolCallTextParser.TryParse(json);

        result.Should().NotBeNull();
        result.Should().HaveCount(1);
        result![0].Id.Should().Be("call_123");
        result[0].FunctionName.Should().Be("get_weather");
        result[0].Arguments.Should().Contain("Seoul");
    }

    [Fact]
    public void TryParse_DirectFunctionCallJson_ParsesCorrectly()
    {
        var json = """
            {"name": "get_weather", "arguments": {"city": "Seoul"}}
            """;

        var result = ToolCallTextParser.TryParse(json);

        result.Should().NotBeNull();
        result.Should().HaveCount(1);
        result![0].FunctionName.Should().Be("get_weather");
        result[0].Arguments.Should().Contain("Seoul");
        result[0].Id.Should().StartWith("call_");
    }

    [Fact]
    public void TryParse_MissingId_GeneratesId()
    {
        var json = """
            {"tool_calls": [{"type": "function", "function": {"name": "search", "arguments": "{\"q\": \"test\"}"}}]}
            """;

        var result = ToolCallTextParser.TryParse(json);

        result.Should().NotBeNull();
        result![0].Id.Should().StartWith("call_");
        result[0].FunctionName.Should().Be("search");
    }

    [Fact]
    public void TryParse_MultipleToolCalls_ParsesAll()
    {
        var json = """
            {"tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Seoul\"}"}},
                {"id": "call_2", "type": "function", "function": {"name": "get_time", "arguments": "{\"timezone\": \"KST\"}"}}
            ]}
            """;

        var result = ToolCallTextParser.TryParse(json);

        result.Should().NotBeNull();
        result.Should().HaveCount(2);
        result![0].FunctionName.Should().Be("get_weather");
        result[1].FunctionName.Should().Be("get_time");
    }

    [Fact]
    public void TryParse_MalformedJson_ReturnsNull()
    {
        var result = ToolCallTextParser.TryParse("{invalid json here}}}");

        result.Should().BeNull();
    }

    [Fact]
    public void TryParse_EmptyString_ReturnsNull()
    {
        ToolCallTextParser.TryParse("").Should().BeNull();
    }

    [Fact]
    public void TryParse_Whitespace_ReturnsNull()
    {
        ToolCallTextParser.TryParse("   \n\t  ").Should().BeNull();
    }

    [Fact]
    public void TryParse_Null_ReturnsNull()
    {
        ToolCallTextParser.TryParse(null).Should().BeNull();
    }

    [Fact]
    public void TryParse_JsonWithExtraWhitespace_ParsesCorrectly()
    {
        var json = """

              {
                "tool_calls": [
                  {
                    "id": "call_abc",
                    "type": "function",
                    "function": {
                      "name": "calculate",
                      "arguments": "{\"expression\": \"2+2\"}"
                    }
                  }
                ]
              }

            """;

        var result = ToolCallTextParser.TryParse(json);

        result.Should().NotBeNull();
        result.Should().HaveCount(1);
        result![0].Id.Should().Be("call_abc");
        result[0].FunctionName.Should().Be("calculate");
    }

    [Fact]
    public void TryParse_DirectFunctionCall_WithObjectArguments_SerializesAsJson()
    {
        var json = """{"name": "search", "arguments": {"query": "hello", "limit": 10}}""";

        var result = ToolCallTextParser.TryParse(json);

        result.Should().NotBeNull();
        result![0].FunctionName.Should().Be("search");
        result[0].Arguments.Should().Contain("query");
        result[0].Arguments.Should().Contain("10");
    }

    [Fact]
    public void TryParse_JsonWithoutToolCallsOrName_ReturnsNull()
    {
        var json = """{"type": "response", "content": "Hello world"}""";

        var result = ToolCallTextParser.TryParse(json);

        result.Should().BeNull();
    }

    [Fact]
    public void TryParse_EmptyToolCallsArray_ReturnsNull()
    {
        var json = """{"tool_calls": []}""";

        var result = ToolCallTextParser.TryParse(json);

        result.Should().BeNull();
    }

    [Fact]
    public void TryParse_ToolCallWithoutFunction_SkipsIt()
    {
        var json = """{"tool_calls": [{"id": "call_1", "type": "function"}]}""";

        var result = ToolCallTextParser.TryParse(json);

        result.Should().BeNull();
    }

    [Fact]
    public void TryParse_ToolCallsWithTrailingBrace_StillParses()
    {
        // Phi-4-mini occasionally appends a stray closing brace after the
        // legitimate tool-call object (Filer Sprint-RR1 RR-A evidence,
        // 2026-05-02). The balanced-brace extractor must isolate the first
        // valid object so the parse succeeds.
        var json = "{\"tool_calls\": [{\"id\": \"call_x\", \"type\": \"function\", " +
                   "\"function\": {\"name\": \"search_knowledge\", " +
                   "\"arguments\": {\"query\": \"python uses\"}}}]}}";

        var result = ToolCallTextParser.TryParse(json);

        result.Should().NotBeNull();
        result.Should().HaveCount(1);
        result![0].FunctionName.Should().Be("search_knowledge");
        result[0].Arguments.Should().Contain("python uses");
    }

    [Fact]
    public void TryParse_DirectCallWithTrailingWhitespaceAndBrace_StillParses()
    {
        var json = "{\"name\": \"calc\", \"arguments\": {\"x\": 1}}\n}\n  ";

        var result = ToolCallTextParser.TryParse(json);

        result.Should().NotBeNull();
        result![0].FunctionName.Should().Be("calc");
    }

    [Fact]
    public void TryParse_BraceInsideStringLiteral_NotConfusingExtractor()
    {
        // Argument value contains a '}' character — must not be mistaken for
        // the end of the top-level object.
        var json = "{\"name\": \"echo\", \"arguments\": {\"text\": \"a}b\"}}";

        var result = ToolCallTextParser.TryParse(json);

        result.Should().NotBeNull();
        result![0].FunctionName.Should().Be("echo");
        result[0].Arguments.Should().Contain("a}b");
    }

    [Fact]
    public void TryParse_MissingOuterClose_RecoversAndParses()
    {
        // Phi-4-mini RR-M evidence (2026-05-02): the model emitted a tool-call
        // envelope that closed the inner objects but forgot the outer object
        // brace (writing `]}]]` instead of `]}}`). Recovery pad supplies the
        // missing `}` and re-walks.
        var json = "{\"tool_calls\": [{\"id\": \"call_x\", \"type\": \"function\", " +
                   "\"function\": {\"name\": \"search_knowledge\", " +
                   "\"arguments\": {\"query\": \"Kafka stages\"}}]}]]";

        var result = ToolCallTextParser.TryParse(json);

        result.Should().NotBeNull();
        result.Should().HaveCount(1);
        result![0].FunctionName.Should().Be("search_knowledge");
        result[0].Arguments.Should().Contain("Kafka");
    }

    [Fact]
    public void TryParse_StrayEscapeBeforeClosingQuote_RecoversAndParses()
    {
        // Phi-4-mini RR-K evidence (2026-05-02): the model emitted an extra
        // backslash before the legitimate closing quote of the arguments
        // string. The strict balanced extractor leaves the parser inside an
        // unterminated string; the best-effort recovery removes the stray
        // backslash and re-runs the walk so the legitimate envelope still
        // parses cleanly.
        var json = "{\"tool_calls\": [{\"id\": \"call_x\", \"type\": \"function\", " +
                   "\"function\": {\"name\": \"search_knowledge\", " +
                   "\"arguments\": \"{\\\"query\\\": \\\"Kafka data pipeline stages\\\"}\\\"\"}}]}]}";

        var result = ToolCallTextParser.TryParse(json);

        result.Should().NotBeNull();
        result.Should().HaveCount(1);
        result![0].FunctionName.Should().Be("search_knowledge");
        result[0].Arguments.Should().Contain("Kafka");
    }
}
