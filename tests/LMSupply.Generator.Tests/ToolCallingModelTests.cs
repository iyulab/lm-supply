using System.Text.Json;
using AwesomeAssertions;
using LMSupply.Generator.Models;

namespace LMSupply.Generator.Tests;

public class ToolCallingModelTests
{
    [Fact]
    public void ChatToolCall_Creation_SetsProperties()
    {
        var toolCall = new ChatToolCall("call_123", "get_weather", """{"location": "Seoul"}""");

        toolCall.Id.Should().Be("call_123");
        toolCall.FunctionName.Should().Be("get_weather");
        toolCall.Arguments.Should().Contain("Seoul");
    }

    [Fact]
    public void ChatToolDefinition_Creation_SetsProperties()
    {
        var parameters = JsonSerializer.Deserialize<JsonElement>(
            """{"type":"object","properties":{"location":{"type":"string"}}}""");

        var tool = new ChatToolDefinition(
            "get_weather",
            "Get weather for a location",
            parameters);

        tool.Name.Should().Be("get_weather");
        tool.Description.Should().NotBeNull();
        tool.Parameters.Should().NotBeNull();
    }

    [Fact]
    public void ChatMessage_AssistantToolCalls_CreatesCorrectMessage()
    {
        var toolCalls = new[]
        {
            new ChatToolCall("call_1", "search", "{}")
        };

        var msg = ChatMessage.AssistantToolCalls(toolCalls);

        msg.Role.Should().Be(ChatRole.Assistant);
        msg.ToolCalls.Should().HaveCount(1);
        msg.ToolCalls![0].FunctionName.Should().Be("search");
    }

    [Fact]
    public void ChatMessage_ToolResult_CreatesCorrectMessage()
    {
        var msg = ChatMessage.ToolResult("call_1", "Result data");

        msg.Role.Should().Be(ChatRole.Tool);
        msg.Content.Should().Be("Result data");
        msg.ToolCallId.Should().Be("call_1");
    }

    [Fact]
    public void ChatRole_Tool_HasValue3()
    {
        ((int)ChatRole.Tool).Should().Be(3);
    }

    [Fact]
    public void ChatCompletionResult_HasToolCalls_ReturnsTrueWhenPresent()
    {
        var result = new ChatCompletionResult
        {
            ToolCalls = new[] { new ChatToolCall("1", "f", "{}") }
        };

        result.HasToolCalls.Should().BeTrue();
    }

    [Fact]
    public void ChatCompletionResult_HasToolCalls_ReturnsFalseWhenEmpty()
    {
        var result = new ChatCompletionResult { Content = "Hello" };

        result.HasToolCalls.Should().BeFalse();
    }

    [Fact]
    public void GenerationOptions_Tools_DefaultIsNull()
    {
        var options = new GenerationOptions();
        options.Tools.Should().BeNull();
    }

    [Fact]
    public void ToolAlias_ImplicitConversion_Works()
    {
        var alias = new Models.ToolCall("id1", "func", "{}");
        ChatToolCall chatToolCall = alias;

        chatToolCall.Id.Should().Be("id1");
        chatToolCall.FunctionName.Should().Be("func");

        var parameters = JsonSerializer.Deserialize<JsonElement>("""{"type":"object"}""");
        var defAlias = new Models.ToolDefinition("fn", "desc", parameters);
        ChatToolDefinition chatToolDef = defAlias;

        chatToolDef.Name.Should().Be("fn");
        chatToolDef.Description.Should().Be("desc");
    }
}
