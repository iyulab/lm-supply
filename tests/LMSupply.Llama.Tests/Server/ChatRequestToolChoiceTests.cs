using System.Text.Json;
using AwesomeAssertions;
using LMSupply.Llama.Server;
using Xunit;

namespace LMSupply.Llama.Tests.Server;

/// <summary>
/// Tests for tool_choice on the chat request body. Pins the wire shape llama-server's
/// OpenAI-compatible /v1/chat/completions endpoint expects: a bare string for none/required,
/// {"type":"function","function":{"name":...}} to force one named function, and omission
/// entirely (server default "auto") when unset. HW-free: builds the request and inspects the JSON.
/// </summary>
public class ChatRequestToolChoiceTests
{
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
        DefaultIgnoreCondition = System.Text.Json.Serialization.JsonIgnoreCondition.WhenWritingNull
    };

    private static JsonDocument Serialize(ChatCompletionOptions options)
    {
        var request = LlamaServerClient.BuildChatRequest(
            new[] { new ChatCompletionMessage { Role = "user", Content = "hi" } },
            options,
            stream: true);
        return JsonDocument.Parse(JsonSerializer.Serialize(request, JsonOptions));
    }

    [Fact]
    public void Unset_OmitsToolChoice_PreservesServerDefaultAuto()
    {
        using var doc = Serialize(new ChatCompletionOptions());

        doc.RootElement.TryGetProperty("tool_choice", out _).Should().BeFalse(
            "unset tool_choice must not alter the request — server's own auto default applies");
    }

    [Fact]
    public void Auto_OmitsToolChoice_PreservesServerDefaultAuto()
    {
        using var doc = Serialize(new ChatCompletionOptions { ToolChoice = LlamaToolChoice.Auto });

        doc.RootElement.TryGetProperty("tool_choice", out _).Should().BeFalse();
    }

    [Fact]
    public void None_EmitsBareStringNone()
    {
        using var doc = Serialize(new ChatCompletionOptions { ToolChoice = LlamaToolChoice.None });

        doc.RootElement.TryGetProperty("tool_choice", out var toolChoice).Should().BeTrue();
        toolChoice.GetString().Should().Be("none");
    }

    [Fact]
    public void Required_EmitsBareStringRequired()
    {
        using var doc = Serialize(new ChatCompletionOptions { ToolChoice = LlamaToolChoice.Required });

        doc.RootElement.TryGetProperty("tool_choice", out var toolChoice).Should().BeTrue();
        toolChoice.GetString().Should().Be("required");
    }

    [Fact]
    public void Function_EmitsTypeAndFunctionNameObject()
    {
        using var doc = Serialize(new ChatCompletionOptions { ToolChoice = LlamaToolChoice.Function("get_weather") });

        doc.RootElement.TryGetProperty("tool_choice", out var toolChoice).Should().BeTrue();
        toolChoice.GetProperty("type").GetString().Should().Be("function");
        toolChoice.GetProperty("function").GetProperty("name").GetString().Should().Be("get_weather");
    }
}
