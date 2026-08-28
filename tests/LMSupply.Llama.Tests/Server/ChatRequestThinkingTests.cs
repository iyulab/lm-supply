using System.Text.Json;
using AwesomeAssertions;
using LMSupply.Llama.Server;
using Xunit;

namespace LMSupply.Llama.Tests.Server;

/// <summary>
/// Tests for thinking control on the chat request body. A thinking-default-on model (Qwen3) only
/// suppresses its reasoning block when the chat request forwards <c>chat_template_kwargs:
/// {enable_thinking:false}</c> to llama-server's GGUF chat template — verified empirically against
/// Qwen3.5 (off -> 0 reasoning chars, on -> ~960). These tests pin the serialization so the wiring
/// can't silently regress. HW-free: builds the request and inspects the JSON.
/// </summary>
public class ChatRequestThinkingTests
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
    public void ThinkingOff_EmitsChatTemplateKwargs_EnableThinkingFalse()
    {
        using var doc = Serialize(new ChatCompletionOptions { EnableThinking = false });

        doc.RootElement.TryGetProperty("chat_template_kwargs", out var kwargs).Should().BeTrue(
            "thinking off must forward chat_template_kwargs to the GGUF chat template");
        kwargs.GetProperty("enable_thinking").GetBoolean().Should().BeFalse();
    }

    [Fact]
    public void ThinkingOn_EmitsChatTemplateKwargs_EnableThinkingTrue()
    {
        using var doc = Serialize(new ChatCompletionOptions { EnableThinking = true });

        doc.RootElement.TryGetProperty("chat_template_kwargs", out var kwargs).Should().BeTrue();
        kwargs.GetProperty("enable_thinking").GetBoolean().Should().BeTrue();
    }

    [Fact]
    public void ThinkingDefault_OmitsChatTemplateKwargs_PreservesModelDefault()
    {
        // EnableThinking == null (the default / Auto): do not send chat_template_kwargs at all, so the
        // model's own template default applies (Qwen3 thinks, Gemma does not) — current behavior.
        using var doc = Serialize(new ChatCompletionOptions());

        doc.RootElement.TryGetProperty("chat_template_kwargs", out _).Should().BeFalse(
            "default (unset) thinking must not alter the request — model template default is preserved");
    }
}
