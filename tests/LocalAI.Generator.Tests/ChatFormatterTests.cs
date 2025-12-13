using FluentAssertions;
using LocalAI.Generator.ChatFormatters;
using LocalAI.Generator.Models;

namespace LocalAI.Generator.Tests;

public class ChatFormatterTests
{
    [Fact]
    public void Phi3ChatFormatter_FormatPrompt_FormatsCorrectly()
    {
        // Arrange
        var formatter = new Phi3ChatFormatter();
        var messages = new[]
        {
            ChatMessage.System("You are a helpful assistant."),
            ChatMessage.User("Hello!")
        };

        // Act
        var result = formatter.FormatPrompt(messages);

        // Assert
        result.Should().Contain("<|system|>");
        result.Should().Contain("You are a helpful assistant.");
        result.Should().Contain("<|end|>");
        result.Should().Contain("<|user|>");
        result.Should().Contain("Hello!");
        result.Should().EndWith("<|assistant|>\n");
    }

    [Fact]
    public void Phi3ChatFormatter_GetStopSequences_ReturnsExpectedSequences()
    {
        // Arrange
        var formatter = new Phi3ChatFormatter();

        // Act
        var stopSequences = formatter.GetStopSequences();

        // Assert
        stopSequences.Should().Contain("<|end|>");
        stopSequences.Should().Contain("<|user|>");
    }

    [Fact]
    public void Llama3ChatFormatter_FormatPrompt_FormatsCorrectly()
    {
        // Arrange
        var formatter = new Llama3ChatFormatter();
        var messages = new[]
        {
            ChatMessage.System("You are a helpful assistant."),
            ChatMessage.User("Hello!")
        };

        // Act
        var result = formatter.FormatPrompt(messages);

        // Assert
        result.Should().Contain("<|begin_of_text|>");
        result.Should().Contain("<|start_header_id|>system<|end_header_id|>");
        result.Should().Contain("You are a helpful assistant.");
        result.Should().Contain("<|eot_id|>");
        result.Should().Contain("<|start_header_id|>user<|end_header_id|>");
        result.Should().EndWith("<|start_header_id|>assistant<|end_header_id|>\n\n");
    }

    [Fact]
    public void ChatMLFormatter_FormatPrompt_FormatsCorrectly()
    {
        // Arrange
        var formatter = new ChatMLFormatter();
        var messages = new[]
        {
            ChatMessage.System("You are a helpful assistant."),
            ChatMessage.User("Hello!")
        };

        // Act
        var result = formatter.FormatPrompt(messages);

        // Assert
        result.Should().Contain("<|im_start|>system");
        result.Should().Contain("You are a helpful assistant.");
        result.Should().Contain("<|im_end|>");
        result.Should().Contain("<|im_start|>user");
        result.Should().EndWith("<|im_start|>assistant\n");
    }

    [Theory]
    [InlineData("phi-3-mini", "phi3")]
    [InlineData("Phi-3.5-mini-instruct", "phi3")]
    [InlineData("llama-3-8b", "llama3")]
    [InlineData("Llama-3.2-1B-Instruct", "llama3")]
    [InlineData("qwen2.5-7b", "chatml")]
    [InlineData("unknown-model", "phi3")] // Default
    public void ChatFormatterFactory_Create_ReturnsCorrectFormatter(string modelName, string expectedFormat)
    {
        // Act
        var formatter = ChatFormatterFactory.Create(modelName);

        // Assert
        formatter.FormatName.Should().Be(expectedFormat);
    }
}
