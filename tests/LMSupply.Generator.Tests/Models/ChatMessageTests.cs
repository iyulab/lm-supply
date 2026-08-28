using AwesomeAssertions;
using LMSupply.Generator.Models;

namespace LMSupply.Generator.Tests;

public class ChatMessageTests
{
    [Fact]
    public void System_ShouldCreateSystemMessage()
    {
        var msg = ChatMessage.System("You are a helpful assistant.");

        msg.Role.Should().Be(ChatRole.System);
        msg.Content.Should().Be("You are a helpful assistant.");
    }

    [Fact]
    public void User_ShouldCreateUserMessage()
    {
        var msg = ChatMessage.User("Hello!");

        msg.Role.Should().Be(ChatRole.User);
        msg.Content.Should().Be("Hello!");
    }

    [Fact]
    public void Assistant_ShouldCreateAssistantMessage()
    {
        var msg = ChatMessage.Assistant("Hi there!");

        msg.Role.Should().Be(ChatRole.Assistant);
        msg.Content.Should().Be("Hi there!");
    }

    [Fact]
    public void StructEquality_SameValues_ShouldBeEqual()
    {
        var a = ChatMessage.User("Hello");
        var b = ChatMessage.User("Hello");

        a.Should().Be(b);
    }

    [Fact]
    public void StructEquality_DifferentContent_ShouldNotBeEqual()
    {
        var a = ChatMessage.User("Hello");
        var b = ChatMessage.User("World");

        a.Should().NotBe(b);
    }

    [Fact]
    public void StructEquality_DifferentRole_ShouldNotBeEqual()
    {
        var a = ChatMessage.User("Hello");
        var b = ChatMessage.Assistant("Hello");

        a.Should().NotBe(b);
    }

    [Fact]
    public void Constructor_ShouldInitializeFromParameters()
    {
        var msg = new ChatMessage(ChatRole.System, "Test content");

        msg.Role.Should().Be(ChatRole.System);
        msg.Content.Should().Be("Test content");
    }

    [Fact]
    public void DefaultStruct_ShouldHaveDefaultValues()
    {
        var msg = default(ChatMessage);

        msg.Role.Should().Be(ChatRole.System); // enum default = 0 = System
        msg.Content.Should().BeNull();
    }

    // --- Multimodal content parts ---

    [Fact]
    public void TextOnlyMessage_HasNoContentParts()
    {
        var msg = ChatMessage.User("Hello");

        msg.ContentParts.Should().BeNull();
        msg.IsMultimodal.Should().BeFalse();
    }

    [Fact]
    public void UserWithImage_CreatesMultimodalMessage()
    {
        var imageUrl = "data:image/jpeg;base64,abc123";
        var msg = ChatMessage.UserWithImage("What is this?", imageUrl);

        msg.Role.Should().Be(ChatRole.User);
        msg.Content.Should().Be("What is this?"); // text fallback for non-vision formatters
        msg.IsMultimodal.Should().BeTrue();
        msg.ContentParts.Should().HaveCount(2);
        msg.ContentParts![0].Should().BeOfType<TextContentPart>()
            .Which.Text.Should().Be("What is this?");
        msg.ContentParts![1].Should().BeOfType<ImageContentPart>()
            .Which.Url.Should().Be(imageUrl);
    }

    [Fact]
    public void UserWithContent_MixedParts_ExtractsTextFallback()
    {
        var parts = new ContentPart[]
        {
            new TextContentPart("First"),
            new ImageContentPart { Url = "http://example.com/img.jpg" },
            new TextContentPart("Second"),
        };

        var msg = ChatMessage.UserWithContent(parts);

        msg.Content.Should().Be("First Second");
        msg.IsMultimodal.Should().BeTrue();
        msg.ContentParts.Should().HaveCount(3);
    }

    [Fact]
    public void IsMultimodal_OnlyTextParts_ReturnsFalse()
    {
        var parts = new ContentPart[] { new TextContentPart("Hello") };
        var msg = ChatMessage.UserWithContent(parts);

        msg.IsMultimodal.Should().BeFalse();
        msg.ContentParts.Should().HaveCount(1);
    }

    [Fact]
    public void TextContentPart_TypeIsText()
    {
        new TextContentPart("hi").Type.Should().Be("text");
    }

    [Fact]
    public void ImageContentPart_TypeIsImageUrl()
    {
        new ImageContentPart { Url = "http://x.y/z.png" }.Type.Should().Be("image_url");
    }
}

public class ChatRoleEnumTests
{
    [Fact]
    public void ShouldHaveFourValues()
    {
        Enum.GetValues<ChatRole>().Should().HaveCount(4);
    }

    [Theory]
    [InlineData(ChatRole.System, 0)]
    [InlineData(ChatRole.User, 1)]
    [InlineData(ChatRole.Assistant, 2)]
    public void ShouldHaveExpectedIntValues(ChatRole role, int expected)
    {
        ((int)role).Should().Be(expected);
    }
}

public class ReasoningResultTests
{
    [Fact]
    public void ShouldInitialize_WithParameters()
    {
        var result = new ReasoningResult("The answer is 42.", "<think>Let me calculate...</think>");

        result.Response.Should().Be("The answer is 42.");
        result.Reasoning.Should().Be("<think>Let me calculate...</think>");
    }

    [Fact]
    public void StructEquality_SameValues_ShouldBeEqual()
    {
        var a = new ReasoningResult("answer", "reasoning");
        var b = new ReasoningResult("answer", "reasoning");

        a.Should().Be(b);
    }

    [Fact]
    public void StructEquality_DifferentValues_ShouldNotBeEqual()
    {
        var a = new ReasoningResult("answer1", "reasoning");
        var b = new ReasoningResult("answer2", "reasoning");

        a.Should().NotBe(b);
    }

    [Fact]
    public void DefaultStruct_ShouldHaveNullValues()
    {
        var result = default(ReasoningResult);

        result.Response.Should().BeNull();
        result.Reasoning.Should().BeNull();
    }

    [Fact]
    public void Deconstruct_ShouldWork()
    {
        var result = new ReasoningResult("response", "thought");
        var (response, reasoning) = result;

        response.Should().Be("response");
        reasoning.Should().Be("thought");
    }
}

public class ModelFormatEnumTests
{
    [Fact]
    public void ShouldHaveThreeValues()
    {
        Enum.GetValues<ModelFormat>().Should().HaveCount(3);
    }

    [Theory]
    [InlineData(ModelFormat.Onnx, 0)]
    [InlineData(ModelFormat.Gguf, 1)]
    [InlineData(ModelFormat.Unknown, 2)]
    public void ShouldHaveExpectedIntValues(ModelFormat format, int expected)
    {
        ((int)format).Should().Be(expected);
    }
}

public class GeneratorBackendTypeEnumTests
{
    [Fact]
    public void ShouldHaveTwoValues()
    {
        Enum.GetValues<GeneratorBackendType>().Should().HaveCount(2);
    }

    [Theory]
    [InlineData(GeneratorBackendType.OnnxGenAI, 0)]
    [InlineData(GeneratorBackendType.LlamaCpp, 1)]
    public void ShouldHaveExpectedIntValues(GeneratorBackendType type, int expected)
    {
        ((int)type).Should().Be(expected);
    }
}
