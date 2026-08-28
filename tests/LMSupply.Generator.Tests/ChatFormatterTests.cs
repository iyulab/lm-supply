using AwesomeAssertions;
using LMSupply.Generator.Abstractions;
using LMSupply.Generator.ChatFormatters;
using LMSupply.Generator.Models;

namespace LMSupply.Generator.Tests;

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

    [Fact]
    public void GemmaChatFormatter_FormatPrompt_FormatsCorrectly()
    {
        var formatter = new GemmaChatFormatter();
        var messages = new[]
        {
            ChatMessage.User("Hello!"),
            ChatMessage.Assistant("Hi there!"),
            ChatMessage.User("How are you?")
        };

        var result = formatter.FormatPrompt(messages);

        result.Should().Contain("<start_of_turn>user");
        result.Should().Contain("<start_of_turn>model");
        result.Should().Contain("<end_of_turn>");
        result.Should().Contain("Hello!");
        result.Should().EndWith("<start_of_turn>model\n");
    }

    [Fact]
    public void GemmaChatFormatter_GetStopSequences_ReturnsExpected()
    {
        var formatter = new GemmaChatFormatter();

        var stops = formatter.GetStopSequences();

        stops.Should().Contain("<end_of_turn>");
        stops.Should().Contain("<start_of_turn>");
    }

    [Fact]
    public void ExaoneChatFormatter_FormatPrompt_FormatsCorrectly()
    {
        var formatter = new ExaoneChatFormatter();
        var messages = new[]
        {
            ChatMessage.System("You are helpful."),
            ChatMessage.User("Hello!")
        };

        var result = formatter.FormatPrompt(messages);

        result.Should().Contain("[|system|]You are helpful.[|endofturn|]");
        result.Should().Contain("[|user|]Hello![|endofturn|]");
        result.Should().EndWith("[|assistant|]");
    }

    [Fact]
    public void ExaoneChatFormatter_GetStopSequences_ReturnsExpected()
    {
        var formatter = new ExaoneChatFormatter();

        var stops = formatter.GetStopSequences();

        stops.Should().Contain("[|endofturn|]");
        stops.Should().Contain("[|user|]");
    }

    [Fact]
    public void DeepSeekChatFormatter_FormatPrompt_FormatsCorrectly()
    {
        var formatter = new DeepSeekChatFormatter();
        var messages = new[]
        {
            ChatMessage.User("What is 2+2?")
        };

        var result = formatter.FormatPrompt(messages);

        result.Should().Contain("<|user|>");
        result.Should().Contain("What is 2+2?");
        result.Should().EndWith("<|assistant|>\n");
    }

    [Fact]
    public void MistralChatFormatter_FormatPrompt_FormatsCorrectly()
    {
        var formatter = new MistralChatFormatter();
        var messages = new[]
        {
            ChatMessage.System("Be helpful."),
            ChatMessage.User("Hello!"),
            ChatMessage.Assistant("Hi!"),
            ChatMessage.User("How are you?")
        };

        var result = formatter.FormatPrompt(messages);

        result.Should().StartWith("<s>");
        result.Should().Contain("[INST]");
        result.Should().Contain("[/INST]");
        result.Should().Contain("Be helpful.");
        result.Should().Contain("Hello!");
    }

    [Fact]
    public void MistralChatFormatter_GetStopSequences_ReturnsExpected()
    {
        var formatter = new MistralChatFormatter();

        var stops = formatter.GetStopSequences();

        stops.Should().Contain("</s>");
        stops.Should().Contain("[INST]");
    }

    [Fact]
    public void Gemma4ChatFormatter_FormatPrompt_SupportsNativeSystemRole()
    {
        var formatter = new Gemma4ChatFormatter();
        var messages = new[]
        {
            ChatMessage.System("You are a helpful assistant."),
            ChatMessage.User("Hello!"),
            ChatMessage.Assistant("Hi there!"),
            ChatMessage.User("How are you?")
        };

        var result = formatter.FormatPrompt(messages);

        // Gemma 4 supports native system role (unlike Gemma 2 which mapped system→user)
        // Uses <|turn> / <turn|> tokens (Gemma 4 format), not <start_of_turn> (Gemma 2).
        result.Should().Contain("<|turn>system\nYou are a helpful assistant.<turn|>");
        result.Should().Contain("<|turn>user\nHello!<turn|>");
        result.Should().Contain("<|turn>model\nHi there!<turn|>");
        result.Should().EndWith("<|turn>model\n");
    }

    [Fact]
    public void Gemma4ChatFormatter_FormatName_IsGemma4()
    {
        var formatter = new Gemma4ChatFormatter();
        formatter.FormatName.Should().Be("gemma4");
    }

    [Fact]
    public void Gemma4ChatFormatter_GetStopSequences_ReturnsExpected()
    {
        var formatter = new Gemma4ChatFormatter();
        var stops = formatter.GetStopSequences();

        stops.Should().Contain("<turn|>");
        stops.Should().Contain("<|turn>");
    }

    [Fact]
    public void GemmaChatFormatter_SystemRole_MapsToUser()
    {
        // Verify Gemma 2 formatter still maps system→user (backward compat)
        var formatter = new GemmaChatFormatter();
        var messages = new[]
        {
            ChatMessage.System("Be helpful."),
            ChatMessage.User("Hello!")
        };

        var result = formatter.FormatPrompt(messages);

        // Gemma 2: system is mapped to user
        result.Should().Contain("<start_of_turn>user\nBe helpful.<end_of_turn>");
        result.Should().NotContain("<start_of_turn>system");
    }

    [Theory]
    [InlineData("gemma-4-E4B-it", "gemma4")]
    [InlineData("gemma4-instruct", "gemma4")]
    [InlineData("gemma-2b-instruct", "gemma")]
    public void ChatFormatterFactory_Create_DistinguishesGemmaVersions(string modelName, string expectedFormat)
    {
        var formatter = ChatFormatterFactory.Create(modelName);
        formatter.FormatName.Should().Be(expectedFormat);
    }

    [Theory]
    [InlineData("gemma")]
    [InlineData("gemma4")]
    [InlineData("exaone")]
    [InlineData("deepseek")]
    [InlineData("mistral")]
    [InlineData("mixtral")]
    [InlineData("llama3")]
    [InlineData("chatml")]
    [InlineData("phi3")]
    public void ChatFormatterFactory_CreateByFormat_CreatesFormatter(string format)
    {
        var formatter = ChatFormatterFactory.CreateByFormat(format);

        formatter.Should().NotBeNull();
        formatter.FormatName.Should().NotBeNullOrEmpty();
    }

    [Fact]
    public void ChatFormatterFactory_CreateByFormat_UnknownFormat_Throws()
    {
        var act = () => ChatFormatterFactory.CreateByFormat("unknown_format");

        act.Should().Throw<ArgumentException>()
            .WithMessage("*Unknown chat format*");
    }

    // ──────────────────────────────────────────────────────────────────────────
    // ChatRole.Tool round-trip: every formatter must accept a 4-message history
    // [system, user, assistant-tool-call, tool-result] without throwing, and
    // include the tool result content in the formatted prompt.
    // Regression guard for the IronHive 0.5.4 unmasking incident (2026-04-28):
    // ToolResult messages must not throw ArgumentOutOfRangeException, and
    // assistant-emitted tool calls must not silently disappear from history.
    // ──────────────────────────────────────────────────────────────────────────

    private static ChatMessage[] ToolRoundTripHistory() =>
    [
        ChatMessage.System("You are helpful."),
        ChatMessage.User("Search for python intro."),
        ChatMessage.AssistantToolCalls([
            new ChatToolCall("call_abc123", "search_knowledge", "{\"query\":\"python intro\"}")
        ]),
        ChatMessage.ToolResult("call_abc123", "Found python-intro.md")
    ];

    [Theory]
    [InlineData(typeof(Phi3ChatFormatter))]
    [InlineData(typeof(Llama3ChatFormatter))]
    [InlineData(typeof(ChatMLFormatter))]
    [InlineData(typeof(GemmaChatFormatter))]
    [InlineData(typeof(Gemma4ChatFormatter))]
    [InlineData(typeof(ExaoneChatFormatter))]
    [InlineData(typeof(DeepSeekChatFormatter))]
    [InlineData(typeof(MistralChatFormatter))]
    public void AllFormatters_FormatPrompt_ToolRoleHistory_DoesNotThrow(Type formatterType)
    {
        var formatter = (IChatFormatter)Activator.CreateInstance(formatterType)!;

        var act = () => formatter.FormatPrompt(ToolRoundTripHistory());

        act.Should().NotThrow();
    }

    [Theory]
    [InlineData(typeof(Phi3ChatFormatter))]
    [InlineData(typeof(Llama3ChatFormatter))]
    [InlineData(typeof(ChatMLFormatter))]
    [InlineData(typeof(GemmaChatFormatter))]
    [InlineData(typeof(Gemma4ChatFormatter))]
    [InlineData(typeof(ExaoneChatFormatter))]
    [InlineData(typeof(DeepSeekChatFormatter))]
    [InlineData(typeof(MistralChatFormatter))]
    public void AllFormatters_FormatPrompt_IncludesToolResultContent(Type formatterType)
    {
        var formatter = (IChatFormatter)Activator.CreateInstance(formatterType)!;

        var result = formatter.FormatPrompt(ToolRoundTripHistory());

        result.Should().Contain("Found python-intro.md");
    }

    [Theory]
    [InlineData(typeof(Phi3ChatFormatter))]
    [InlineData(typeof(Llama3ChatFormatter))]
    [InlineData(typeof(ChatMLFormatter))]
    [InlineData(typeof(GemmaChatFormatter))]
    [InlineData(typeof(Gemma4ChatFormatter))]
    [InlineData(typeof(ExaoneChatFormatter))]
    [InlineData(typeof(DeepSeekChatFormatter))]
    [InlineData(typeof(MistralChatFormatter))]
    public void AllFormatters_FormatPrompt_RendersAssistantToolCalls(Type formatterType)
    {
        var formatter = (IChatFormatter)Activator.CreateInstance(formatterType)!;

        var result = formatter.FormatPrompt(ToolRoundTripHistory());

        // The assistant turn used AssistantToolCalls (empty Content + ToolCalls).
        // Formatters must render the tool-call function name into the prompt so the
        // model can see what it called in the previous turn.
        result.Should().Contain("search_knowledge");
    }

    [Theory]
    [InlineData(typeof(Phi3ChatFormatter))]
    [InlineData(typeof(Llama3ChatFormatter))]
    [InlineData(typeof(ChatMLFormatter))]
    [InlineData(typeof(GemmaChatFormatter))]
    [InlineData(typeof(Gemma4ChatFormatter))]
    [InlineData(typeof(ExaoneChatFormatter))]
    [InlineData(typeof(DeepSeekChatFormatter))]
    [InlineData(typeof(MistralChatFormatter))]
    public void AllFormatters_FormatPrompt_MultipleToolResults_DoesNotThrow(Type formatterType)
    {
        // Re-entrancy guard from issue §9: multiple Tool messages in history must
        // also format successfully, not just the first.
        var formatter = (IChatFormatter)Activator.CreateInstance(formatterType)!;

        var history = new[]
        {
            ChatMessage.User("Compare python and kotlin intros."),
            ChatMessage.AssistantToolCalls([
                new ChatToolCall("call_1", "search_knowledge", "{\"query\":\"python\"}")
            ]),
            ChatMessage.ToolResult("call_1", "python-intro.md found"),
            ChatMessage.AssistantToolCalls([
                new ChatToolCall("call_2", "search_knowledge", "{\"query\":\"kotlin\"}")
            ]),
            ChatMessage.ToolResult("call_2", "kotlin-intro.md found")
        };

        var act = () => formatter.FormatPrompt(history);

        act.Should().NotThrow();
        var result = formatter.FormatPrompt(history);
        result.Should().Contain("python-intro.md found");
        result.Should().Contain("kotlin-intro.md found");
    }

    [Fact]
    public void Llama3ChatFormatter_ToolResult_UsesIpythonRole()
    {
        // Llama 3.1+ official chat template uses the ipython role for tool results.
        var formatter = new Llama3ChatFormatter();
        var messages = new[]
        {
            ChatMessage.User("Search."),
            ChatMessage.AssistantToolCalls([
                new ChatToolCall("call_1", "search", "{}")
            ]),
            ChatMessage.ToolResult("call_1", "result-payload")
        };

        var result = formatter.FormatPrompt(messages);

        result.Should().Contain("<|start_header_id|>ipython<|end_header_id|>");
        result.Should().Contain("result-payload");
    }

    [Fact]
    public void ChatMLFormatter_ToolResult_UsesToolRole()
    {
        // ChatML extension (Qwen 2.5+) uses <|im_start|>tool ... <|im_end|>.
        var formatter = new ChatMLFormatter();
        var messages = new[]
        {
            ChatMessage.User("q"),
            ChatMessage.AssistantToolCalls([
                new ChatToolCall("call_1", "fn", "{}")
            ]),
            ChatMessage.ToolResult("call_1", "tool-output")
        };

        var result = formatter.FormatPrompt(messages);

        result.Should().Contain("<|im_start|>tool");
        result.Should().Contain("tool-output");
    }

    [Fact]
    public void MistralChatFormatter_ToolResult_UsesToolResultsTokens()
    {
        // Mistral v3 instruct uses [TOOL_RESULTS] ... [/TOOL_RESULTS] tokens.
        var formatter = new MistralChatFormatter();
        var messages = new[]
        {
            ChatMessage.User("q"),
            ChatMessage.AssistantToolCalls([
                new ChatToolCall("call_1", "fn", "{\"x\":1}")
            ]),
            ChatMessage.ToolResult("call_1", "tool-output")
        };

        var result = formatter.FormatPrompt(messages);

        result.Should().Contain("[TOOL_RESULTS]");
        result.Should().Contain("[/TOOL_RESULTS]");
        result.Should().Contain("tool-output");
        result.Should().Contain("call_1");
    }

    // ──────────────────────────────────────────────────────────────────────────
    // RenderToolPromptFragment — 2026-04-30 ecosystem ISSUE Option D-1.
    // Small/quantized models (Gemma 4 E4B at gguf:default) emit empty tool args
    // and fail to self-correct under ResilientFunctionInvoker directives because
    // llama-server's native chat template renders JSON schema raw, which the
    // model misinterprets. The formatter exposes an opt-in textual reinforcement
    // fragment that LlamaServerGeneratorModel prepends as a system message.
    // ──────────────────────────────────────────────────────────────────────────

    private static System.Text.Json.JsonElement BuildJsonSchema(string json)
    {
        return System.Text.Json.JsonDocument.Parse(json).RootElement;
    }

    [Fact]
    public void Gemma4ChatFormatter_RenderToolPromptFragment_WithRequiredParam_ReturnsModelFriendlyMarkerText()
    {
        var formatter = new Gemma4ChatFormatter();
        var schema = BuildJsonSchema(
            """
            {
              "type": "object",
              "properties": {
                "path": { "type": "string", "description": "absolute file path" },
                "content": { "type": "string" }
              },
              "required": ["path"]
            }
            """);
        var tools = new[]
        {
            new ChatToolDefinition("WriteFile", "Write content to a file", schema)
        };

        var fragment = formatter.RenderToolPromptFragment(tools);

        fragment.Should().NotBeNullOrEmpty(
            because: "Gemma 4 small/quantized models need textual reinforcement of tool schemas — the native chat template's JSON schema rendering is too dense for them to follow");
        fragment!.Should().Contain("WriteFile",
            because: "tool name must be visible to anchor the model's attention");
        fragment.Should().Contain("Required parameters",
            because: "the issue specifies a model-friendly 'Required parameters (MUST be provided): ...' marker line");
        fragment.Should().Contain("path",
            because: "the required parameter name must be enumerated explicitly so the model fills it in");
        fragment.Should().Contain("string",
            because: "type information helps the model produce a well-typed argument value");
    }

    [Fact]
    public void Gemma4ChatFormatter_RenderToolPromptFragment_OptionalParamsListedSeparately()
    {
        var formatter = new Gemma4ChatFormatter();
        var schema = BuildJsonSchema(
            """
            {
              "type": "object",
              "properties": {
                "path": { "type": "string" },
                "content": { "type": "string" }
              },
              "required": ["path"]
            }
            """);
        var tools = new[]
        {
            new ChatToolDefinition("WriteFile", "Write content", schema)
        };

        var fragment = formatter.RenderToolPromptFragment(tools);

        fragment!.Should().Contain("Optional parameters",
            because: "optional vs required distinction prevents the model from confusing the two when it tries to fill arguments");
        fragment.Should().Contain("content",
            because: "optional parameter names must still be enumerated so the model knows they exist");
    }

    [Fact]
    public void Gemma4ChatFormatter_RenderToolPromptFragment_NullOrEmptyTools_ReturnsNull()
    {
        var formatter = new Gemma4ChatFormatter();

        formatter.RenderToolPromptFragment(null).Should().BeNull(
            because: "no tools = no fragment to inject; caller should not prepend an empty system message");
        formatter.RenderToolPromptFragment(Array.Empty<ChatToolDefinition>()).Should().BeNull(
            because: "empty tools collection is semantically equivalent to null");
    }

    [Fact]
    public void Gemma4ChatFormatter_RenderToolPromptFragment_MultipleTools_AllListed()
    {
        var formatter = new Gemma4ChatFormatter();
        var writeSchema = BuildJsonSchema(
            """{ "type": "object", "properties": { "path": {"type":"string"} }, "required": ["path"] }""");
        var readSchema = BuildJsonSchema(
            """{ "type": "object", "properties": { "path": {"type":"string"} }, "required": ["path"] }""");
        var tools = new[]
        {
            new ChatToolDefinition("WriteFile", "Write a file", writeSchema),
            new ChatToolDefinition("ReadFile", "Read a file", readSchema)
        };

        var fragment = formatter.RenderToolPromptFragment(tools);

        fragment!.Should().Contain("WriteFile");
        fragment.Should().Contain("ReadFile");
    }

    [Fact]
    public void Gemma4ChatFormatter_RenderToolPromptFragment_ToolWithoutRequired_StillIncludesNameAndOptional()
    {
        var formatter = new Gemma4ChatFormatter();
        var schema = BuildJsonSchema(
            """
            {
              "type": "object",
              "properties": {
                "query": { "type": "string" }
              }
            }
            """);
        var tools = new[]
        {
            new ChatToolDefinition("Search", "Search anything", schema)
        };

        var fragment = formatter.RenderToolPromptFragment(tools);

        fragment.Should().NotBeNullOrEmpty(
            because: "tools with no required params still benefit from textual exposure of their optional parameter names");
        fragment!.Should().Contain("Search");
        fragment.Should().Contain("query");
    }

    [Theory]
    [InlineData(typeof(Phi3ChatFormatter))]
    [InlineData(typeof(Llama3ChatFormatter))]
    [InlineData(typeof(ChatMLFormatter))]
    [InlineData(typeof(GemmaChatFormatter))]
    [InlineData(typeof(ExaoneChatFormatter))]
    [InlineData(typeof(DeepSeekChatFormatter))]
    [InlineData(typeof(MistralChatFormatter))]
    public void NonGemma4Formatters_RenderToolPromptFragment_ReturnsNullByDefault(Type formatterType)
    {
        // Default IChatFormatter contract: only Gemma 4 currently opts in to schema reinforcement.
        // Other formatters return null so LlamaServerGeneratorModel does not inject a duplicate
        // schema layer on top of llama-server's native template (would waste tokens for models
        // that already follow the raw schema).
        var formatter = (IChatFormatter)Activator.CreateInstance(formatterType)!;
        var schema = BuildJsonSchema(
            """{ "type": "object", "properties": { "p": {"type":"string"} }, "required": ["p"] }""");
        var tools = new[] { new ChatToolDefinition("AnyTool", "any", schema) };

        var fragment = formatter.RenderToolPromptFragment(tools);

        fragment.Should().BeNull(
            because: $"{formatterType.Name} did not opt in to textual reinforcement and must rely on llama-server's native template");
    }
}
