using System.Text.Json;
using AwesomeAssertions;
using LMSupply.Generator.Abstractions;
using LMSupply.Generator.ChatFormatters;
using LMSupply.Generator.Internal.Llama;
using LMSupply.Generator.Models;

namespace LMSupply.Generator.Tests.Internal.Llama;

/// <summary>
/// Tests for internal static helper methods on LlamaServerGeneratorModel.
/// These test the VRAM-adaptive logic without requiring actual model files or servers.
/// </summary>
public class LlamaServerGeneratorModelHelperTests
{
    // ─── IsOomError ───

    [Theory]
    [InlineData("CUDA error: out of memory")]
    [InlineData("CUDA_ERROR_OUT_OF_MEMORY")]
    [InlineData("Could not allocate 2048 MiB")]
    [InlineData("ggml_backend_cuda_buffer_type_alloc_buffer: failed to alloc")]
    [InlineData("Error: Out of Memory while loading model")]
    public void IsOomError_TrueForGpuOomMessages(string message)
    {
        var ex = new InvalidOperationException(message);
        LlamaServerGeneratorModel.IsOomError(ex).Should().BeTrue();
    }

    [Theory]
    [InlineData("Connection refused")]
    [InlineData("File not found")]
    [InlineData("Invalid model format")]
    [InlineData("Server startup timeout")]
    [InlineData("")]
    public void IsOomError_FalseForNonOomErrors(string message)
    {
        var ex = new InvalidOperationException(message);
        LlamaServerGeneratorModel.IsOomError(ex).Should().BeFalse();
    }

    // ─── EstimateTotalLayers ───

    [Theory]
    [InlineData(1L * 1024 * 1024 * 1024, 22)]      // <2GB → 22 layers
    [InlineData(3L * 1024 * 1024 * 1024, 28)]      // 2-5GB → 28 layers
    [InlineData(7L * 1024 * 1024 * 1024, 32)]      // 5-10GB → 32 layers
    [InlineData(15L * 1024 * 1024 * 1024, 40)]     // >10GB → 40 layers
    public void EstimateTotalLayers_ReturnsExpectedForFileSize(long fileSize, int expectedLayers)
    {
        LlamaServerGeneratorModel.EstimateTotalLayers(fileSize).Should().Be(expectedLayers);
    }

    [Fact]
    public void EstimateTotalLayers_BoundaryValues()
    {
        // Exactly 2GB boundary
        var just_under_2gb = 2L * 1024 * 1024 * 1024 - 1;
        var exactly_2gb = 2L * 1024 * 1024 * 1024;

        LlamaServerGeneratorModel.EstimateTotalLayers(just_under_2gb).Should().Be(22);
        LlamaServerGeneratorModel.EstimateTotalLayers(exactly_2gb).Should().Be(28);
    }

    // ─── MaybeInjectToolPromptFragment (Option D-1, 2026-04-30) ───

    private static JsonElement BuildSchema(string json) => JsonDocument.Parse(json).RootElement;

    [Fact]
    public void MaybeInjectToolPromptFragment_FormatterReturnsFragment_PrependsSystemMessage()
    {
        var formatter = new Gemma4ChatFormatter();
        var schema = BuildSchema("""{ "type":"object", "properties":{ "path":{"type":"string"} }, "required":["path"] }""");
        var tools = new[] { new ChatToolDefinition("WriteFile", "Write a file", schema) };
        var original = new[]
        {
            ChatMessage.System("You are helpful."),
            ChatMessage.User("write /tmp/x")
        };

        var result = LlamaServerGeneratorModel.MaybeInjectToolPromptFragment(original, tools, formatter).ToList();

        result.Should().HaveCount(3,
            because: "fragment is prepended as a new system message — original 2 messages stay");
        result[0].Role.Should().Be(ChatRole.System);
        result[0].Content.Should().Contain("Required parameters",
            because: "the prepended message must carry Gemma4's textual reinforcement so llama-server forwards it to the model");
        result[1].Should().Be(original[0],
            because: "original messages are passed through unchanged");
        result[2].Should().Be(original[1]);
    }

    [Fact]
    public void MaybeInjectToolPromptFragment_FormatterReturnsNull_PassesMessagesThrough()
    {
        // Phi3 doesn't opt in to schema reinforcement.
        var formatter = new Phi3ChatFormatter();
        var schema = BuildSchema("""{ "type":"object", "properties":{ "p":{"type":"string"} }, "required":["p"] }""");
        var tools = new[] { new ChatToolDefinition("AnyTool", "any", schema) };
        var original = new[]
        {
            ChatMessage.System("sys"),
            ChatMessage.User("u")
        };

        var result = LlamaServerGeneratorModel.MaybeInjectToolPromptFragment(original, tools, formatter).ToList();

        result.Should().HaveCount(2,
            because: "Phi3 returned null — pipeline must not synthesize an empty system message");
        result[0].Should().Be(original[0]);
        result[1].Should().Be(original[1]);
    }

    [Fact]
    public void MaybeInjectToolPromptFragment_NullTools_PassesMessagesThrough()
    {
        var formatter = new Gemma4ChatFormatter();
        var original = new[] { ChatMessage.User("hello") };

        var result = LlamaServerGeneratorModel.MaybeInjectToolPromptFragment(original, null, formatter).ToList();

        result.Should().HaveCount(1);
        result[0].Should().Be(original[0]);
    }

    [Fact]
    public void MaybeInjectToolPromptFragment_EmptyTools_PassesMessagesThrough()
    {
        var formatter = new Gemma4ChatFormatter();
        var original = new[] { ChatMessage.User("hello") };

        var result = LlamaServerGeneratorModel.MaybeInjectToolPromptFragment(
            original, Array.Empty<ChatToolDefinition>(), formatter).ToList();

        result.Should().HaveCount(1);
    }

    // ─── MaybeInjectToolPromptFragment — thinking mode (Gap B, 2026-05-08) ───

    [Fact]
    public void MaybeInjectToolPromptFragment_ThinkingEnabled_Gemma4_UsesMinimalFragment()
    {
        var formatter = new Gemma4ChatFormatter();
        var schema = BuildSchema("""{ "type":"object", "properties":{ "query":{"type":"string"} }, "required":["query"] }""");
        var tools = new[] { new ChatToolDefinition("search_knowledge", "Search tool", schema) };
        var original = new[] { ChatMessage.User("search something") };

        var result = LlamaServerGeneratorModel.MaybeInjectToolPromptFragment(original, tools, formatter, enableThinking: true).ToList();

        result.Should().HaveCount(2, because: "minimal fragment is still prepended as a system message");
        result[0].Role.Should().Be(ChatRole.System);
        result[0].Content.Should().Contain("search_knowledge",
            because: "tool name must be present in the minimal hint");
        result[0].Content.Should().Contain("query (string)",
            because: "required parameter hint must be present");
        result[0].Content.Should().NotContain("You have access to the following tools",
            because: "full D-1 preamble must be absent — only the minimal hint is injected when thinking is on");
        result[0].Content.Should().NotContain("Optional parameters",
            because: "optional params are omitted in the minimal fragment to reduce context pressure");
    }

    [Fact]
    public void MaybeInjectToolPromptFragment_ThinkingEnabled_NonGemma4_SkipsFragment()
    {
        // Non-Gemma4 formatters return null from RenderToolPromptFragmentWhenThinking → no fragment.
        var formatter = new Phi3ChatFormatter();
        var schema = BuildSchema("""{ "type":"object", "properties":{ "q":{"type":"string"} }, "required":["q"] }""");
        var tools = new[] { new ChatToolDefinition("AnyTool", "any", schema) };
        var original = new[] { ChatMessage.System("sys"), ChatMessage.User("u") };

        var result = LlamaServerGeneratorModel.MaybeInjectToolPromptFragment(original, tools, formatter, enableThinking: true).ToList();

        result.Should().HaveCount(2,
            because: "Phi3 returns null from RenderToolPromptFragmentWhenThinking — no extra system message when thinking is on");
        result[0].Should().Be(original[0]);
    }

    [Fact]
    public void MaybeInjectToolPromptFragment_ThinkingDisabled_Gemma4_UsesFullFragment()
    {
        var formatter = new Gemma4ChatFormatter();
        var schema = BuildSchema("""{ "type":"object", "properties":{ "path":{"type":"string"} }, "required":["path"] }""");
        var tools = new[] { new ChatToolDefinition("WriteFile", "Write a file", schema) };
        var original = new[] { ChatMessage.User("write") };

        var result = LlamaServerGeneratorModel.MaybeInjectToolPromptFragment(original, tools, formatter, enableThinking: false).ToList();

        result.Should().HaveCount(2);
        result[0].Content.Should().Contain("You have access to the following tools",
            because: "enableThinking=false uses the full D-1 fragment as before");
    }

    [Fact]
    public void MaybeInjectToolPromptFragment_ThinkingEnabled_NoRequiredParams_SkipsFragment()
    {
        // All-optional tool: RenderToolPromptFragmentWhenThinking returns null → no fragment.
        var formatter = new Gemma4ChatFormatter();
        var schema = BuildSchema("""{ "type":"object", "properties":{ "opt":{"type":"string"} } }""");
        var tools = new[] { new ChatToolDefinition("OptTool", "optional tool", schema) };
        var original = new[] { ChatMessage.User("go") };

        var result = LlamaServerGeneratorModel.MaybeInjectToolPromptFragment(original, tools, formatter, enableThinking: true).ToList();

        result.Should().HaveCount(1,
            because: "no required parameters → RenderToolPromptFragmentWhenThinking returns null → no fragment injected");
    }

    // ─── Gemma4ChatFormatter.RenderToolPromptFragmentWhenThinking (Gap B, 2026-05-08) ───

    [Fact]
    public void Gemma4_RenderToolPromptFragmentWhenThinking_SingleTool_RequiredParamsOnly()
    {
        var formatter = new Gemma4ChatFormatter();
        var schema = BuildSchema("""{ "type":"object", "properties":{ "query":{"type":"string"}, "limit":{"type":"integer"} }, "required":["query"] }""");
        var tools = new[] { new ChatToolDefinition("search", "Search docs", schema) };

        var result = formatter.RenderToolPromptFragmentWhenThinking(tools);

        result.Should().NotBeNull();
        result.Should().Contain("search: query (string)",
            because: "only required params are listed, with tool name prefix");
        result.Should().NotContain("limit",
            because: "optional params are omitted entirely");
        result.Should().NotContain("Search docs",
            because: "tool description is omitted to minimize context pressure");
    }

    [Fact]
    public void Gemma4_RenderToolPromptFragmentWhenThinking_MultipleTools_RequiredOnly()
    {
        var formatter = new Gemma4ChatFormatter();
        var schema1 = BuildSchema("""{ "type":"object", "properties":{ "q":{"type":"string"} }, "required":["q"] }""");
        var schema2 = BuildSchema("""{ "type":"object", "properties":{ "id":{"type":"integer"} }, "required":["id"] }""");
        var tools = new[]
        {
            new ChatToolDefinition("search", "s", schema1),
            new ChatToolDefinition("get_item", "g", schema2)
        };

        var result = formatter.RenderToolPromptFragmentWhenThinking(tools);

        result.Should().NotBeNull();
        result.Should().Contain("search: q (string)");
        result.Should().Contain("get_item: id (integer)");
    }

    [Fact]
    public void Gemma4_RenderToolPromptFragmentWhenThinking_NullTools_ReturnsNull()
    {
        var formatter = new Gemma4ChatFormatter();
        formatter.RenderToolPromptFragmentWhenThinking(null).Should().BeNull();
    }

    [Fact]
    public void Gemma4_RenderToolPromptFragmentWhenThinking_EmptyTools_ReturnsNull()
    {
        var formatter = new Gemma4ChatFormatter();
        formatter.RenderToolPromptFragmentWhenThinking(Array.Empty<ChatToolDefinition>()).Should().BeNull();
    }

    [Fact]
    public void Gemma4_RenderToolPromptFragmentWhenThinking_AllOptionalParams_ReturnsNull()
    {
        var formatter = new Gemma4ChatFormatter();
        var schema = BuildSchema("""{ "type":"object", "properties":{ "opt":{"type":"string"} } }""");
        var tools = new[] { new ChatToolDefinition("tool", "desc", schema) };

        formatter.RenderToolPromptFragmentWhenThinking(tools).Should().BeNull(
            because: "tools with no required params produce no hint, so null is returned to skip fragment injection");
    }

    [Fact]
    public void MaybeInjectToolPromptFragment_ThinkingEnabled_ThenThinkingToken_MinimalFragmentFirst()
    {
        // Simulates real call order with thinking ON: minimal fragment then thinking token.
        var formatter = new Gemma4ChatFormatter();
        var schema = BuildSchema("""{ "type":"object", "properties":{ "query":{"type":"string"} }, "required":["query"] }""");
        var tools = new[] { new ChatToolDefinition("search_knowledge", "Search", schema) };
        var original = new[] { ChatMessage.User("search something") };

        var withFragment = LlamaServerGeneratorModel.MaybeInjectToolPromptFragment(original, tools, formatter, enableThinking: true).ToList();
        var result = LlamaServerGeneratorModel.MaybeInjectThinkingToken(withFragment, enableThinking: true, formatter).ToList();

        result.Should().HaveCount(2);
        result[0].Role.Should().Be(ChatRole.System);
        result[0].Content.Should().StartWith("<|think|>",
            because: "thinking token precedes the minimal fragment in the combined system message");
        result[0].Content.Should().Contain("search_knowledge: query (string)",
            because: "minimal required-params hint is preserved after thinking token injection");
        result[0].Content.Should().NotContain("You have access to the following tools",
            because: "full D-1 preamble must not be present when thinking is active");
    }

    // ─── MaybeInjectThinkingToken (Gap A — Gemma 4 thinking mode, 2026-05-05) ───

    [Fact]
    public void MaybeInjectThinkingToken_EnabledWithGemma4_PrependsTokenToSystemMessage()
    {
        var formatter = new Gemma4ChatFormatter();
        var messages = new[]
        {
            ChatMessage.System("You are helpful."),
            ChatMessage.User("call a tool")
        };

        var result = LlamaServerGeneratorModel.MaybeInjectThinkingToken(messages, enableThinking: true, formatter).ToList();

        result.Should().HaveCount(2, because: "no new messages are inserted — the existing system message is modified");
        result[0].Role.Should().Be(ChatRole.System);
        result[0].Content.Should().StartWith("<|think|>",
            because: "thinking token must be the first content in the system message");
        result[0].Content.Should().Contain("You are helpful.",
            because: "original system content must be preserved");
        result[1].Should().Be(messages[1]);
    }

    [Fact]
    public void MaybeInjectThinkingToken_EnabledWithNoSystemMessage_PrependsBareSytemMessage()
    {
        var formatter = new Gemma4ChatFormatter();
        var messages = new[] { ChatMessage.User("call a tool") };

        var result = LlamaServerGeneratorModel.MaybeInjectThinkingToken(messages, enableThinking: true, formatter).ToList();

        result.Should().HaveCount(2, because: "a bare system message with the thinking token is prepended");
        result[0].Role.Should().Be(ChatRole.System);
        result[0].Content.Should().Be("<|think|>");
        result[1].Should().Be(messages[0]);
    }

    [Fact]
    public void MaybeInjectThinkingToken_DisabledWithGemma4_PassesMessagesThrough()
    {
        var formatter = new Gemma4ChatFormatter();
        var messages = new[]
        {
            ChatMessage.System("You are helpful."),
            ChatMessage.User("hello")
        };

        var result = LlamaServerGeneratorModel.MaybeInjectThinkingToken(messages, enableThinking: false, formatter).ToList();

        result.Should().HaveCount(2);
        result[0].Content.Should().Be("You are helpful.", because: "system content must not be modified when thinking is disabled");
    }

    [Fact]
    public void MaybeInjectThinkingToken_EnabledWithNonThinkingFormatter_PassesMessagesThrough()
    {
        // Phi3 returns null from GetThinkingToken() — must be a no-op.
        var formatter = new Phi3ChatFormatter();
        var messages = new[]
        {
            ChatMessage.System("sys"),
            ChatMessage.User("user")
        };

        var result = LlamaServerGeneratorModel.MaybeInjectThinkingToken(messages, enableThinking: true, formatter).ToList();

        result.Should().HaveCount(2);
        result[0].Content.Should().Be("sys", because: "non-thinking formatters must leave messages unchanged");
    }

    [Fact]
    public void MaybeInjectThinkingToken_WithToolFragmentAlreadyInjected_TokenLeadsFragment()
    {
        // Simulates the real call order: MaybeInjectToolPromptFragment first, then MaybeInjectThinkingToken.
        var formatter = new Gemma4ChatFormatter();
        var schema = BuildSchema("""{ "type":"object", "properties":{ "q":{"type":"string"} }, "required":["q"] }""");
        var tools = new[] { new ChatToolDefinition("search", "search tool", schema) };
        var original = new[] { ChatMessage.User("search something") };

        // Step 1: tool fragment prepended as first system message
        var withFragment = LlamaServerGeneratorModel.MaybeInjectToolPromptFragment(original, tools, formatter).ToList();
        // Step 2: thinking token prepended to that system message
        var result = LlamaServerGeneratorModel.MaybeInjectThinkingToken(withFragment, enableThinking: true, formatter).ToList();

        result.Should().HaveCount(2);
        result[0].Role.Should().Be(ChatRole.System);
        result[0].Content.Should().StartWith("<|think|>",
            because: "thinking token must precede the tool fragment in the combined system message");
        result[0].Content.Should().Contain("Required parameters",
            because: "tool fragment content must be preserved");
    }
}

/// <summary>
/// Tests for MemoryEstimator.EstimateForGguf partial offload calculations.
/// </summary>
public class MemoryEstimatorPartialOffloadTests
{
    [Fact]
    public void EstimateForGguf_SmallVram_RecommendsPartialLayers()
    {
        // 8GB model, only 4GB VRAM available → partial offload
        var estimate = MemoryEstimator.EstimateForGguf(
            modelFileSizeBytes: 8L * 1024 * 1024 * 1024,
            contextLength: 4096,
            availableVramBytes: 4L * 1024 * 1024 * 1024,
            availableRamBytes: 32L * 1024 * 1024 * 1024);

        estimate.CanFitInVram.Should().BeFalse("8GB model + KV cache exceeds 4GB VRAM");
        estimate.RecommendedGpuLayers.Should().BeGreaterThan(0, "some layers should fit in 4GB");
        estimate.RecommendedGpuLayers.Should().BeLessThan(estimate.TotalLayers, "not all layers fit");
    }

    [Fact]
    public void EstimateForGguf_AmpleVram_RecommendsAllLayers()
    {
        // 3GB model, 24GB VRAM → all layers on GPU
        var estimate = MemoryEstimator.EstimateForGguf(
            modelFileSizeBytes: 3L * 1024 * 1024 * 1024,
            contextLength: 4096,
            availableVramBytes: 24L * 1024 * 1024 * 1024,
            availableRamBytes: 64L * 1024 * 1024 * 1024);

        estimate.CanFitInVram.Should().BeTrue();
        estimate.RecommendedGpuLayers.Should().Be(estimate.TotalLayers);
    }

    [Fact]
    public void EstimateForGguf_NoVram_RecommendsZeroLayers()
    {
        var estimate = MemoryEstimator.EstimateForGguf(
            modelFileSizeBytes: 3L * 1024 * 1024 * 1024,
            contextLength: 4096,
            availableVramBytes: null,
            availableRamBytes: 16L * 1024 * 1024 * 1024);

        estimate.RecommendedGpuLayers.Should().Be(0);
        estimate.CanFitInVram.Should().BeFalse();
    }

    [Fact]
    public void EstimateForGguf_LargeContext_IncreasesMemory()
    {
        var small = MemoryEstimator.EstimateForGguf(
            modelFileSizeBytes: 5L * 1024 * 1024 * 1024,
            contextLength: 4096,
            availableVramBytes: 8L * 1024 * 1024 * 1024);

        var large = MemoryEstimator.EstimateForGguf(
            modelFileSizeBytes: 5L * 1024 * 1024 * 1024,
            contextLength: 32768,
            availableVramBytes: 8L * 1024 * 1024 * 1024);

        // Larger context needs more memory, so may not fit while smaller does
        large.TotalMemoryBytes.Should().BeGreaterThanOrEqualTo(small.TotalMemoryBytes);
    }

    [Fact]
    public void EstimateForGguf_VramBudget_PartialOffload_LayerCountDecreases()
    {
        // Same model, decreasing VRAM → decreasing recommended layers
        var layers8gb = MemoryEstimator.EstimateForGguf(
            modelFileSizeBytes: 8L * 1024 * 1024 * 1024,
            contextLength: 4096,
            availableVramBytes: 8L * 1024 * 1024 * 1024).RecommendedGpuLayers;

        var layers4gb = MemoryEstimator.EstimateForGguf(
            modelFileSizeBytes: 8L * 1024 * 1024 * 1024,
            contextLength: 4096,
            availableVramBytes: 4L * 1024 * 1024 * 1024).RecommendedGpuLayers;

        layers4gb.Should().BeLessThanOrEqualTo(layers8gb,
            "less VRAM should recommend fewer or equal GPU layers");
    }

    [Fact]
    public void EstimateForGguf_GpuOffloadRatio_IsValid()
    {
        var estimate = MemoryEstimator.EstimateForGguf(
            modelFileSizeBytes: 5L * 1024 * 1024 * 1024,
            contextLength: 4096,
            availableVramBytes: 4L * 1024 * 1024 * 1024);

        estimate.GpuOffloadRatio.Should().BeInRange(0f, 1f);
    }
}
