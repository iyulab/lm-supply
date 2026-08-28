using System.Text;
using System.Text.Json;
using AwesomeAssertions;
using LMSupply.Generator.Abstractions;
using LMSupply.Generator.ChatFormatters;
using LMSupply.Generator.Internal.Llama;
using LMSupply.Generator.Models;
using LMSupply.Llama.Server;
using Xunit;
using Xunit.Abstractions;

namespace LMSupply.Generator.Tests;

// ============================================================
// Part B — Static validation (no model downloads required).
//
// Verifies that every registered Gemma 4 alias wires up the correct
// chat formatter, tool-call stream parser, and version gate, so that
// inference + tool-call plumbing is correct before any live test runs.
// ============================================================
public class Gemma4StaticValidationTests
{
    // All aliases in the registry whose models use the Gemma 4 chat format.
    // xlarge (Qwen3.5) is intentionally excluded — it uses chatml.
    private static readonly string[] Gemma4Aliases =
        ["gguf:gemma4-fast", "gguf:gemma4-default", "gguf:gemma4-balanced", "gguf:gemma4-quality", "gguf:gemma4-large"];

    [Theory]
    [InlineData("gguf:gemma4-fast")]
    [InlineData("gguf:gemma4-default")]
    [InlineData("gguf:gemma4-balanced")]
    [InlineData("gguf:gemma4-quality")]
    [InlineData("gguf:gemma4-large")]
    public void GgufAlias_ChatFormat_IsGemma4(string alias)
    {
        var model = GgufModelRegistry.Resolve(alias);

        model.Should().NotBeNull();
        model!.ChatFormat.Should().Be("gemma4",
            because: $"{alias} must route through Gemma4ChatFormatter for native tool-call and reasoning support");
    }

    [Fact]
    public void ChatFormatterFactory_CreateByFormat_Gemma4_ReturnsGemma4Formatter()
    {
        var formatter = ChatFormatterFactory.CreateByFormat("gemma4");

        formatter.Should().BeOfType<Gemma4ChatFormatter>();
        formatter.FormatName.Should().Be("gemma4");
    }

    [Theory]
    [InlineData("gemma4")]
    [InlineData("gemma-4")]
    [InlineData("Gemma4")]   // case-insensitive
    public void ChatFormatterFactory_CreateByFormat_AcceptsCaseVariants(string formatName)
    {
        var formatter = ChatFormatterFactory.CreateByFormat(formatName);

        formatter.Should().BeOfType<Gemma4ChatFormatter>();
    }

    [Fact]
    public void Gemma4ChatFormatter_CreateToolCallStreamParser_ReturnsNonNull()
    {
        // CreateToolCallStreamParser() must return a live parser so that
        // LlamaServerGeneratorModel routes text-channel tool-call wrappers through
        // Gemma4ToolCallStreamParser (ecosystem ISSUE D-5, 2026-05-01).
        var formatter = new Gemma4ChatFormatter();

        var parser = formatter.CreateToolCallStreamParser();

        parser.Should().NotBeNull(
            because: "Gemma 4 emits tool calls in a text-channel wrapper that only Gemma4ToolCallStreamParser can extract");
    }

    [Fact]
    public void Gemma4ChatFormatter_CreateToolCallStreamParser_IsGemma4Parser()
    {
        var formatter = new Gemma4ChatFormatter();

        var parser = formatter.CreateToolCallStreamParser();

        parser.Should().BeOfType<Gemma4ToolCallStreamParser>(
            because: "the concrete type must be Gemma4ToolCallStreamParser for the LlamaServerGeneratorModel suppression path");
    }

    [Theory]
    [InlineData("b8994")] // installed build on this machine
    [InlineData("b8672")] // minimum required build
    [InlineData("b9000")] // future build
    public void LlamaServerVersionRequirements_MeetsMinimum_DoesNotThrow(string version)
    {
        // b8994 is the version installed at %LocalAppData%\LMSupply\cache\llama-server\.
        // Validate() must pass silently.
        var act = () => LlamaServerVersionRequirements.Validate(version, "gemma4");

        act.Should().NotThrow(
            because: $"{version} >= b8672 (minimum for gemma4 native support)");
    }

    [Theory]
    [InlineData("b8671")]
    [InlineData("b1000")]
    [InlineData("b0001")]
    public void LlamaServerVersionRequirements_BelowMinimum_ThrowsInvalidOperation(string version)
    {
        var act = () => LlamaServerVersionRequirements.Validate(version, "gemma4");

        act.Should().Throw<InvalidOperationException>(
            because: $"{version} < b8672 (Gemma 4 minimum)");
    }

    [Fact]
    public void LlamaServerVersionRequirements_NullVersion_DoesNotThrow()
    {
        // Graceful degradation: if version can't be parsed, let inference proceed.
        var act = () => LlamaServerVersionRequirements.Validate(null, "gemma4");

        act.Should().NotThrow();
    }

    [Fact]
    public void Gemma4ChatFormatter_FormatPrompt_ProducesTurnTokens()
    {
        var formatter = new Gemma4ChatFormatter();
        var messages = new[]
        {
            ChatMessage.System("You are a helpful assistant."),
            ChatMessage.User("Hello")
        };

        var prompt = formatter.FormatPrompt(messages);

        // Gemma 4 uses <|turn> / <turn|> (new format), not <start_of_turn> (Gemma 2).
        prompt.Should().Contain("<|turn>system");
        prompt.Should().Contain("<|turn>user");
        // Final token opens the model's generation slot.
        prompt.Should().EndWith("<|turn>model\n");
    }

    [Fact]
    public void Gemma4ChatFormatter_GetThinkingToken_ReturnsThinkToken()
    {
        var formatter = new Gemma4ChatFormatter();

        var token = formatter.GetThinkingToken();

        token.Should().Be("<|think|>",
            because: "Gemma 4 activates thinking mode via <|think|> at the start of the system prompt");
    }

    [Fact]
    public void OtherFormatters_GetThinkingToken_ReturnsNull()
    {
        // Default interface method — non-Gemma formatters must return null.
        IChatFormatter phi3 = new Phi3ChatFormatter();
        IChatFormatter llama3 = new Llama3ChatFormatter();

        phi3.GetThinkingToken().Should().BeNull();
        llama3.GetThinkingToken().Should().BeNull();
    }

    // ---- W1: Gemma 4 tool-use risk advisory path -----------------------------------------

    [Fact]
    public void Gemma4ChatFormatter_IsGemma4FormatterInstance_EvaluatesTrue()
    {
        // LlamaServerGeneratorModel.LoadAsync uses `chatFormatter is Gemma4ChatFormatter`
        // to gate the W1 Trace.TraceWarning advisory (llama.cpp #21375 / #21882).
        // Verify the type-check holds for the formatter produced by the factory.
        IChatFormatter formatter = ChatFormatterFactory.CreateByFormat("gemma4");

        (formatter is Gemma4ChatFormatter).Should().BeTrue(
            because: "LoadAsync W1 path requires Gemma4ChatFormatter identity check to fire; " +
                     "if the factory changes the concrete type the warning will silently stop emitting");
    }

    [Theory]
    [InlineData("llama3")]
    [InlineData("chatml")]
    [InlineData("phi3")]
    [InlineData("qwen")]
    public void NonGemma4Formatters_AreNotGemma4Instance(string format)
    {
        // W1 warning must NOT fire for other model families.
        IChatFormatter formatter = ChatFormatterFactory.CreateByFormat(format);

        (formatter is Gemma4ChatFormatter).Should().BeFalse(
            because: $"{format} formatter must not trigger the Gemma 4 W1 advisory");
    }

    [Fact]
    public void Gemma4ChatFormatter_RenderToolPromptFragment_ListsRequiredParams()
    {
        var formatter = new Gemma4ChatFormatter();
        var parameters = JsonSerializer.Deserialize<JsonElement>(
            """
            {
                "type": "object",
                "properties": {
                    "location": { "type": "string" },
                    "unit": { "type": "string" }
                },
                "required": ["location"]
            }
            """);
        var tools = new[] { new ChatToolDefinition("get_weather", "Get weather", parameters) };

        var fragment = formatter.RenderToolPromptFragment(tools);

        fragment.Should().NotBeNull();
        fragment.Should().Contain("get_weather");
        // Must list required params to work around E4B empty-args bug (ISSUE D-1).
        fragment.Should().Contain("Required parameters (MUST be provided):");
        fragment.Should().Contain("location (string)");
    }

    [Fact]
    public void Gemma4ChatFormatter_RenderToolPromptFragment_NullTools_ReturnsNull()
    {
        var formatter = new Gemma4ChatFormatter();

        var fragment = formatter.RenderToolPromptFragment(null);

        fragment.Should().BeNull();
    }

    [Fact]
    public void AllGemma4Aliases_ChatFormatterFactory_ResolvesToGemma4()
    {
        // Confirm that every Gemma 4 alias in the registry yields a model whose
        // ChatFormat maps to Gemma4ChatFormatter via CreateByFormat.
        foreach (var alias in Gemma4Aliases)
        {
            var model = GgufModelRegistry.Resolve(alias)!;
            var formatter = ChatFormatterFactory.CreateByFormat(model.ChatFormat);
            formatter.Should().BeOfType<Gemma4ChatFormatter>(
                because: $"{alias} has ChatFormat='{model.ChatFormat}' which must map to Gemma4ChatFormatter");
        }
    }
}

// ============================================================
// Part A — Live inference validation.
//
// Requires actual model files in the HuggingFace cache and a running
// llama-server binary. Skipped in CI; run locally with:
//   dotnet test --filter "Category=Integration&ClassName=Gemma4LiveInferenceTests"
//
// Environment on this machine:
//   GPU: RTX 4060 Laptop (4 GB VRAM)
//   llama-server: b8994 (CUDA12, %LocalAppData%\LMSupply\cache\llama-server\)
//   HF cache: E2B Q4_K_M (2963 MB), E4B Q4_K_M (5088 MB)
//
// gguf:gemma4-fast (E2B, ~3 GB) fits fully in VRAM.
// gguf:gemma4-default (E4B, ~5 GB) requires partial CPU offload on 4 GB VRAM.
// ============================================================
[Trait("Category", "Integration")]
public class Gemma4LiveInferenceTests(ITestOutputHelper output)
{
    // Gemma 4 uses thinking tokens (reasoning_content) before answering.
    // MaxTokens must be large enough to cover thinking (~100-200 tokens) + answer.
    private static GenerationOptions MakeGemma4Options(int maxTokens = 512) => new()
    {
        Temperature = 1.0f,
        TopP = 0.95f,
        TopK = 64,
        RepetitionPenalty = 1.0f,
        MaxTokens = maxTokens,
    };

    private static readonly ChatToolDefinition WeatherTool = new(
        "get_weather",
        "Return current weather for a city.",
        JsonSerializer.Deserialize<JsonElement>(
            """
            {
                "type": "object",
                "properties": {
                    "city": { "type": "string", "description": "City name" }
                },
                "required": ["city"]
            }
            """));

    // ---- gguf:gemma4-fast (E2B Q4_K_M, ~3 GB) ------------------------------------------------

    [Fact(Timeout = 120_000)]
    public async Task GgufFast_Chat_ReturnsCoherentResponse()
    {
        await using var model = await LocalGenerator.LoadAsync("gguf:gemma4-fast");

        var messages = new[]
        {
            ChatMessage.System("You are a helpful assistant. Answer in one sentence."),
            ChatMessage.User("What is 3 + 5?")
        };

        var result = new StringBuilder();
        await foreach (var token in model.GenerateChatAsync(messages, MakeGemma4Options()))
        {
            result.Append(token);
        }

        output.WriteLine($"[gguf:gemma4-fast chat] → {result}");

        result.ToString().Should().NotBeNullOrWhiteSpace(
            because: "E2B must produce a non-empty response to a simple arithmetic question");
        result.ToString().Should().Contain("8",
            because: "3 + 5 = 8 — a minimal coherence check");
    }

    [Fact(Timeout = 120_000)]
    public async Task GgufFast_ToolCall_InvokesGetWeather()
    {
        await using var model = await LocalGenerator.LoadAsync("gguf:gemma4-fast");

        var messages = new[]
        {
            ChatMessage.User("What is the weather in Seoul right now?")
        };

        var options = MakeGemma4Options(512);
        options.Tools = [WeatherTool];

        var result = await model.GenerateChatWithToolsAsync(messages, options);

        output.WriteLine($"[gguf:gemma4-fast tool] finish={result.FinishReason} " +
                         $"hasTools={result.HasToolCalls} content={result.Content}");

        if (result.HasToolCalls)
        {
            result.ToolCalls![0].FunctionName.Should().Be("get_weather",
                because: "the model must call get_weather when asked about weather");
            var args = result.ToolCalls![0].Arguments;
            args.Should().Contain("city", because: "required param 'city' must be present");
            output.WriteLine($"  → tool call args: {args}");
        }
        else
        {
            // E2B Q4_K_M has lower tool-call compliance — record but don't fail.
            output.WriteLine("  ⚠ No tool call produced (within expected E2B Q4_K_M variance).");
        }
    }

    [Fact(Timeout = 180_000)]
    public async Task GgufFast_ToolCall_ComplianceRate_AtLeast60Percent()
    {
        // Runs the tool-call prompt 5 times and measures how many produce a
        // structured tool call. E2B at Q4_K_M is known to be less reliable;
        // ≥60% success is the acceptance threshold for this tier.
        // Adjust if empirically too tight or loose.
        await using var model = await LocalGenerator.LoadAsync("gguf:gemma4-fast");

        var options = MakeGemma4Options(512);
        options.Tools = [WeatherTool];

        const int runs = 5;
        int successes = 0;

        for (int i = 0; i < runs; i++)
        {
            var messages = new[]
            {
                ChatMessage.User("What is the weather in Tokyo right now?")
            };
            var result = await model.GenerateChatWithToolsAsync(messages, options);
            var ok = result.HasToolCalls && result.ToolCalls![0].FunctionName == "get_weather";
            if (ok) successes++;
            output.WriteLine($"  run {i + 1}/{runs}: {(ok ? "✓ tool call" : "✗ no/wrong tool call")} " +
                             $"finish={result.FinishReason}");
        }

        var rate = (double)successes / runs;
        output.WriteLine($"[gguf:gemma4-fast compliance] {successes}/{runs} = {rate:P0}");

        rate.Should().BeGreaterThanOrEqualTo(0.60,
            because: "E2B Q4_K_M must call tools successfully ≥60% of the time (ISSUE G1 baseline)");
    }

    // ---- gguf:gemma4-default (E4B Q4_K_M, ~5 GB, partial CPU offload on 4 GB VRAM) ----------

    [Fact(Timeout = 300_000)]
    public async Task GgufDefault_Chat_ReturnsCoherentResponse()
    {
        // E4B requires CPU offload on 4 GB VRAM; inference is slower.
        await using var model = await LocalGenerator.LoadAsync("gguf:gemma4-default");

        var messages = new[]
        {
            ChatMessage.System("You are a helpful assistant. Answer in one sentence."),
            ChatMessage.User("What is the capital of France?")
        };

        var result = new StringBuilder();
        await foreach (var token in model.GenerateChatAsync(messages, MakeGemma4Options()))
        {
            result.Append(token);
        }

        output.WriteLine($"[gguf:gemma4-default chat] → {result}");

        result.ToString().Should().NotBeNullOrWhiteSpace();
        result.ToString().ToLowerInvariant().Should().Contain("paris",
            because: "E4B must answer a basic factual question correctly");
    }

    [Fact(Timeout = 300_000)]
    public async Task GgufDefault_ToolCall_InvokesGetWeather()
    {
        await using var model = await LocalGenerator.LoadAsync("gguf:gemma4-default");

        var messages = new[]
        {
            ChatMessage.User("What is the weather in Paris right now?")
        };

        var options = MakeGemma4Options(512);
        options.Tools = [WeatherTool];

        var result = await model.GenerateChatWithToolsAsync(messages, options);

        output.WriteLine($"[gguf:gemma4-default tool] finish={result.FinishReason} " +
                         $"hasTools={result.HasToolCalls} content={result.Content}");

        result.HasToolCalls.Should().BeTrue(
            because: "E4B Q4_K_M must reliably invoke tools for a straightforward weather query");
        result.ToolCalls![0].FunctionName.Should().Be("get_weather");

        var args = result.ToolCalls![0].Arguments;
        args.Should().Contain("city", because: "required param 'city' must appear in the JSON args");
        output.WriteLine($"  → tool call args: {args}");
    }
}
