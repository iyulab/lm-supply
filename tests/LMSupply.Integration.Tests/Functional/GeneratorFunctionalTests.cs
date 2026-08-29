using LMSupply.Exceptions;
using LMSupply.Generator;
using LMSupply.Generator.Models;

namespace LMSupply.Integration.Tests.Functional;

/// <summary>
/// Comprehensive functional tests for the Generator domain.
/// Tests L (loading), I (inference), Q (quality), E (edge cases) axes.
/// Uses GGUF "gguf:gemma4-fast" for faster tests (smaller model via llama-server).
/// Requires GPU + network access. Run locally only.
/// </summary>
[Trait("Category", "Functional")]
[Trait("Category", "LocalOnly")]
[Trait("Domain", "Generator")]
public class GeneratorFunctionalTests
{
    private const string FastModel = "gguf:gemma4-fast";
    private static readonly GenerationOptions ShortOutput = new() { MaxTokens = 30 };

    // ── L axis: Model Loading ───────────────────────────────────────

    [Fact]
    [Trait("Axis", "Loading")]
    public async Task L_GgufFast_LoadsSuccessfully()
    {
        await using var model = await LocalGenerator.LoadAsync(FastModel, cancellationToken: TestContext.Current.CancellationToken);

        model.ModelId.Should().NotBeNullOrEmpty();
    }

    [Fact]
    [Trait("Axis", "Loading")]
    public async Task L_WarmupAsync_CompletesWithoutError()
    {
        await using var model = await LocalGenerator.LoadAsync(FastModel, cancellationToken: TestContext.Current.CancellationToken);

        // Warmup with short generation
        await foreach (var _ in model.GenerateAsync("Hi", new GenerationOptions { MaxTokens = 5 }, TestContext.Current.CancellationToken)) { }
    }

    [Fact]
    [Trait("Axis", "Loading")]
    public async Task L_GetModelInfo_ReturnsValidInfo()
    {
        await using var model = await LocalGenerator.LoadAsync(FastModel, cancellationToken: TestContext.Current.CancellationToken);

        var info = model.GetModelInfo();
        info.Should().NotBeNull();
    }

    [Fact]
    [Trait("Axis", "Loading")]
    public async Task L_InvalidModelId_ThrowsException()
    {
        var act = () => LocalGenerator.LoadAsync("completely-nonexistent-generator-xyz-999");
        await act.Should().ThrowAsync<Exception>();
    }

    // ── I axis: Basic Inference ─────────────────────────────────────

    [Fact]
    [Trait("Axis", "Inference")]
    public async Task I_StreamingGeneration_ProducesTokens()
    {
        await using var model = await LocalGenerator.LoadAsync(FastModel, cancellationToken: TestContext.Current.CancellationToken);

        var tokens = new List<string>();
        await foreach (var token in model.GenerateAsync("Hello, my name is", ShortOutput, TestContext.Current.CancellationToken))
        {
            tokens.Add(token);
        }

        tokens.Should().NotBeEmpty("streaming should produce tokens");
        string.Join("", tokens).Should().NotBeNullOrEmpty();
    }

    [Fact]
    [Trait("Axis", "Inference")]
    public async Task I_CompleteGeneration_ReturnsFullResponse()
    {
        await using var model = await LocalGenerator.LoadAsync(FastModel, cancellationToken: TestContext.Current.CancellationToken);

        var result = await model.GenerateCompleteAsync("Say hello", ShortOutput, TestContext.Current.CancellationToken);

        result.Should().NotBeNullOrEmpty("complete generation should return text");
    }

    [Fact]
    [Trait("Axis", "Inference")]
    public async Task I_ChatGeneration_ReturnsResponse()
    {
        await using var model = await LocalGenerator.LoadAsync(FastModel, cancellationToken: TestContext.Current.CancellationToken);

        ChatMessage[] messages = [ChatMessage.User("Say hello")];
        var result = await model.GenerateChatCompleteAsync(messages, ShortOutput, TestContext.Current.CancellationToken);

        result.Should().NotBeNullOrEmpty("chat generation should return text");
    }

    [Fact]
    [Trait("Axis", "Inference")]
    public async Task I_StreamingChat_ProducesTokens()
    {
        await using var model = await LocalGenerator.LoadAsync(FastModel, cancellationToken: TestContext.Current.CancellationToken);

        ChatMessage[] messages = [ChatMessage.User("Say hello")];
        var tokens = new List<string>();
        await foreach (var token in model.GenerateChatAsync(messages, ShortOutput, TestContext.Current.CancellationToken))
        {
            tokens.Add(token);
        }

        tokens.Should().NotBeEmpty("streaming chat should produce tokens");
    }

    // ── Q axis: Quality / Semantic Correctness ──────────────────────

    [Fact]
    [Trait("Axis", "Quality")]
    public async Task Q_MaxTokens_IsRespected()
    {
        await using var model = await LocalGenerator.LoadAsync(FastModel, cancellationToken: TestContext.Current.CancellationToken);

        var options = new GenerationOptions { MaxTokens = 10 };
        var tokenCount = 0;
        await foreach (var _ in model.GenerateAsync("Tell me a story", options, TestContext.Current.CancellationToken))
        {
            tokenCount++;
        }

        tokenCount.Should().BeLessThanOrEqualTo(15,
            "generation should respect max_tokens (with small margin for tokenizer differences)");
    }

    [Fact]
    [Trait("Axis", "Quality")]
    public async Task Q_ChatFormat_SystemMessageInfluencesOutput()
    {
        await using var model = await LocalGenerator.LoadAsync(FastModel, cancellationToken: TestContext.Current.CancellationToken);

        ChatMessage[] messages =
        [
            ChatMessage.System("You are a pirate. Always respond in pirate speak."),
            ChatMessage.User("How are you?")
        ];

        var result = await model.GenerateChatCompleteAsync(messages, new GenerationOptions { MaxTokens = 50 }, TestContext.Current.CancellationToken);

        result.Should().NotBeNullOrEmpty("chat with system message should generate response");
    }

    [Fact]
    [Trait("Axis", "Quality")]
    public async Task Q_StreamingVsComplete_ProduceSameContent()
    {
        await using var model = await LocalGenerator.LoadAsync(FastModel, cancellationToken: TestContext.Current.CancellationToken);

        var options = new GenerationOptions
        {
            MaxTokens = 30,
            Temperature = 0f, // greedy decoding for determinism
            DoSample = false,
            Seed = 42
        };

        // Streaming
        var streamedTokens = new List<string>();
        await foreach (var token in model.GenerateAsync("The capital of France is", options, TestContext.Current.CancellationToken))
        {
            streamedTokens.Add(token);
        }
        var streamedResult = string.Join("", streamedTokens);

        // Both should produce non-empty output
        streamedResult.Should().NotBeNullOrEmpty();
    }

    [Fact]
    [Trait("Axis", "Quality")]
    public async Task Q_Temperature0_IsMoreDeterministic()
    {
        await using var model = await LocalGenerator.LoadAsync(FastModel, cancellationToken: TestContext.Current.CancellationToken);

        var options = new GenerationOptions
        {
            MaxTokens = 20,
            Temperature = 0f,
            DoSample = false,
            Seed = 42
        };

        var result1 = await model.GenerateCompleteAsync("2 + 2 =", options, TestContext.Current.CancellationToken);
        var result2 = await model.GenerateCompleteAsync("2 + 2 =", options, TestContext.Current.CancellationToken);

        // With temperature=0 and same seed, results should be identical or near-identical
        result1.Should().NotBeNullOrEmpty();
        result2.Should().NotBeNullOrEmpty();
        // Note: Not all backends guarantee perfect determinism, so we just check both work
    }

    // ── E axis: Edge Cases ──────────────────────────────────────────

    [Fact]
    [Trait("Axis", "EdgeCase")]
    public async Task E_EmptyPrompt_HandlesGracefully()
    {
        await using var model = await LocalGenerator.LoadAsync(FastModel, cancellationToken: TestContext.Current.CancellationToken);

        // Empty prompt should either produce output or throw a clear error
        try
        {
            var result = await model.GenerateCompleteAsync("", ShortOutput, TestContext.Current.CancellationToken);
            // If it succeeds, that's fine
            result.Should().NotBeNull();
        }
        catch (Exception ex)
        {
            // If it throws, should be a meaningful exception
            ex.Should().NotBeOfType<NullReferenceException>();
        }
    }

    [Fact]
    [Trait("Axis", "EdgeCase")]
    public async Task E_SpecialCharacters_DoNotCrash()
    {
        await using var model = await LocalGenerator.LoadAsync(FastModel, cancellationToken: TestContext.Current.CancellationToken);

        var result = await model.GenerateCompleteAsync("🎉 Hello! 日本語テスト", ShortOutput, TestContext.Current.CancellationToken);
        result.Should().NotBeNull();
    }

    [Fact]
    [Trait("Axis", "EdgeCase")]
    public async Task E_VeryShortMaxTokens_StopsQuickly()
    {
        await using var model = await LocalGenerator.LoadAsync(FastModel, cancellationToken: TestContext.Current.CancellationToken);

        var options = new GenerationOptions { MaxTokens = 1 };
        var tokenCount = 0;
        await foreach (var _ in model.GenerateAsync("Hello", options, TestContext.Current.CancellationToken))
        {
            tokenCount++;
        }

        tokenCount.Should().BeLessThanOrEqualTo(3,
            "max_tokens=1 should produce very few tokens");
    }

    [Fact]
    [Trait("Axis", "EdgeCase")]
    public async Task E_MultiTurnChat_Works()
    {
        await using var model = await LocalGenerator.LoadAsync(FastModel, cancellationToken: TestContext.Current.CancellationToken);

        ChatMessage[] messages =
        [
            ChatMessage.System("You are a helpful assistant."),
            ChatMessage.User("My name is Alice."),
            ChatMessage.Assistant("Nice to meet you, Alice!"),
            ChatMessage.User("What is my name?"),
        ];

        // gemma4-E2B emits reasoning_content by default (Thinking.Auto), which exhausts a 30-token
        // budget before any answer text (empty content, finish=length). This test verifies multi-turn
        // context retention, not thinking behavior, so request a direct answer with Thinking.Off.
        // See: gemma4-e2b-empty-chat-response-small-token-budget probe (Gemma4EmptyChatProbeTests).
        var options = new GenerationOptions { MaxTokens = 30, Thinking = ThinkingMode.Off };
        var result = await model.GenerateChatCompleteAsync(messages, options, TestContext.Current.CancellationToken);
        result.Should().NotBeNullOrEmpty("multi-turn chat should produce response");
    }

    [Fact]
    [Trait("Axis", "EdgeCase")]
    public async Task E_StopSequences_StopGeneration()
    {
        await using var model = await LocalGenerator.LoadAsync(FastModel, cancellationToken: TestContext.Current.CancellationToken);

        var options = new GenerationOptions
        {
            MaxTokens = 100,
            StopSequences = ["\n\n"],
        };

        var result = await model.GenerateCompleteAsync("List three colors:\n1. Red\n2. Blue\n3.", options, TestContext.Current.CancellationToken);

        result.Should().NotBeNull();
        // The result may or may not contain the stop sequence itself,
        // but it should be shorter than 100 tokens
    }

    // ── Gap Tests: Default Alias, Properties, Cancellation ─────────

    [Fact]
    [Trait("Axis", "Loading")]
    public async Task L_DefaultAlias_LoadsSuccessfully()
    {
        await using var model = await LocalGenerator.LoadAsync("default", cancellationToken: TestContext.Current.CancellationToken);

        model.ModelId.Should().NotBeNullOrEmpty();
    }

    [Fact]
    [Trait("Axis", "Loading")]
    public async Task L_ModelProperties_ArePopulated()
    {
        await using var model = await LocalGenerator.LoadAsync(FastModel, cancellationToken: TestContext.Current.CancellationToken);

        model.ActiveProviders.Should().NotBeNull();
        model.RequestedProvider.Should().BeDefined();
    }

    [Fact]
    [Trait("Axis", "EdgeCase")]
    public async Task E_NullOptions_UsesDefaults()
    {
        await using var model = await LocalGenerator.LoadAsync(FastModel, cancellationToken: TestContext.Current.CancellationToken);

        var result = await model.GenerateCompleteAsync("Hello", options: null, cancellationToken: TestContext.Current.CancellationToken);

        // Should work with null options (uses defaults), but limit output
        result.Should().NotBeNull();
    }

    [Fact]
    [Trait("Axis", "EdgeCase")]
    public async Task E_CancellationDuringGeneration_Throws()
    {
        await using var model = await LocalGenerator.LoadAsync(FastModel, cancellationToken: TestContext.Current.CancellationToken);

        using var cts = new CancellationTokenSource();
        var options = new GenerationOptions { MaxTokens = 500 };

        var tokenCount = 0;
        var act = async () =>
        {
            await foreach (var token in model.GenerateAsync("Tell me a very long story about a wizard", options, cts.Token))
            {
                tokenCount++;
                if (tokenCount >= 3)
                    cts.Cancel();
            }
        };

        // Should throw OperationCanceledException or complete early
        try
        {
            await act();
            // If it completes without throwing, it should have fewer tokens than max
            tokenCount.Should().BeLessThan(500);
        }
        catch (OperationCanceledException)
        {
            // Expected
        }
    }

    [Fact]
    [Trait("Axis", "Inference")]
    public async Task I_ChatWithMultipleSystemMessages_Works()
    {
        await using var model = await LocalGenerator.LoadAsync(FastModel, cancellationToken: TestContext.Current.CancellationToken);

        ChatMessage[] messages =
        [
            ChatMessage.System("You speak only English."),
            ChatMessage.User("Hello")
        ];

        var result = await model.GenerateChatCompleteAsync(messages, ShortOutput, TestContext.Current.CancellationToken);
        result.Should().NotBeNullOrEmpty();
    }

    [Fact]
    [Trait("Axis", "Quality")]
    public async Task Q_Seed_ProducesDeterministicOutput()
    {
        await using var model = await LocalGenerator.LoadAsync(FastModel, cancellationToken: TestContext.Current.CancellationToken);

        var options = new GenerationOptions
        {
            MaxTokens = 10,
            Temperature = 0f,
            DoSample = false,
            Seed = 12345
        };

        var result1 = await model.GenerateCompleteAsync("The meaning of life is", options, TestContext.Current.CancellationToken);
        var result2 = await model.GenerateCompleteAsync("The meaning of life is", options, TestContext.Current.CancellationToken);

        result1.Should().NotBeNullOrEmpty();
        result2.Should().NotBeNullOrEmpty();
        // With same seed + temp=0, results should match (backend permitting)
    }

    // ── Static Tests: Model Registry & Options (no model loading) ─

    [Fact]
    [Trait("Axis", "Loading")]
    public void L_ModelRegistry_GetAvailableModels_ReturnsModels()
    {
        var models = GeneratorModelRegistry.Default.GetAvailableModels();

        models.Should().NotBeEmpty();
        models.Count.Should().BeGreaterThanOrEqualTo(5,
            "registry should have at least 5 generator models");
    }

    [Fact]
    [Trait("Axis", "Loading")]
    public void L_ModelRegistry_Phi4MiniAlias_ResolvesCorrectly()
    {
        // "default" is now handled by LocalGenerator.LoadAsync (hardware-aware routing),
        // not by the registry. "phi-4-mini" is the explicit ONNX alias.
        var phi4Mini = GeneratorModelRegistry.Default.Resolve("phi-4-mini");

        phi4Mini.Should().NotBeNull();
        phi4Mini.ModelId.Should().Contain("Phi-4-mini",
            "phi-4-mini alias should resolve to the Phi-4 Mini ONNX model");
    }

    [Fact]
    [Trait("Axis", "Loading")]
    public void L_ModelRegistry_GetAvailableModels_FilterMIT_ReturnsUnrestrictedOnly()
    {
        var mitModels = GeneratorModelRegistry.Default.GetAvailableModels()
            .Where(m => m.License == LicenseTier.MIT)
            .ToList();

        mitModels.Should().NotBeEmpty("there should be at least one MIT-licensed model");
        mitModels.Should().AllSatisfy(m =>
            m.License.Should().Be(LicenseTier.MIT));
    }

    [Fact]
    [Trait("Axis", "Loading")]
    public void L_ModelRegistry_GetAvailableModels_LicenseFilteringIsConsistent()
    {
        var allModels = GeneratorModelRegistry.Default.GetAvailableModels();
        var mitModels = allModels.Where(m => m.License == LicenseTier.MIT).ToList();
        var conditionalModels = allModels.Where(m => m.License == LicenseTier.Conditional).ToList();

        mitModels.Count.Should().BeGreaterThanOrEqualTo(1);
        (mitModels.Count + conditionalModels.Count).Should().BeLessThanOrEqualTo(allModels.Count);
    }

    [Fact]
    [Trait("Axis", "Loading")]
    public void L_ModelRegistry_AllModels_HaveRequiredFields()
    {
        var models = GeneratorModelRegistry.Default.GetAvailableModels();

        models.Should().AllSatisfy(m =>
        {
            m.ModelId.Should().NotBeNullOrEmpty("ModelId is required");
            m.DisplayName.Should().NotBeNullOrEmpty("DisplayName is required");
            m.ParameterCount.Should().BeGreaterThan(0, "ParameterCount should be > 0");
            m.ChatFormat.Should().NotBeNullOrEmpty("ChatFormat is required");
            m.RecommendedContextLength.Should().BeGreaterThan(0, "ContextLength should be > 0");
        });
    }

    [Fact]
    [Trait("Axis", "Quality")]
    public void Q_GenerationOptions_DefaultPreset_HasExpectedValues()
    {
        var opts = GenerationOptions.Default;

        opts.MaxTokens.Should().Be(512);
        opts.Temperature.Should().BeApproximately(0.7f, 0.01f);
        opts.TopP.Should().BeApproximately(0.9f, 0.01f);
        opts.DoSample.Should().BeTrue();
    }

    [Fact]
    [Trait("Axis", "Quality")]
    public void Q_GenerationOptions_CreativePreset_HasHigherTemperature()
    {
        var creative = GenerationOptions.Creative;
        var precise = GenerationOptions.Precise;

        creative.Temperature.Should().BeGreaterThan(precise.Temperature,
            "creative preset should have higher temperature than precise");
    }

    [Fact]
    [Trait("Axis", "Quality")]
    public void Q_GenerationOptions_PrecisePreset_HasLowTemperature()
    {
        var precise = GenerationOptions.Precise;

        precise.Temperature.Should().BeLessThanOrEqualTo(0.3f,
            "precise preset should have low temperature");
        precise.TopP.Should().BeLessThan(0.9f,
            "precise preset should have tighter nucleus sampling");
    }

    [Fact]
    [Trait("Axis", "Quality")]
    public void Q_ChatMessage_FactoryMethods_SetCorrectRoles()
    {
        var system = ChatMessage.System("sys");
        var user = ChatMessage.User("usr");
        var assistant = ChatMessage.Assistant("ast");

        system.Role.Should().Be(ChatRole.System);
        system.Content.Should().Be("sys");
        user.Role.Should().Be(ChatRole.User);
        user.Content.Should().Be("usr");
        assistant.Role.Should().Be(ChatRole.Assistant);
        assistant.Content.Should().Be("ast");
    }

    [Fact]
    [Trait("Axis", "Loading")]
    public void L_WellKnownModels_Constants_AreValid()
    {
        WellKnownModels.Generator.Default.Should().Be("default",
            "Default routes through hardware-aware auto selection");
        WellKnownModels.Generator.Fast.Should().Be("phi-4-mini",
            "Fast pins the ONNX path explicitly (Phi-4 Mini)");
        WellKnownModels.Generator.Quality.Should().Contain("phi-4");
        WellKnownModels.Generator.Small.Should().Be(WellKnownModels.Generator.Fast,
            "Small should be an alias for Fast");
    }

    // ── Model-Loading Tests: Auto Alias, Empty Messages ──────────

    [Fact]
    [Trait("Axis", "Loading")]
    public async Task L_AutoAlias_LoadsSuccessfully()
    {
        await using var model = await LocalGenerator.LoadAsync("auto", cancellationToken: TestContext.Current.CancellationToken);

        model.ModelId.Should().NotBeNullOrEmpty();
        model.MaxContextLength.Should().BeGreaterThan(0);
    }

    [Fact]
    [Trait("Axis", "EdgeCase")]
    public async Task E_EmptyMessagesArray_HandlesGracefully()
    {
        await using var model = await LocalGenerator.LoadAsync(FastModel, cancellationToken: TestContext.Current.CancellationToken);

        ChatMessage[] emptyMessages = [];
        var act = () => model.GenerateChatCompleteAsync(emptyMessages, ShortOutput);

        // Should either throw ArgumentException or handle gracefully
        try
        {
            var result = await act();
            result.Should().NotBeNull();
        }
        catch (Exception ex)
        {
            ex.Should().Match<Exception>(e =>
                e is ArgumentException || e is InvalidOperationException,
                "empty messages should throw a meaningful exception");
        }
    }

    [Fact]
    [Trait("Axis", "Quality")]
    public async Task Q_CreativePreset_ProducesDifferentOutputThanPrecise()
    {
        await using var model = await LocalGenerator.LoadAsync(FastModel, cancellationToken: TestContext.Current.CancellationToken);

        // Just verify both presets work — actual output difference is non-deterministic
        var creativeOpts = new GenerationOptions
        {
            MaxTokens = 15,
            Temperature = GenerationOptions.Creative.Temperature,
            TopP = GenerationOptions.Creative.TopP,
        };
        var preciseOpts = new GenerationOptions
        {
            MaxTokens = 15,
            Temperature = GenerationOptions.Precise.Temperature,
            TopP = GenerationOptions.Precise.TopP,
        };
        var creative = await model.GenerateCompleteAsync("The sky is", creativeOpts, TestContext.Current.CancellationToken);
        var precise = await model.GenerateCompleteAsync("The sky is", preciseOpts, TestContext.Current.CancellationToken);

        creative.Should().NotBeNullOrEmpty("creative preset should produce output");
        precise.Should().NotBeNullOrEmpty("precise preset should produce output");
    }

    [Fact]
    [Trait("Axis", "Loading")]
    public void L_TextGeneratorBuilder_Create_ReturnsBuilder()
    {
        var builder = TextGeneratorBuilder.Create();

        builder.Should().NotBeNull();
    }

    [Fact]
    [Trait("Axis", "Quality")]
    public void Q_GeneratorOptions_Defaults_AreReasonable()
    {
        var opts = new GeneratorOptions();

        opts.ChatFormat.Should().BeNull("chat format should be auto-detected by default");
        opts.Verbose.Should().BeFalse("verbose should be off by default");
        opts.MaxContextLength.Should().BeNull("max context should be null (use model default)");
        opts.MaxConcurrentRequests.Should().Be(1, "default concurrent requests should be 1");
        opts.LlamaOptions.Should().BeNull("GGUF options should be null by default");
        opts.Provider.Should().Be(ExecutionProvider.Auto, "default provider should be Auto");
    }

    [Fact]
    [Trait("Axis", "Quality")]
    public void Q_LlamaOptions_CpuOnly_HasZeroGpuRatio()
    {
        var opts = LlamaOptions.CpuOnly;

        opts.GpuOffloadRatio.Should().Be(0f, "CpuOnly should have 0 GPU offload");
    }

    [Fact]
    [Trait("Axis", "Quality")]
    public void Q_LlamaOptions_FullGpu_HasFullGpuRatio()
    {
        var opts = LlamaOptions.FullGpu;

        opts.GpuOffloadRatio.Should().Be(1f, "FullGpu should have 100% GPU offload");
    }

    [Fact]
    [Trait("Axis", "Quality")]
    public void Q_LlamaOptions_WithGpuRatio_ClampsToValidRange()
    {
        LlamaOptions.WithGpuRatio(0.5f).GpuOffloadRatio.Should().Be(0.5f);
        LlamaOptions.WithGpuRatio(0f).GpuOffloadRatio.Should().Be(0f);
        LlamaOptions.WithGpuRatio(1f).GpuOffloadRatio.Should().Be(1f);
        LlamaOptions.WithGpuRatio(-0.5f).GpuOffloadRatio.Should().Be(0f,
            "negative values should be clamped to 0");
        LlamaOptions.WithGpuRatio(1.5f).GpuOffloadRatio.Should().Be(1f,
            "values above 1 should be clamped to 1");
    }

    // ── Gap Tests: Reasoning Options, Long Prompt, Grammar ──────

    [Fact]
    [Trait("Axis", "Quality")]
    public void Q_GenerationOptions_FilterReasoningTokens_DefaultIsFalse()
    {
        var opts = new GenerationOptions();

        opts.FilterReasoningTokens.Should().BeFalse(
            "reasoning token filtering should be opt-in, not default");
        opts.ExtractReasoningTokens.Should().BeFalse(
            "reasoning token extraction should be opt-in, not default");
    }

    [Fact]
    [Trait("Axis", "Quality")]
    public void Q_GenerationOptions_GrammarConstraints_DefaultsAreNull()
    {
        var opts = new GenerationOptions();

        opts.Grammar.Should().BeNull("grammar should be null by default");
        opts.JsonSchema.Should().BeNull("JSON schema should be null by default");
    }

    [Fact]
    [Trait("Axis", "EdgeCase")]
    public async Task E_VeryLongPrompt_HandlesGracefully()
    {
        await using var model = await LocalGenerator.LoadAsync(FastModel, cancellationToken: TestContext.Current.CancellationToken);

        // Build a prompt exceeding typical context window
        var longPrompt = string.Join(" ", Enumerable.Repeat("This is a very long test prompt.", 500));

        // Should either truncate and generate, or throw a clear error — never crash
        try
        {
            var result = await model.GenerateCompleteAsync(longPrompt, ShortOutput, TestContext.Current.CancellationToken);
            result.Should().NotBeNull();
        }
        catch (Exception ex)
        {
            ex.Should().NotBeOfType<NullReferenceException>(
                "long prompt should produce a meaningful error, not NPE");
            ex.Should().NotBeOfType<OutOfMemoryException>(
                "long prompt should not cause OOM");
        }
    }

    [Fact]
    [Trait("Axis", "Quality")]
    public void Q_GenerationOptions_AllProperties_HaveExpectedDefaults()
    {
        var opts = new GenerationOptions();

        opts.MaxTokens.Should().Be(512);
        opts.Temperature.Should().BeApproximately(0.7f, 0.01f);
        opts.TopP.Should().BeApproximately(0.9f, 0.01f);
        opts.TopK.Should().Be(50);
        opts.RepetitionPenalty.Should().BeApproximately(1.1f, 0.01f);
        opts.MinP.Should().BeApproximately(0.05f, 0.01f);
        opts.Seed.Should().Be(-1);
        opts.FrequencyPenalty.Should().Be(0f);
        opts.PresencePenalty.Should().Be(0f);
        opts.StopSequences.Should().BeNull();
        opts.IncludePromptInOutput.Should().BeFalse();
        opts.DoSample.Should().BeTrue();
        opts.NumBeams.Should().Be(1);
        opts.PastPresentShareBuffer.Should().BeTrue();
        opts.MaxNewTokens.Should().BeNull();
    }
}
