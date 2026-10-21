using System.Text.Json;
using AwesomeAssertions;
using LMSupply.Generator.Abstractions;
using LMSupply.Generator.Models;
using Microsoft.Extensions.AI;
using NSubstitute;
using AIChatMessage = Microsoft.Extensions.AI.ChatMessage;
using AIChatRole = Microsoft.Extensions.AI.ChatRole;
using LmChatMessage = LMSupply.Generator.Models.ChatMessage;
using LmChatRole = LMSupply.Generator.Models.ChatRole;

namespace LMSupply.Generator.Tests;

/// <summary>
/// <see cref="LocalChatClient"/>: what a Microsoft.Extensions.AI caller sends reaches the generator as the library's own
/// messages and options, and what the generator answers comes back as the standard response, whole or streamed.
/// </summary>
public class LocalChatClientTests
{
    private const string Model = "test/model";

    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    private readonly ITextGenerator _generator = Substitute.For<ITextGenerator>();
    private List<LmChatMessage> _sent = [];
    private GenerationOptions _options = null!;

    public LocalChatClientTests()
    {
        _generator.ModelId.Returns(Model);
        Answer(new ChatCompletionResult { Content = "ok", FinishReason = "stop" });
    }

    private void Answer(ChatCompletionResult result) =>
        _generator.GenerateChatWithToolsAsync(Arg.Any<IEnumerable<LmChatMessage>>(), Arg.Any<GenerationOptions?>(), Arg.Any<CancellationToken>())
            .Returns(call =>
            {
                _sent = [.. call.Arg<IEnumerable<LmChatMessage>>()];
                _options = call.Arg<GenerationOptions?>()!;
                return result;
            });

    private void Stream(params ChatStreamChunk[] chunks) =>
        _generator.GenerateChatStreamAsync(Arg.Any<IEnumerable<LmChatMessage>>(), Arg.Any<GenerationOptions?>(), Arg.Any<CancellationToken>())
            .Returns(call =>
            {
                _sent = [.. call.Arg<IEnumerable<LmChatMessage>>()];
                _options = call.Arg<GenerationOptions?>()!;
                return chunks.ToAsyncEnumerable();
            });

    private Task<ChatResponse> AskAsync(IEnumerable<AIChatMessage> messages, ChatOptions? options = null) =>
        _generator.AsChatClient().GetResponseAsync(messages, options, Ct);

    private Task<ChatResponse> AskAsync(ChatOptions? options = null) => AskAsync([new(AIChatRole.User, "hi")], options);

    // ── what is sent ────────────────────────────────────────────────────────────────────────────────────

    [Fact]
    public async Task Messages_KeepTheirRoles_AndTheInstructionsComeFirstAsASystemMessage()
    {
        await AskAsync(
            [new(AIChatRole.System, "be brief"), new(AIChatRole.User, "hello"), new(AIChatRole.Assistant, "hi there")],
            new ChatOptions { Instructions = "answer in English" });

        _sent.Select(m => (m.Role, m.Content)).Should().Equal(
            (LmChatRole.System, "answer in English"),
            (LmChatRole.System, "be brief"),
            (LmChatRole.User, "hello"),
            (LmChatRole.Assistant, "hi there"));
    }

    [Fact]
    public async Task Images_AreSentAsImageParts_InTheOrderGiven()
    {
        var png = new DataContent(new byte[] { 1, 2, 3 }, "image/png");
        await AskAsync([new(AIChatRole.User, [new TextContent("what is this?"), png, new UriContent("https://example.org/a.jpg", "image/jpeg")])]);

        var message = _sent.Should().ContainSingle().Which;
        message.Content.Should().Be("what is this?");
        message.ContentParts.Should().HaveCount(3);
        message.ContentParts![0].Should().Be(new TextContentPart("what is this?"));
        ((ImageContentPart)message.ContentParts[1]).Url.Should().Be("data:image/png;base64,AQID");
        ((ImageContentPart)message.ContentParts[2]).Url.Should().Be("https://example.org/a.jpg");
    }

    [Fact]
    public async Task ToolCallsAndResults_RoundTripAsTheLibrarysToolMessages()
    {
        var call = new FunctionCallContent("call_1", "weather", new Dictionary<string, object?> { ["city"] = "Seoul" });
        await AskAsync(
        [
            new(AIChatRole.User, "weather?"),
            new(AIChatRole.Assistant, [call]),
            new(AIChatRole.Tool, [new FunctionResultContent("call_1", "sunny")]),
            new(AIChatRole.Tool, [new FunctionResultContent("call_2", new AIContent[] { new TextContent("a chart"), new DataContent(new byte[4], "image/png") })]),
        ]);

        _sent[1].Role.Should().Be(LmChatRole.Assistant);
        _sent[1].ToolCalls.Should().ContainSingle().Which.Should().Be(new ChatToolCall("call_1", "weather", """{"city":"Seoul"}"""));
        (_sent[2].Role, _sent[2].ToolCallId, _sent[2].Content).Should().Be((LmChatRole.Tool, "call_1", "sunny"));
        _sent[3].Content.Should().Be("a chart\n[image/png content left out (4 bytes): the model reads text only]");
    }

    [Fact]
    public async Task StandardOptions_MapToTheirNamesakes_AndMaxOutputTokensCapsTheGeneratedTokens()
    {
        await AskAsync(new ChatOptions
        {
            MaxOutputTokens = 64, Temperature = 0.2f, TopP = 0.5f, TopK = 7, FrequencyPenalty = 0.3f,
            PresencePenalty = 0.4f, Seed = 42, StopSequences = ["END"],
        });

        _options.MaxNewTokens.Should().Be(64);
        (_options.Temperature, _options.TopP, _options.TopK).Should().Be((0.2f, 0.5f, 7));
        (_options.FrequencyPenalty, _options.PresencePenalty, _options.Seed).Should().Be((0.3f, 0.4f, 42));
        _options.StopSequences.Should().Equal("END");
    }

    [Fact]
    public async Task OptionsWithNoStandardName_ComeFromTheRawRepresentation_WithTheStandardOnesAppliedOver()
    {
        var raw = new GenerationOptions { RepetitionPenalty = 1.3f, Grammar = "root ::= \"x\"", Temperature = 1.5f };
        await AskAsync(new ChatOptions { Temperature = 0.1f, RawRepresentationFactory = _ => raw });

        (_options.RepetitionPenalty, _options.Grammar, _options.Temperature).Should().Be((1.3f, "root ::= \"x\"", 0.1f));
        _options.Should().NotBeSameAs(raw, "the caller's options object is not changed");
        raw.Temperature.Should().Be(1.5f);
    }

    [Theory]
    [InlineData(null, ThinkingMode.Auto)]
    [InlineData(ReasoningEffort.None, ThinkingMode.Off)]
    [InlineData(ReasoningEffort.Low, ThinkingMode.On)]
    [InlineData(ReasoningEffort.High, ThinkingMode.On)]
    public async Task ReasoningEffort_DrivesThinking(ReasoningEffort? effort, ThinkingMode thinking)
    {
        await AskAsync(new ChatOptions { Reasoning = new ReasoningOptions { Effort = effort } });

        _options.Thinking.Should().Be(thinking);
        _options.ExtractReasoningTokens.Should().BeTrue();
    }

    [Fact]
    public async Task ReasoningOutputNone_StopsReasoningBeingAskedForOrReturned()
    {
        Answer(new ChatCompletionResult { Content = "42", Reasoning = "thinking...", FinishReason = "stop" });

        var response = await AskAsync(new ChatOptions { Reasoning = new ReasoningOptions { Output = ReasoningOutput.None } });

        _options.ExtractReasoningTokens.Should().BeFalse();
        response.Messages.Single().Contents.Should().ContainSingle().Which.Should().BeOfType<TextContent>();
    }

    [Fact]
    public async Task Tools_AreOfferedWithTheirSchema_AndTheToolModeIsTheToolChoice()
    {
        var function = AIFunctionFactory.CreateDeclaration("weather", "The weather in a city.",
            JsonDocument.Parse("""{"type":"object","properties":{"city":{"type":"string"}}}""").RootElement);

        await AskAsync(new ChatOptions { Tools = [function], ToolMode = ChatToolMode.RequireSpecific("weather") });

        var tool = _options.Tools.Should().ContainSingle().Which;
        (tool.Name, tool.Description).Should().Be(("weather", "The weather in a city."));
        tool.Parameters!.Value.GetProperty("properties").GetProperty("city").GetProperty("type").GetString().Should().Be("string");
        _options.ToolChoice!.Mode.Should().Be(ToolChoiceMode.Function);
        _options.ToolChoice.FunctionName.Should().Be("weather");

        await AskAsync(new ChatOptions { Tools = [function], ToolMode = ChatToolMode.None });
        _options.ToolChoice.Should().BeSameAs(ToolChoice.None);
    }

    [Fact]
    public async Task AJsonResponseFormat_ConstrainsTheOutputToItsSchema_OrToAnObject()
    {
        var schema = JsonDocument.Parse("""{"type":"object","properties":{"a":{"type":"integer"}}}""").RootElement;

        await AskAsync(new ChatOptions { ResponseFormat = ChatResponseFormat.ForJsonSchema(schema) });
        JsonDocument.Parse(_options.JsonSchema!).RootElement.GetProperty("properties").GetProperty("a").GetProperty("type")
            .GetString().Should().Be("integer");

        await AskAsync(new ChatOptions { ResponseFormat = ChatResponseFormat.Json });
        _options.JsonSchema.Should().Be("""{"type":"object"}""");

        await AskAsync(new ChatOptions { ResponseFormat = ChatResponseFormat.Text });
        _options.JsonSchema.Should().BeNull();
    }

    [Fact]
    public async Task AModelIdThatNamesAnotherModel_IsRefused()
    {
        var act = () => AskAsync(new ChatOptions { ModelId = "other/model" });

        (await act.Should().ThrowAsync<ArgumentException>()).WithMessage("*'test/model'*'other/model'*");
        await AskAsync(new ChatOptions { ModelId = "TEST/model" });
    }

    // ── what comes back ─────────────────────────────────────────────────────────────────────────────────

    [Fact]
    public async Task AResponse_CarriesReasoningTextToolCallsFinishReasonUsageAndTheRawResult()
    {
        var result = new ChatCompletionResult
        {
            Reasoning = "let me see",
            Content = "calling",
            ToolCalls = [new ChatToolCall("call_9", "weather", """{"city":"Busan"}""")],
            FinishReason = "tool_calls",
            Usage = new ChatTokenUsage { PromptTokens = 10, CompletionTokens = 5, TotalTokens = 15 },
            Timings = new GenerationTimings { CachedPromptTokens = 8 },
        };
        Answer(result);

        var response = await AskAsync();

        var contents = response.Messages.Single().Contents;
        contents[0].Should().BeOfType<TextReasoningContent>().Which.Text.Should().Be("let me see");
        contents[1].Should().BeOfType<TextContent>().Which.Text.Should().Be("calling");
        var call = contents[2].Should().BeOfType<FunctionCallContent>().Which;
        (call.CallId, call.Name).Should().Be(("call_9", "weather"));
        call.Arguments!["city"].Should().BeOfType<JsonElement>().Which.GetString().Should().Be("Busan");
        response.FinishReason.Should().Be(ChatFinishReason.ToolCalls);
        response.ModelId.Should().Be(Model);
        (response.Usage!.InputTokenCount, response.Usage.OutputTokenCount, response.Usage.TotalTokenCount, response.Usage.CachedInputTokenCount)
            .Should().Be((10L, 5L, 15L, 8L));
        response.RawRepresentation.Should().BeSameAs(result);
    }

    [Fact]
    public async Task ToolCallArgumentsThatAreNotJson_ComeBackAsACallWithNoArguments()
    {
        Answer(new ChatCompletionResult { ToolCalls = [new ChatToolCall("c", "f", "{not json")], FinishReason = "tool_calls" });

        var response = await AskAsync();

        response.Messages.Single().Contents.OfType<FunctionCallContent>().Single().Arguments.Should().BeNull();
    }

    [Fact]
    public async Task AToolCallWithoutAnId_IsGivenOne_AndOneWithoutAName_IsLeftOut()
    {
        Answer(new ChatCompletionResult
        {
            ToolCalls = [new ChatToolCall("", "weather", "{}"), new ChatToolCall("c", "", "{}")],
            FinishReason = "tool_calls",
        });

        var response = await AskAsync();

        var call = response.Messages.Single().Contents.OfType<FunctionCallContent>().Should().ContainSingle().Which;
        call.Name.Should().Be("weather");
        call.CallId.Should().StartWith("call_").And.HaveLength("call_".Length + 32);
    }

    [Fact]
    public async Task AStreamedToolCallWithoutAnId_IsGivenOne_AndOneWithoutAName_IsLeftOut()
    {
        Stream(
            new ChatStreamChunk { ToolCalls = [new ChatToolCallDelta { Index = 0, Name = "weather", Arguments = "{}" }] },
            new ChatStreamChunk { ToolCalls = [new ChatToolCallDelta { Index = 1, Id = "nameless", Arguments = "{}" }] },
            new ChatStreamChunk { FinishReason = "tool_calls" });

        var updates = await _generator.AsChatClient().GetStreamingResponseAsync([new(AIChatRole.User, "hi")], cancellationToken: Ct).ToListAsync(Ct);

        var call = updates[^1].Contents.OfType<FunctionCallContent>().Should().ContainSingle().Which;
        call.Name.Should().Be("weather");
        call.CallId.Should().StartWith("call_");
    }

    [Fact]
    public async Task AStream_YieldsTextAndReasoningAsTheyCome_AndTheAssembledToolCallsAndUsageAtTheEnd()
    {
        Stream(
            new ChatStreamChunk { ReasoningDelta = "hmm" },
            new ChatStreamChunk { Text = "Hel" },
            new ChatStreamChunk { Text = "lo" },
            new ChatStreamChunk { ToolCalls = [new ChatToolCallDelta { Index = 1, Id = "b", Name = "second", Arguments = "{}" }] },
            new ChatStreamChunk { ToolCalls = [new ChatToolCallDelta { Index = 0, Id = "a", Name = "first", Arguments = """{"x":""" }] },
            new ChatStreamChunk { ToolCalls = [new ChatToolCallDelta { Index = 0, Arguments = "1}" }] },
            new ChatStreamChunk
            {
                FinishReason = "tool_calls",
                Usage = new ChatTokenUsage { PromptTokens = 3, CompletionTokens = 4, TotalTokens = 7 },
            });

        var updates = await _generator.AsChatClient().GetStreamingResponseAsync([new(AIChatRole.User, "hi")], cancellationToken: Ct).ToListAsync(Ct);

        updates.Select(u => u.Text).Take(3).Should().Equal("", "Hel", "lo");
        updates.Should().AllSatisfy(u => u.ModelId.Should().Be(Model));
        var last = updates[^1];
        last.FinishReason.Should().Be(ChatFinishReason.ToolCalls);
        last.Contents.OfType<FunctionCallContent>().Select(c => (c.CallId, c.Name)).Should().Equal(("a", "first"), ("b", "second"));
        last.Contents.OfType<FunctionCallContent>().First().Arguments!["x"].Should().BeOfType<JsonElement>().Which.GetInt32().Should().Be(1);

        var response = updates.ToChatResponse();
        response.Text.Should().Be("Hello");
        response.Usage!.TotalTokenCount.Should().Be(7);
        response.Messages.SelectMany(m => m.Contents).OfType<TextReasoningContent>().Single().Text.Should().Be("hmm");
    }

    [Fact]
    public async Task AStreamThatEndsWithoutAFinishReason_StillHandsOverItsToolCalls()
    {
        Stream(new ChatStreamChunk { ToolCalls = [new ChatToolCallDelta { Index = 0, Id = "a", Name = "f", Arguments = "{}" }] });

        var updates = await _generator.AsChatClient().GetStreamingResponseAsync([new(AIChatRole.User, "hi")], cancellationToken: Ct).ToListAsync(Ct);

        var last = updates.Should().ContainSingle().Which;
        last.FinishReason.Should().Be(ChatFinishReason.ToolCalls);
        last.Contents.OfType<FunctionCallContent>().Single().Name.Should().Be("f");
    }

    // ── the client ──────────────────────────────────────────────────────────────────────────────────────

    [Fact]
    public void TheClient_ReportsItsProviderAndModel_HandsOutItsGenerator_AndDoesNotDisposeIt()
    {
        var client = _generator.AsChatClient();

        var metadata = client.GetService<ChatClientMetadata>()!;
        (metadata.ProviderName, metadata.DefaultModelId).Should().Be(("LMSupply", Model));
        client.GetService<ITextGenerator>().Should().BeSameAs(_generator);
        client.GetService<IChatClient>().Should().BeSameAs(client);
        client.GetService(typeof(ITextGenerator), serviceKey: "keyed").Should().BeNull();

        client.Dispose();
        _generator.ReceivedCalls().Should().NotContain(c => c.GetMethodInfo().Name == nameof(IAsyncDisposable.DisposeAsync));
    }

    [Fact]
    public void ALoadedModel_IsHandedOutAsItself_SoACallerReadsItsContextLength()
    {
        var model = Substitute.For<IGeneratorModel>();
        model.ModelId.Returns(Model);
        model.MaxContextLength.Returns(32768);

        var client = model.AsChatClient();

        client.GetService<IGeneratorModel>()!.MaxContextLength.Should().Be(32768);
        _generator.AsChatClient().GetService<IGeneratorModel>().Should().BeNull();
    }
}
