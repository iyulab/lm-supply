using System.ComponentModel;
using System.Text.Json;
using AwesomeAssertions;
using LMSupply.Exceptions;
using LMSupply.Generator;
using Microsoft.Extensions.AI;
using Xunit;

namespace LMSupply.Integration.Tests.Functional;

/// <summary>
/// A GGUF model through <see cref="TextGeneratorChatClientExtensions.AsChatClient"/>: an answer, a streamed answer, and a tool call —
/// whole and streamed — that arrives with its name and parsed arguments. The streamed call is the case that a dropped
/// tool-call channel loses while the turn still ends with the <c>tool_calls</c> finish reason. A model that is not in the
/// local model cache is skipped, not downloaded.
/// </summary>
[Trait("Category", "Integration")]
[Trait("Category", "LocalOnly")]
[Trait("Domain", "Generator")]
public sealed class LocalChatClientLiveTests
{
    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    public static TheoryData<string> Models { get; } = ["gguf:gemma4-default", "gguf:qwen3-fast"];

    [Description("Gets the current weather for a city.")]
    private static string GetWeather([Description("The city name")] string city) => $"sunny in {city}";

    private static ChatOptions Options(int maxOutputTokens, params AITool[] tools) => new()
    {
        MaxOutputTokens = maxOutputTokens,
        Reasoning = new ReasoningOptions { Effort = ReasoningEffort.None },
        Tools = tools.Length > 0 ? [.. tools] : null,
    };

    [Theory]
    [MemberData(nameof(Models))]
    public async Task Answers_WholeAndStreamed(string model)
    {
        await using var generator = await LoadCachedAsync(model);
        using var client = generator.AsChatClient();

        var whole = await client.GetResponseAsync("Count from 1 to 5, separated by spaces.", Options(60), Ct);
        var streamed = (await client.GetStreamingResponseAsync("Count from 1 to 5, separated by spaces.", Options(60), Ct).ToListAsync(Ct))
            .ToChatResponse();

        foreach (var response in new[] { whole, streamed })
        {
            response.Text.Should().Contain("1").And.Contain("5");
            response.FinishReason.Should().Be(ChatFinishReason.Stop);
            response.ModelId.Should().Be(generator.ModelId);
            response.Usage!.OutputTokenCount.Should().BePositive();
        }
    }

    [Theory]
    [MemberData(nameof(Models))]
    public async Task AToolCall_ArrivesWithItsNameAndArguments_WholeAndStreamed(string model)
    {
        await using var generator = await LoadCachedAsync(model);
        using var client = generator.AsChatClient();
        var options = Options(200, AIFunctionFactory.Create(GetWeather));
        const string prompt = "What is the weather in Busan right now? Use the tool.";

        var whole = await client.GetResponseAsync(prompt, options, Ct);
        var streamed = (await client.GetStreamingResponseAsync(prompt, options, Ct).ToListAsync(Ct)).ToChatResponse();

        foreach (var (how, response) in new[] { ("whole", whole), ("streamed", streamed) })
        {
            var call = response.Messages.SelectMany(m => m.Contents).OfType<FunctionCallContent>()
                .Should().ContainSingle($"the {how} response asks for the tool once").Which;
            call.Name.Should().Be(nameof(GetWeather));
            call.CallId.Should().NotBeNullOrEmpty();
            var city = call.Arguments!["city"].Should().BeOfType<JsonElement>().Which.GetString();
            city.Should().Contain("Busan", $"the {how} call carries its argument");
            response.FinishReason.Should().Be(ChatFinishReason.ToolCalls);
        }
    }

    private static async Task<LMSupply.Generator.Abstractions.IGeneratorModel> LoadCachedAsync(string model)
    {
        try
        {
            return await LocalGenerator.LoadAsync(model, new GeneratorOptions { DisableAutoDownload = true }, cancellationToken: Ct);
        }
        catch (ModelNotFoundException ex)
        {
            Assert.Skip($"Not in the local model cache: {ex.Message}");
            throw;
        }
    }
}
