using AwesomeAssertions;
using LMSupply.Generator;
using LMSupply.Generator.Models;
using Xunit;

namespace LMSupply.Integration.Tests.Functional;

/// <summary>
/// Regression guard for the gemma4-E2B empty chat response at small token budget
/// (issue: ...gemma4-e2b-empty-chat-response-small-token-budget.md).
///
/// Root cause (confirmed by a Text-vs-ReasoningDelta sweep on RTX 4060, 2026-06-21):
/// gemma4-E2B emits reasoning_content by default (Thinking.Auto). At MaxTokens=30 the reasoning
/// channel consumes the whole budget (~89 reasoning chars, 27 chunks, text=0, finish=length) so
/// GenerateChatCompleteAsync returns "". Thinking.Off (enable_thinking=false) recovers a direct
/// answer in the same budget; Auto at 256 tokens also answers ("Your name is Alice.").
///
/// Guarantees encoded here:
///   1. Thinking.Off at a tight budget (30) returns a non-empty answer (the consumer remedy).
///   2. Thinking.Auto at a reasonable budget (256) returns a non-empty answer (acceptance #1:
///      "reasonable budget works"). 30 tokens for a thinking model is genuinely too small; we do
///      NOT assert Auto+30 returns text — that would mean defaulting gemma4 to Off, breaking the
///      Auto = preserve-model-default contract.
///
/// Run with: dotnet test --filter "FullyQualifiedName~Gemma4EmptyChatProbe"
/// </summary>
[Trait("Category", "Integration")]
[Trait("Category", "LocalOnly")]
public class Gemma4EmptyChatProbeTests
{
    private const string FastModel = "gguf:gemma4-fast";

    private static ChatMessage[] MultiTurn() =>
    [
        ChatMessage.System("You are a helpful assistant."),
        ChatMessage.User("My name is Alice."),
        ChatMessage.Assistant("Nice to meet you, Alice!"),
        ChatMessage.User("What is my name?"),
    ];

    [Fact]
    public async Task ThinkingOff_TightBudget_ReturnsNonEmptyAnswer()
    {
        await using var model = await LocalGenerator.LoadAsync(FastModel, cancellationToken: TestContext.Current.CancellationToken);

        // Temperature=0 (greedy) makes the Contain("Alice") assertion deterministic; the
        // NotBeNullOrEmpty guard is the actual regression check (the bug was an empty string).
        var options = new GenerationOptions
        {
            MaxTokens = 30,
            Thinking = ThinkingMode.Off,
            Temperature = 0f,
            DoSample = false,
        };
        var result = await model.GenerateChatCompleteAsync(MultiTurn(), options, TestContext.Current.CancellationToken);

        result.Should().NotBeNullOrEmpty(
            "Thinking.Off skips the reasoning block so a 30-token budget yields answer text");
        result.Should().Contain("Alice", "the answer should recall the name from earlier turns");
    }

    [Fact]
    public async Task ThinkingAuto_ReasonableBudget_ReturnsNonEmptyAnswer()
    {
        await using var model = await LocalGenerator.LoadAsync(FastModel, cancellationToken: TestContext.Current.CancellationToken);

        var options = new GenerationOptions { MaxTokens = 256, Thinking = ThinkingMode.Auto };
        var result = await model.GenerateChatCompleteAsync(MultiTurn(), options, TestContext.Current.CancellationToken);

        result.Should().NotBeNullOrEmpty(
            "a reasonable budget leaves room for both reasoning and a final answer");
    }
}
