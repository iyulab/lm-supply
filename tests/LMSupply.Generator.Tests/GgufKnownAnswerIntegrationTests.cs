using System.Text;
using AwesomeAssertions;
using LMSupply.Generator.Models;

namespace LMSupply.Generator.Tests;

/// <summary>
/// The exact text of a short, known answer through the GGUF (llama-server) path. A check that only asks for «not empty» or «contains 8»
/// lets a repeated, dropped or leaked token through — the ONNX path shipped a doubled last token that way. Needs the model in the HuggingFace
/// cache and a cached llama-server; skipped in CI with the other Integration tests.
/// </summary>
[Trait("Category", "Integration")]
[Trait("Axis", "KnownAnswer")]
public class GgufKnownAnswerIntegrationTests
{
    [Theory(Timeout = 180_000)]
    [InlineData("What is 2+2? Answer with just the number.", "4")]
    [InlineData("Is Paris the capital of France? Answer yes or no.", "Yes")]
    public async Task A_short_answer_is_exactly_the_answer(string question, string answer)
    {
        await using var model = await LocalGenerator.LoadAsync("gguf:gemma4-default", cancellationToken: TestContext.Current.CancellationToken);

        var text = new StringBuilder();
        await foreach (var piece in model.GenerateChatAsync(
            [ChatMessage.System("Reply with only the answer. No explanation, no extra sentences."), ChatMessage.User(question)],
            new GenerationOptions { Temperature = 0f, MaxTokens = 512 }, TestContext.Current.CancellationToken))
        {
            text.Append(piece);
        }

        text.ToString().Trim().TrimEnd('.').Should().Be(answer);
    }
}
