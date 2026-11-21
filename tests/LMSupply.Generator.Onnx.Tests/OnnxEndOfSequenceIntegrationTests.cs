using AwesomeAssertions;
using LMSupply.Generator.Models;

namespace LMSupply.Generator.Onnx.Tests;

/// <summary>
/// When the model emits its end token the sequence does not grow, so a loop that reads the sequence's last token after every step reads
/// the token before it a second time: «what is 2+2» came back as «44», «waterways.» as «waterways..». The end token is not output.
/// </summary>
[Trait("Category", "Integration")]
public class OnnxEndOfSequenceIntegrationTests
{
    private const string Phi4MiniModel = "microsoft/Phi-4-mini-instruct-onnx";

    [Fact]
    public async Task A_short_answer_is_not_followed_by_a_copy_of_its_last_token()
    {
        OnnxGeneratorBackend.Register();
        await using var model = await LocalGenerator.LoadAsync(
            Phi4MiniModel, new GeneratorOptions { Provider = ExecutionProvider.Cpu }, cancellationToken: TestContext.Current.CancellationToken);

        var pieces = new List<string>();
        await foreach (var piece in model.GenerateChatAsync(
            [ChatMessage.User("What is 2+2? Answer with just the number.")],
            new GenerationOptions { MaxTokens = 20, DoSample = false }, TestContext.Current.CancellationToken))
        {
            pieces.Add(piece);
        }

        string.Concat(pieces).Trim().Should().Be("4");
    }
}
