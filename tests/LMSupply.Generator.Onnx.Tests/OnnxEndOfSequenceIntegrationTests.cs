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
    // Every cached ONNX generator, asked a question whose whole answer is known: the exact text, not its shape, is what a repeated or
    // dropped last token changes. The answers end in a number and in a word so both a digit piece and a word piece are covered.
    [Theory]
    [InlineData("microsoft/Phi-4-mini-instruct-onnx", "What is 2+2? Answer with just the number.", "4")]
    [InlineData("microsoft/Phi-4-mini-instruct-onnx", "Is Paris the capital of France? Answer yes or no.", "Yes")]
    [InlineData("microsoft/Phi-3.5-mini-instruct-onnx", "What is 2+2? Answer with just the number.", "4")]
    [InlineData("microsoft/Phi-3.5-mini-instruct-onnx", "Is Paris the capital of France? Answer yes or no.", "Yes")]
    public async Task A_short_answer_is_not_followed_by_a_copy_of_its_last_token(string modelId, string question, string answer)
    {
        OnnxGeneratorBackend.Register();
        await using var model = await LocalGenerator.LoadAsync(
            modelId, new GeneratorOptions { Provider = ExecutionProvider.Cpu }, cancellationToken: TestContext.Current.CancellationToken);

        var pieces = new List<string>();
        await foreach (var piece in model.GenerateChatAsync(
            [ChatMessage.System("Reply with only the answer. No explanation, no extra sentences."), ChatMessage.User(question)],
            new GenerationOptions { MaxTokens = 20, DoSample = false }, TestContext.Current.CancellationToken))
        {
            pieces.Add(piece);
        }

        string.Concat(pieces).Trim().TrimEnd('.').Should().Be(answer);
    }
}
