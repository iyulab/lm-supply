using AwesomeAssertions;
using LMSupply.Generator.Models;

namespace LMSupply.Generator.Onnx.Tests;

/// <summary>
/// On a real ONNX model, a completion cut off at <see cref="GenerationOptions.MaxTokens"/> reports
/// <c>"length"</c> on both result-bearing paths; before, the ONNX path answered <c>"stop"</c> for every
/// non-tool completion.
/// </summary>
[Trait("Category", "Integration")]
public class OnnxFinishReasonIntegrationTests
{
    private const string Phi35Model = "microsoft/Phi-3.5-mini-instruct-onnx";

    [Fact]
    public async Task A_completion_cut_off_at_MaxTokens_reports_length_on_both_result_paths()
    {
        OnnxGeneratorBackend.Register();
        await using var model = await LocalGenerator.LoadAsync(
            Phi35Model, new GeneratorOptions { Provider = ExecutionProvider.Cpu }, cancellationToken: TestContext.Current.CancellationToken);
        var messages = new[] { ChatMessage.User("Count from one to fifty in words, separated by commas.") };

        var cut = await model.GenerateChatWithToolsAsync(
            messages, new GenerationOptions { MaxTokens = 8, DoSample = false }, TestContext.Current.CancellationToken);
        cut.FinishReason.Should().Be("length", "eight tokens cannot hold fifty numbers");

        string? streamed = null;
        await foreach (var chunk in model.GenerateChatStreamAsync(
            messages, new GenerationOptions { MaxTokens = 8, DoSample = false }, TestContext.Current.CancellationToken))
        {
            streamed = chunk.FinishReason ?? streamed;
        }
        streamed.Should().Be("length");

        var whole = await model.GenerateChatWithToolsAsync(
            [ChatMessage.User("Reply with the single word: yes")],
            new GenerationOptions { MaxTokens = 64, DoSample = false }, TestContext.Current.CancellationToken);
        whole.FinishReason.Should().Be("stop", "a short answer ends on the model's end token");
    }
}
