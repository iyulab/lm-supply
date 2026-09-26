using AwesomeAssertions;
using LMSupply.Generator.Abstractions;
using LMSupply.Generator.Models;
using NSubstitute;
using Xunit;

namespace LMSupply.Generator.Tests;

/// <summary>
/// The usage extensions return the backend's own result. <c>GenerateWithUsageAsync</c> has delegated to
/// <c>GenerateCompleteResultAsync</c> since 0.73.0. Its chat twin kept estimating from the streamed text: it reported
/// the estimate without <see cref="TokenUsage.IsEstimated"/>, missed a reasoning model's hidden tokens, and dropped the
/// finish reason and timings. The lm-supply console's chat endpoint served that estimate as if it were measured.
/// </summary>
public class GeneratorExtensionsUsageTests
{
    private static readonly GenerationResult ServerResult =
        new("hi", new TokenUsage(41, 512), "length") { Timings = new GenerationTimings { CompletionTokensPerSecond = 119.0 } };

    [Fact]
    public async Task GenerateChatWithUsageAsync_ReturnsTheBackendsResult_NotAnEstimate()
    {
        var generator = Substitute.For<ITextGenerator>();
        ChatMessage[] messages = [ChatMessage.User("hi")];
        generator.GenerateChatCompleteResultAsync(messages, Arg.Any<GenerationOptions?>(), Arg.Any<CancellationToken>())
            .Returns(ServerResult);

        var result = await generator.GenerateChatWithUsageAsync(messages, cancellationToken: TestContext.Current.CancellationToken);

        result.Should().Be(ServerResult);
        result.Usage.CompletionTokens.Should().Be(512, "the server's count includes reasoning the text never shows");
        result.Usage.IsEstimated.Should().BeFalse();
        result.FinishReason.Should().Be("length");
        result.Timings!.CompletionTokensPerSecond.Should().Be(119.0);
        generator.ReceivedCalls().Select(c => c.GetMethodInfo().Name)
            .Should().NotContain(nameof(ITextGenerator.GenerateChatAsync), "the text stream is not what the usage comes from");
    }

    [Fact]
    public async Task GenerateWithUsageAsync_ReturnsTheBackendsResult()
    {
        var generator = Substitute.For<ITextGenerator>();
        generator.GenerateCompleteResultAsync("p", Arg.Any<GenerationOptions?>(), Arg.Any<CancellationToken>())
            .Returns(ServerResult);

        var result = await generator.GenerateWithUsageAsync("p", cancellationToken: TestContext.Current.CancellationToken);

        result.Should().Be(ServerResult);
    }
}
