using AwesomeAssertions;
using LMSupply.Transcriber.Decoding;

namespace LMSupply.Transcriber.Tests;

/// <summary>
/// <see cref="TranscribeOptions.MaxTokens"/> was declared and documented but never read: every window
/// decoded up to the model's 448-token context whatever the caller asked for. These pin how the
/// option now bounds the decode loop.
/// </summary>
public class WhisperDecoderTokenLimitTests
{
    private const int Context = 448;
    private const int Prompt = 4;

    [Fact]
    public void Default_IsTheModelContext()
    {
        WhisperDecoder.ComputeTokenLimit(Context, Prompt, options: null).Should().Be(Context);
        WhisperDecoder.ComputeTokenLimit(Context, Prompt, new TranscribeOptions()).Should().Be(Context);
    }

    [Fact]
    public void MaxTokens_BoundsTheGeneratedTokens_AfterThePrompt()
    {
        WhisperDecoder.ComputeTokenLimit(Context, Prompt, new TranscribeOptions { MaxTokens = 20 })
            .Should().Be(Prompt + 20, "the limit counts generated tokens, not the prompt");
    }

    [Fact]
    public void MaxTokens_AboveTheContext_IsCappedByIt()
    {
        WhisperDecoder.ComputeTokenLimit(Context, Prompt, new TranscribeOptions { MaxTokens = 10_000 })
            .Should().Be(Context);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-5)]
    public void MaxTokens_BelowOne_IsRejected(int maxTokens)
    {
        var act = () => WhisperDecoder.ComputeTokenLimit(Context, Prompt, new TranscribeOptions { MaxTokens = maxTokens });

        act.Should().Throw<ArgumentOutOfRangeException>().WithMessage("*MaxTokens*");
    }
}
