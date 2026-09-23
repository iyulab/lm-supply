using AwesomeAssertions;

namespace LMSupply.Generator.Onnx.Tests;

/// <summary>
/// The ONNX path reports why a generation ended. It used to answer <c>"stop"</c> for every non-tool completion,
/// so a response cut off at <see cref="LMSupply.Generator.Models.GenerationOptions.MaxTokens"/> looked finished.
/// </summary>
public sealed class FinishReasonTests
{
    [Theory]
    [InlineData(true, 10, 4096, "length")]   // stopped at the output token limit
    [InlineData(false, 4096, 4096, "length")] // generator done because the sequence filled the context
    [InlineData(false, 120, 4096, "stop")]    // end token (or a stop sequence) before any limit
    [InlineData(false, 120, 0, "stop")]       // no known maximum: only the output limit can say "length"
    public void ResolveFinishReason(bool reachedOutputLimit, int sequenceLength, int maxSequenceLength, string expected)
        => Internal.OnnxGeneratorModel.ResolveFinishReason(reachedOutputLimit, sequenceLength, maxSequenceLength).Should().Be(expected);
}
