using AwesomeAssertions;
using LMSupply.Hardware;

namespace LMSupply.Core.Tests.Hardware;

public class SpeculativeDecodingModeTests
{
    [Theory]
    [InlineData(SpeculativeDecodingMode.Auto,       0)]
    [InlineData(SpeculativeDecodingMode.None,       1)]
    [InlineData(SpeculativeDecodingMode.Ngram,      2)]
    [InlineData(SpeculativeDecodingMode.DraftModel, 3)]
    public void SpeculativeDecodingMode_HasExpectedIntValue(SpeculativeDecodingMode mode, int expected)
    {
        ((int)mode).Should().Be(expected);
    }
}
