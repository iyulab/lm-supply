using AwesomeAssertions;
using LMSupply.Hardware;

namespace LMSupply.Core.Tests.Hardware;

public class RopeScalingModeTests
{
    [Theory]
    [InlineData(RopeScalingMode.Default,  0)]
    [InlineData(RopeScalingMode.Linear,   1)]
    [InlineData(RopeScalingMode.YaRN,     2)]
    [InlineData(RopeScalingMode.LongRoPE, 3)]
    public void RopeScalingMode_HasExpectedIntValue(RopeScalingMode mode, int expected)
    {
        ((int)mode).Should().Be(expected);
    }
}
