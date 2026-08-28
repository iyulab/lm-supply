using AwesomeAssertions;
using LMSupply.Hardware;

namespace LMSupply.Core.Tests.Hardware;

public class KvCacheQuantizationTypeTests
{
    [Theory]
    [InlineData(KvCacheQuantizationType.Auto,  -1)]
    [InlineData(KvCacheQuantizationType.F16,    0)]
    [InlineData(KvCacheQuantizationType.Q8_0,   1)]
    [InlineData(KvCacheQuantizationType.Q4_0,   2)]
    [InlineData(KvCacheQuantizationType.F32,    3)]
    public void KvCacheQuantizationType_HasExpectedIntValue(KvCacheQuantizationType type, int expected)
    {
        ((int)type).Should().Be(expected);
    }
}
