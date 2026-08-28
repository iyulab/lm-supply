using AwesomeAssertions;
using LMSupply.Hardware;

namespace LMSupply.Core.Tests.Hardware;

public class ModelMemoryEstimatorTests
{
    private const long ThreeBillion = 3_000_000_000L;
    private const long OneGB = 1_073_741_824L;

    [Fact]
    public void EstimateModelSize_FromParameters_Q4KM()
    {
        // 3B × 0.5625 × 1.1 = 1,856,250,000
        var result = ModelMemoryEstimator.EstimateModelSizeBytes(ThreeBillion, "Q4_K_M");

        var expected = (long)(ThreeBillion * 0.5625 * 1.1);
        result.Should().Be(expected);
        result.Should().BeInRange(1_855_000_000L, 1_857_000_000L);
    }

    [Fact]
    public void EstimateModelSize_FromParameters_FP16()
    {
        // 3B × 2.0 × 1.1 = 6,600,000,000
        var result = ModelMemoryEstimator.EstimateModelSizeBytes(ThreeBillion, "fp16");

        var expected = (long)(ThreeBillion * 2.0 * 1.1);
        result.Should().Be(expected);
        result.Should().BeInRange(6_599_000_000L, 6_601_000_000L);
    }

    [Fact]
    public void EstimateModelSize_FromParameters_FP32Default()
    {
        // 3B × 4.0 × 1.1 = 13,200,000,000
        var result = ModelMemoryEstimator.EstimateModelSizeBytes(ThreeBillion, null);

        var expected = (long)(ThreeBillion * 4.0 * 1.1);
        result.Should().Be(expected);
        result.Should().BeInRange(13_199_000_000L, 13_201_000_000L);
    }

    [Fact]
    public void EstimateModelSize_WithKnownSize_ReturnsKnownSize()
    {
        long knownSize = (long)(2.5 * OneGB);

        var result = ModelMemoryEstimator.EstimateModelSizeBytes(
            ThreeBillion, "Q4_K_M", knownSizeBytes: knownSize);

        result.Should().Be(knownSize);
    }

    [Fact]
    public void EstimateKvCacheBytes_F16()
    {
        var result = ModelMemoryEstimator.EstimateKvCacheBytes(
            contextLength: 4096,
            numLayers: 32,
            hiddenSize: 4096,
            kvQuantType: KvCacheQuantizationType.F16);

        // 2 × 32 × 4096 × 4096 × 2.0 = 2,147,483,648 (2 GB)
        result.Should().BeGreaterThan(0);
        result.Should().BeLessThan(4L * OneGB);
        result.Should().Be(2L * 32 * 4096 * 4096 * 2);
    }

    [Fact]
    public void EstimateKvCacheBytes_Q4_SmallerThanF16()
    {
        var f16 = ModelMemoryEstimator.EstimateKvCacheBytes(
            contextLength: 4096,
            numLayers: 32,
            hiddenSize: 4096,
            kvQuantType: KvCacheQuantizationType.F16);

        var q4 = ModelMemoryEstimator.EstimateKvCacheBytes(
            contextLength: 4096,
            numLayers: 32,
            hiddenSize: 4096,
            kvQuantType: KvCacheQuantizationType.Q4_0);

        q4.Should().BeLessThan(f16);
        q4.Should().Be(f16 / 4); // Q4 is 0.5 bytes vs F16's 2.0 bytes
    }

    [Theory]
    [InlineData("Q4_K_M", 0.5625)]
    [InlineData("Q8_0", 1.0)]
    [InlineData("fp16", 2.0)]
    [InlineData("fp32", 4.0)]
    [InlineData(null, 4.0)]
    public void GetBytesPerParameter_ReturnsExpectedValue(string? quantType, double expected)
    {
        var result = ModelMemoryEstimator.GetBytesPerParameter(quantType);
        result.Should().Be(expected);
    }

    [Fact]
    public void EstimateTotalVramBytes_CombinesModelAndKvCache()
    {
        var modelOnly = ModelMemoryEstimator.EstimateModelSizeBytes(ThreeBillion, "Q4_K_M");
        var kvOnly = ModelMemoryEstimator.EstimateKvCacheBytes(4096, 32, 4096);
        var total = ModelMemoryEstimator.EstimateTotalVramBytes(
            ThreeBillion, "Q4_K_M", null, 4096, 32, 4096);

        total.Should().Be(modelOnly + kvOnly);
    }
}
