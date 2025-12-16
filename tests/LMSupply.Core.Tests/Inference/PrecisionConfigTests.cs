using FluentAssertions;
using LMSupply.Inference;

namespace LMSupply.Core.Tests.Inference;

public class PrecisionConfigTests
{
    #region GetBytesPerParameter Tests

    [Theory]
    [InlineData(ModelPrecision.FP32, 4.0)]
    [InlineData(ModelPrecision.FP16, 2.0)]
    [InlineData(ModelPrecision.INT8, 1.0)]
    [InlineData(ModelPrecision.INT4, 0.5)]
    [InlineData(ModelPrecision.Auto, 2.0)] // Defaults to FP16 estimate
    public void GetBytesPerParameter_ReturnsCorrectValue(ModelPrecision precision, double expected)
    {
        var result = PrecisionConfig.GetBytesPerParameter(precision);

        result.Should().Be(expected);
    }

    #endregion

    #region EstimateModelMemory Tests

    [Fact]
    public void EstimateModelMemory_FP32_CalculatesCorrectly()
    {
        var config = new PrecisionConfig { ModelPrecision = ModelPrecision.FP32 };
        const long parameterCount = 1_000_000;

        var memory = config.EstimateModelMemory(parameterCount);

        memory.Should().Be(4_000_000); // 1M params * 4 bytes
    }

    [Fact]
    public void EstimateModelMemory_FP16_CalculatesCorrectly()
    {
        var config = new PrecisionConfig { ModelPrecision = ModelPrecision.FP16 };
        const long parameterCount = 1_000_000;

        var memory = config.EstimateModelMemory(parameterCount);

        memory.Should().Be(2_000_000); // 1M params * 2 bytes
    }

    [Fact]
    public void EstimateModelMemory_INT8_CalculatesCorrectly()
    {
        var config = new PrecisionConfig { ModelPrecision = ModelPrecision.INT8 };
        const long parameterCount = 1_000_000;

        var memory = config.EstimateModelMemory(parameterCount);

        memory.Should().Be(1_000_000); // 1M params * 1 byte
    }

    [Fact]
    public void EstimateModelMemory_INT4_CalculatesCorrectly()
    {
        var config = new PrecisionConfig { ModelPrecision = ModelPrecision.INT4 };
        const long parameterCount = 1_000_000;

        var memory = config.EstimateModelMemory(parameterCount);

        memory.Should().Be(500_000); // 1M params * 0.5 bytes
    }

    [Fact]
    public void EstimateModelMemory_LargeModel_HandlesCorrectly()
    {
        var config = new PrecisionConfig { ModelPrecision = ModelPrecision.FP16 };
        const long parameterCount = 7_000_000_000; // 7B parameters

        var memory = config.EstimateModelMemory(parameterCount);

        memory.Should().Be(14_000_000_000); // 14GB
    }

    #endregion

    #region GetRecommendedPrecision Tests

    [Fact]
    public void GetRecommendedPrecision_SufficientMemory_ReturnsFP32()
    {
        const long availableMemory = 10_000_000_000; // 10GB
        const long parameterCount = 1_000_000_000; // 1B params needs ~6GB for FP32

        var precision = PrecisionConfig.GetRecommendedPrecision(availableMemory, parameterCount);

        precision.Should().Be(ModelPrecision.FP32);
    }

    [Fact]
    public void GetRecommendedPrecision_ModerateMemory_ReturnsFP16()
    {
        const long availableMemory = 4_000_000_000; // 4GB
        const long parameterCount = 1_000_000_000; // 1B params needs ~3GB for FP16

        var precision = PrecisionConfig.GetRecommendedPrecision(availableMemory, parameterCount);

        precision.Should().Be(ModelPrecision.FP16);
    }

    [Fact]
    public void GetRecommendedPrecision_LimitedMemory_ReturnsINT8()
    {
        const long availableMemory = 2_000_000_000; // 2GB
        const long parameterCount = 1_000_000_000; // 1B params needs ~1.5GB for INT8

        var precision = PrecisionConfig.GetRecommendedPrecision(availableMemory, parameterCount);

        precision.Should().Be(ModelPrecision.INT8);
    }

    [Fact]
    public void GetRecommendedPrecision_VeryLimitedMemory_ReturnsINT4()
    {
        const long availableMemory = 500_000_000; // 500MB
        const long parameterCount = 1_000_000_000; // 1B params needs ~750MB for INT4

        var precision = PrecisionConfig.GetRecommendedPrecision(availableMemory, parameterCount);

        precision.Should().Be(ModelPrecision.INT4);
    }

    #endregion

    #region Preset Configurations Tests

    [Fact]
    public void Default_HasExpectedSettings()
    {
        var config = PrecisionConfig.Default;

        config.ModelPrecision.Should().Be(ModelPrecision.Auto);
        config.ComputePrecision.Should().Be(ComputePrecision.Mixed);
        config.EnableGraphOptimization.Should().BeTrue();
        config.OptimizationLevel.Should().Be(OptimizationLevel.Basic);
    }

    [Fact]
    public void HighAccuracy_HasExpectedSettings()
    {
        var config = PrecisionConfig.HighAccuracy;

        config.ModelPrecision.Should().Be(ModelPrecision.FP32);
        config.ComputePrecision.Should().Be(ComputePrecision.FP32);
        config.EnableGraphOptimization.Should().BeTrue();
        config.OptimizationLevel.Should().Be(OptimizationLevel.Extended);
    }

    [Fact]
    public void HighPerformance_HasExpectedSettings()
    {
        var config = PrecisionConfig.HighPerformance;

        config.ModelPrecision.Should().Be(ModelPrecision.FP16);
        config.ComputePrecision.Should().Be(ComputePrecision.Mixed);
        config.EnableGraphOptimization.Should().BeTrue();
        config.OptimizationLevel.Should().Be(OptimizationLevel.All);
    }

    [Fact]
    public void LowMemory_HasExpectedSettings()
    {
        var config = PrecisionConfig.LowMemory;

        config.ModelPrecision.Should().Be(ModelPrecision.INT8);
        config.ComputePrecision.Should().Be(ComputePrecision.FP16);
        config.EnableGraphOptimization.Should().BeTrue();
        config.OptimizationLevel.Should().Be(OptimizationLevel.All);
    }

    #endregion
}
