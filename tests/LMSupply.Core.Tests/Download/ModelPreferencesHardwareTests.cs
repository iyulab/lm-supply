using AwesomeAssertions;
using LMSupply.Core.Download;
using LMSupply.Hardware;

namespace LMSupply.Core.Tests.Download;

public class ModelPreferencesHardwareTests
{
    [Fact]
    public void ForTier_Low_PrefersQuant4AndSetsLowMemory()
    {
        var prefs = ModelPreferences.ForTier(PerformanceTier.Low);

        prefs.QuantizationPriority[0].Should().Be(Quantization.Quant4);
        prefs.PreferLowMemory.Should().BeTrue();
    }

    [Fact]
    public void ForTier_Medium_PrefersQuant8()
    {
        var prefs = ModelPreferences.ForTier(PerformanceTier.Medium);

        prefs.QuantizationPriority[0].Should().Be(Quantization.Quant8);
        prefs.PreferLowMemory.Should().BeFalse();
    }

    [Fact]
    public void ForTier_High_PrefersFp16()
    {
        var prefs = ModelPreferences.ForTier(PerformanceTier.High);

        prefs.QuantizationPriority[0].Should().Be(Quantization.Fp16);
        prefs.PreferLowMemory.Should().BeFalse();
    }

    [Fact]
    public void ForTier_Ultra_PrefersDefault()
    {
        var prefs = ModelPreferences.ForTier(PerformanceTier.Ultra);

        prefs.QuantizationPriority[0].Should().Be(Quantization.Default);
        prefs.PreferLowMemory.Should().BeFalse();
    }

    [Fact]
    public void ForCurrentHardware_ReturnsNonNull()
    {
        var prefs = ModelPreferences.ForCurrentHardware();
        prefs.Should().NotBeNull();
        prefs.QuantizationPriority.Should().HaveCount(4);
    }

    [Fact]
    public void ForTier_AllTiers_ContainAllQuantizations()
    {
        foreach (var tier in Enum.GetValues<PerformanceTier>())
        {
            var prefs = ModelPreferences.ForTier(tier);
            prefs.QuantizationPriority.Should().Contain(Quantization.Default);
            prefs.QuantizationPriority.Should().Contain(Quantization.Quant4);
            prefs.QuantizationPriority.Should().Contain(Quantization.Fp16);
            prefs.QuantizationPriority.Should().Contain(Quantization.Quant8);
        }
    }

    [Theory]
    [InlineData(PerformanceTier.Ultra)]
    [InlineData(PerformanceTier.High)]
    [InlineData(PerformanceTier.Medium)]
    [InlineData(PerformanceTier.Low)]
    public void OnTheCpu_Fp16IsTheLastResort_AtEveryTier(PerformanceTier tier)
    {
        ModelPreferences.ForCpuTier(tier).QuantizationPriority[^1].Should().Be(Quantization.Fp16, "the CPU provider runs FP16 through casts, and some FP16 exports do not load there");
    }

    [Fact]
    public void OnTheCpu_AMachineWithMuchMemoryTakesFullPrecision()
    {
        ModelPreferences.ForCpuTier(PerformanceTier.High).QuantizationPriority[0].Should().Be(Quantization.Default);
        ModelPreferences.ForCpuTier(PerformanceTier.Medium).QuantizationPriority[0].Should().Be(Quantization.Quant8);
        ModelPreferences.ForCpuTier(PerformanceTier.Low).QuantizationPriority[0].Should().Be(Quantization.Quant4);
    }

    [Fact]
    public void AnExplicitCpuLoad_RanksAsTheCpuRunsIt()
    {
        var tier = HardwareProfile.For(ExecutionProvider.Cpu).Tier;

        ModelPreferences.ForProvider(ExecutionProvider.Cpu).QuantizationPriority.Should().Equal(ModelPreferences.ForCpuTier(tier).QuantizationPriority);
    }
}
