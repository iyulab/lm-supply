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
}
