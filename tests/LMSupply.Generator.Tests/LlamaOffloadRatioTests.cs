using AwesomeAssertions;
using LMSupply.Generator.Internal.Llama;
using LMSupply.Hardware;

namespace LMSupply.Generator.Tests;

/// <summary>
/// <see cref="LlamaOptions.GpuOffloadRatio"/> is documented to take precedence over
/// <see cref="LlamaOptions.GpuLayerCount"/>, but the server launch took the count alone, so a ratio
/// did nothing — and <see cref="LlamaOptions.CpuOnly"/>, which is a ratio of 0 with no count, launched
/// with every layer on the GPU backend's default of all layers. The load path now turns the ratio into
/// a count for the model before anything reads the count; these tests pin that conversion.
/// </summary>
public class LlamaOffloadRatioTests
{
    [Fact]
    public void CpuOnly_KeepsEveryLayerOffTheGpu()
    {
        var resolved = LlamaServerGeneratorModel.ResolveGpuOffloadRatio(LlamaOptions.CpuOnly, totalLayers: 32);

        resolved.GpuLayerCount.Should().Be(0);
        resolved.GpuOffloadRatio.Should().BeNull("the ratio has become the count");
    }

    // The load path's first step, as the load calls it: the caller's options, ratio resolved.
    [Fact]
    public void LoadPath_StartsFromTheCallersOptionsWithTheRatioResolved()
    {
        var options = new GeneratorOptions { LlamaOptions = LlamaOptions.CpuOnly };

        var chosen = LlamaServerGeneratorModel.ChooseLlamaOptions(options, modelPath: "not-read.gguf", ggufMetadata: null);

        chosen.GpuLayerCount.Should().Be(0, "CpuOnly must reach the VRAM fit and the server launch as zero layers");
        chosen.GpuOffloadRatio.Should().BeNull();
    }

    [Fact]
    public void FullGpu_OffloadsEveryLayer()
    {
        var resolved = LlamaServerGeneratorModel.ResolveGpuOffloadRatio(LlamaOptions.FullGpu, totalLayers: 32);

        resolved.GpuLayerCount.Should().Be(-1);
        resolved.GpuOffloadRatio.Should().BeNull();
    }

    [Fact]
    public void PartialRatio_BecomesThatShareOfTheModelsLayers()
    {
        var resolved = LlamaServerGeneratorModel.ResolveGpuOffloadRatio(LlamaOptions.WithGpuRatio(0.5f), totalLayers: 32);

        resolved.GpuLayerCount.Should().Be(16);
    }

    [Theory]
    [InlineData(null)]
    [InlineData(0)]
    public void PartialRatio_WithoutALayerCount_OffloadsEveryLayer(int? totalLayers)
    {
        var resolved = LlamaServerGeneratorModel.ResolveGpuOffloadRatio(LlamaOptions.WithGpuRatio(0.5f), totalLayers);

        resolved.GpuLayerCount.Should().Be(-1, "a share of an unknown number cannot be computed");
        resolved.GpuOffloadRatio.Should().BeNull();
    }

    [Fact]
    public void RatioTakesPrecedenceOverTheCount()
    {
        var options = new LlamaOptions { GpuLayerCount = 10, GpuOffloadRatio = 0f };

        var resolved = LlamaServerGeneratorModel.ResolveGpuOffloadRatio(options, totalLayers: 32);

        resolved.GpuLayerCount.Should().Be(0);
    }

    [Fact]
    public void NoRatio_LeavesTheOptionsAlone()
    {
        var options = new LlamaOptions { GpuLayerCount = 10 };

        LlamaServerGeneratorModel.ResolveGpuOffloadRatio(options, totalLayers: 32).Should().BeSameAs(options);
    }

    [Fact]
    public void NaNRatio_IsRejected()
    {
        var options = new LlamaOptions { GpuOffloadRatio = float.NaN };

        var act = () => LlamaServerGeneratorModel.ResolveGpuOffloadRatio(options, totalLayers: 32);

        act.Should().Throw<ArgumentException>();
    }

    // The hardware-fitted defaults carry the exact count only: a ratio derived from it would now take
    // precedence and be turned back into a count, and rounding can lose a layer (20/33 -> 19).
    [Fact]
    public void OptimalForHardware_CarriesTheCountWithoutADerivedRatio()
    {
        var options = LlamaOptions.GetOptimalForHardware(HardwareProfile.Current.GpuInfo, modelSizeBytes: 4L * 1024 * 1024 * 1024, totalLayers: 33);

        options.GpuOffloadRatio.Should().BeNull();
    }
}
