using AwesomeAssertions;
using LMSupply.Generator.Internal.Llama;
using LMSupply.Llama.Server;

namespace LMSupply.Generator.Tests;

/// <summary>
/// An unset <see cref="LlamaOptions.GpuLayerCount"/> means "all layers" at launch, and the VRAM fit is what lowers
/// "all" to what the GPU holds. Until 0.81.0 the fit ran only for an explicit -1, so a caller that passed
/// <see cref="LlamaOptions"/> to change one unrelated knob (a request timeout, the thread count) offloaded every layer
/// regardless of VRAM.
/// </summary>
public class LlamaOptionsVramFitTests
{
    [Fact]
    public void OptionsThatOnlySetATimeout_KeepTheVramFit()
    {
        var opts = new LlamaOptions { RequestTimeout = TimeSpan.FromMinutes(20) };

        LlamaServerGeneratorModel.NeedsVramFit(opts, LlamaServerBackend.Cuda12).Should().BeTrue();
    }

    [Fact]
    public void AnExplicitMinusOne_IsFitted()
    {
        LlamaServerGeneratorModel.NeedsVramFit(new LlamaOptions { GpuLayerCount = -1 }, LlamaServerBackend.Vulkan)
            .Should().BeTrue();
    }

    [Theory]
    [InlineData(0)]
    [InlineData(20)]
    public void AnExplicitCount_IsTheCallersChoice(int layers)
    {
        LlamaServerGeneratorModel.NeedsVramFit(new LlamaOptions { GpuLayerCount = layers }, LlamaServerBackend.Cuda12)
            .Should().BeFalse();
    }

    [Fact]
    public void TheCpuBackend_HasNothingToFit()
    {
        LlamaServerGeneratorModel.NeedsVramFit(new LlamaOptions(), LlamaServerBackend.Cpu).Should().BeFalse();
    }
}
