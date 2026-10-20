using AwesomeAssertions;
using LMSupply.Generator.Internal.Llama;

namespace LMSupply.Generator.Tests;

/// <summary>
/// A GGUF load that runs out of memory on start is retried with fewer GPU layers. The layers the model reports
/// (<c>GeneratorModelInfo.GpuLayers</c>) are those its server started with, not those first planned.
/// </summary>
public class LlamaLoadSizingTests
{
    [Fact]
    public void AStartWithThePlannedLayers_KeepsTheSizing()
    {
        var sizing = new LlamaServerGeneratorModel.LoadSizing(GpuLayers: 20, TotalLayers: 32);

        sizing.AfterRetry(plannedGpuLayers: 20, startedGpuLayers: 20).Should().BeSameAs(sizing);
    }

    [Fact]
    public void APartialOffloadThatRetried_ReportsTheLayersItStartedWith()
    {
        var sizing = new LlamaServerGeneratorModel.LoadSizing(GpuLayers: 20, TotalLayers: 32, EstimatedVramBytes: 1);

        sizing.AfterRetry(plannedGpuLayers: 20, startedGpuLayers: 15)
            .Should().Be(sizing with { GpuLayers = 15 }, "the estimates stay those the plan was made on");
    }

    [Fact]
    public void ACallersLayerCountThatRetried_IsReportedToo()
    {
        // No sizing was decided: the caller set the layer count, and the GPU could not hold it.
        LlamaServerGeneratorModel.LoadSizing.None.AfterRetry(plannedGpuLayers: 40, startedGpuLayers: 30)
            .GpuLayers.Should().Be(30);
    }
}
