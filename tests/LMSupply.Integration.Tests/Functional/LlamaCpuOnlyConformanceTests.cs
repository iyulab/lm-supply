using AwesomeAssertions;
using LMSupply.Generator;
using LMSupply.Generator.Models;
using LMSupply.Llama;

namespace LMSupply.Integration.Tests.Functional;

/// <summary>
/// <see cref="LlamaOptions.CpuOnly"/> end to end on a real GGUF model: the preset loads and generates.
/// Before 0.64.0 its offload ratio of 0 was never read and llama-server started with
/// <c>--n-gpu-layers -1</c> — every layer on the GPU; the unit tests pin the resolution to a layer count
/// of 0, and this exercises the path that carries it into a running server.
/// <para>
/// It cannot confirm the layer placement itself. llama-server's startup log carries no layer or offload
/// report on this build (checked at its default verbosity: model load and listening lines only, no
/// "offloaded N/M layers"), and <c>GetModelInfo().GpuLayers</c> is null by contract unless a partial
/// offload happened. Confirming placement from outside would need the launch arguments exposed
/// (proposed) or a device-memory reading. Needs a GGUF model and llama-server, so it is local only.
/// </para>
/// </summary>
[Trait("Category", "Functional")]
[Trait("Category", "LocalOnly")]
public sealed class LlamaCpuOnlyConformanceTests
{
    private const string FastModel = "gguf:gemma4-fast";

    [Fact]
    public async Task CpuOnly_LoadsAndGenerates()
    {
        await using var generator = await LocalGenerator.LoadAsync(
            FastModel,
            new GeneratorOptions { LlamaOptions = LlamaOptions.CpuOnly },
            cancellationToken: TestContext.Current.CancellationToken);

        var log = generator.GetModelInfo().BackendLog;
        log.Should().NotBeNullOrEmpty("llama-server's startup log is the evidence that a server really ran");
        log.Should().Contain("model loaded").And.Contain("listening on",
            $"the CpuOnly preset must reach a running server. The log was:\n{log}");

        var reply = await generator.GenerateCompleteAsync(
            "Say hi.", new GenerationOptions { MaxTokens = 4 }, TestContext.Current.CancellationToken);
        reply.Should().NotBeNullOrWhiteSpace("a CPU-only server must still generate");
    }
}
