using AwesomeAssertions;
using LMSupply.Generator;
using Xunit;

namespace LMSupply.Generator.Onnx.Tests;

/// <summary>
/// ONNX Runtime GenAI has no OpenVINO path here: an explicit <see cref="ExecutionProvider.OpenVino"/> request is refused
/// before anything is provisioned, never run on the CPU runtime instead.
/// </summary>
[Trait("Category", "Unit")]
public sealed class OpenVinoRefusalTests
{
    [Fact]
    public async Task EnsureRuntime_OpenVino_Throws()
    {
        OnnxGeneratorBackend.Register();
        var backend = OnnxGeneratorBackendRegistry.Require();

        var act = () => backend.EnsureRuntimeAsync(ExecutionProvider.OpenVino, progress: null, TestContext.Current.CancellationToken);

        (await act.Should().ThrowAsync<NotSupportedException>()).WithMessage("*OpenVino*text generation*");
    }
}
