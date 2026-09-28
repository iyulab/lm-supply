using AwesomeAssertions;
using LMSupply.Llama.Server;
using Xunit;
using Resident = LMSupply.Llama.Server.LlamaServerPool.ResidentServer;

namespace LMSupply.Llama.Tests.Server;

/// <summary>
/// A load shares a running server of its model when that server already holds the context the load needs, instead
/// of sizing a new one against memory the running server is holding.
/// </summary>
public class SharableContextTests
{
    private const string Model = @"C:\models\qwen.gguf";

    private static int? Select(int minContext, params Resident[] servers)
        => LlamaServerPool.SelectSharableContext(servers, Model, LlamaServerBackend.Cuda12, ServerMode.Generation, minContext);

    [Fact]
    public void ServerWithEnoughContext_IsShared()
        => Select(16_384, new Resident(Model, LlamaServerBackend.Cuda12, ServerMode.Generation, 16_384, true))
            .Should().Be(16_384);

    [Fact]
    public void LargerContext_IsShared_TheSmallestThatFits()
        => Select(
                8_192,
                new Resident(Model, LlamaServerBackend.Cuda12, ServerMode.Generation, 32_768, true),
                new Resident(Model, LlamaServerBackend.Cuda12, ServerMode.Generation, 16_384, true))
            .Should().Be(16_384);

    [Fact]
    public void SmallerContext_IsNotShared()
        => Select(16_384, new Resident(Model, LlamaServerBackend.Cuda12, ServerMode.Generation, 6_163, true))
            .Should().BeNull();

    [Fact]
    public void OtherModelBackendModeOrADeadServer_IsNotShared()
        => Select(
                4_096,
                new Resident(@"C:\models\other.gguf", LlamaServerBackend.Cuda12, ServerMode.Generation, 16_384, true),
                new Resident(Model, LlamaServerBackend.Vulkan, ServerMode.Generation, 16_384, true),
                new Resident(Model, LlamaServerBackend.Cuda12, ServerMode.Embedding, 16_384, true),
                new Resident(Model, LlamaServerBackend.Cuda12, ServerMode.Generation, 16_384, false))
            .Should().BeNull();

    [Fact]
    public void ModelPath_ComparesCaseInsensitively()
        => Select(4_096, new Resident(Model.ToUpperInvariant(), LlamaServerBackend.Cuda12, ServerMode.Generation, 4_096, true))
            .Should().Be(4_096);
}
