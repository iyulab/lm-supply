using AwesomeAssertions;
using LMSupply.Generator.Internal.Llama;

namespace LMSupply.Generator.Tests.Internal.Llama;

/// <summary>
/// The context llama-server is started with when the caller does not choose one: the model's trained length up to a
/// ceiling, then bounded by the memory that holds the KV cache. <c>MaxContextLength</c> is that effective length — the
/// length requests are checked and trimmed against — and the trained length stays readable as
/// <c>GgufMetadata.ContextLength</c>. (Before, an unset option started every GGUF at 4096 while the model reported the
/// trained length, so a request the model said it could take was refused by its own server.)
/// </summary>
public class LlamaServerGeneratorModelContextLengthTests
{
    private const long GiB = 1024L * 1024 * 1024;

    // ─── requested length ───

    [Fact]
    public void Unset_UsesTheTrainedLength_UpToTheCeiling()
    {
        LlamaServerGeneratorModel.ResolveRequestedContextLength(null, 32768).Should().Be(32768);
        LlamaServerGeneratorModel.ResolveRequestedContextLength(null, 262144)
            .Should().Be(LlamaServerGeneratorModel.DefaultContextCeiling);
        LlamaServerGeneratorModel.ResolveRequestedContextLength(null, 2048)
            .Should().Be(2048, because: "a model trained shorter than the old default is not stretched past its training");
    }

    [Fact]
    public void Unset_WithoutMetadata_Uses4096()
    {
        LlamaServerGeneratorModel.ResolveRequestedContextLength(null, null).Should().Be(4096);
        LlamaServerGeneratorModel.ResolveRequestedContextLength(null, 0).Should().Be(4096);
    }

    [Fact]
    public void Explicit_IsTheCallers()
    {
        LlamaServerGeneratorModel.ResolveRequestedContextLength(8192, 262144).Should().Be(8192);
        LlamaServerGeneratorModel.ResolveRequestedContextLength(131072, 32768)
            .Should().Be(131072, because: "an explicit length is the caller's choice, ceiling or not");
    }

    // ─── RAM bound (CPU backend) ───

    private static GgufMetadata Attention(int layers = 28, int kvHeads = 8, int headDim = 128) => new()
    {
        LayerCount = layers,
        HeadCountKv = kvHeads,
        KeyLength = headDim,
        ValueLength = headDim,
        ContextLength = 262144,
    };

    [Fact]
    public void Ram_AmpleMemory_KeepsTheRequest()
    {
        // 28 layers × 8 KV heads × 128 × (K+V) × f16 = 112 KiB per token → 32768 tokens ≈ 3.5 GiB, in 16 GiB of a 32 GiB box.
        var context = LlamaServerGeneratorModel.EstimateRamBoundContextLength(
            modelFileSize: 1 * GiB, requestedContext: 32768, Attention(), null, null, sequences: 1, ubatch: 512,
            systemMemoryBytes: 32 * GiB);

        context.Should().Be(32768);
    }

    [Fact]
    public void Ram_SmallMemory_BoundsTheContext_AboveTheFloor()
    {
        // 8 GiB box: 4 GiB budget − 1.1 GiB weights ≈ 2.9 GiB → about 27 k tokens of 112 KiB.
        var context = LlamaServerGeneratorModel.EstimateRamBoundContextLength(
            1 * GiB, 32768, Attention(), null, null, 1, 512, systemMemoryBytes: 8 * GiB);

        context.Should().BeInRange(20000, 32767);
    }

    [Fact]
    public void Ram_NoRoomAtAll_StaysAt4096()
    {
        var context = LlamaServerGeneratorModel.EstimateRamBoundContextLength(
            6 * GiB, 32768, Attention(), null, null, 1, 512, systemMemoryBytes: 8 * GiB);

        context.Should().Be(4096, because: "the bound never goes below the old default — a load that cannot fit fails on its own terms");
    }

    [Fact]
    public void Ram_UnknownMemory_KeepsTheRequest()
    {
        LlamaServerGeneratorModel.EstimateRamBoundContextLength(1 * GiB, 32768, Attention(), null, null, 1, 512, 0)
            .Should().Be(32768);
    }
}
