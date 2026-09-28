using AwesomeAssertions;
using LMSupply.Generator.Internal.Llama;
using LMSupply.Hardware;
using Xunit;

namespace LMSupply.Generator.Tests;

/// <summary>
/// The KV cache size is derived from the GGUF attention metadata. Expected values are what llama-server
/// itself reported (<c>llama_kv_cache: size = …</c>) when loading these models at <c>-c 16384</c> with the
/// default f16 cache and 4 slots — the metadata below is copied from the files.
/// </summary>
public class KvCacheGeometryTests
{
    private const long MiB = 1024 * 1024;

    /// <summary>Qwen2.5-7B-Instruct: grouped-query attention, 4 KV heads for 28 attention heads.</summary>
    private static readonly GgufMetadata Qwen25_7B = new()
    {
        Architecture = "qwen2",
        LayerCount = 28,
        EmbeddingLength = 3584,
        HeadCount = 28,
        HeadCountKv = 4,
    };

    /// <summary>Qwen3.5-2B: hybrid — one full-attention layer in four, the others recurrent.</summary>
    private static readonly GgufMetadata Qwen35_2B = new()
    {
        Architecture = "qwen35",
        LayerCount = 24,
        EmbeddingLength = 2048,
        HeadCount = 8,
        HeadCountKv = 2,
        KeyLength = 256,
        ValueLength = 256,
        FullAttentionInterval = 4,
    };

    /// <summary>Gemma 4 E4B: sliding-window layers with a narrower head, and 18 trailing layers sharing KV.</summary>
    private static readonly GgufMetadata Gemma4_E4B = new()
    {
        Architecture = "gemma4",
        LayerCount = 42,
        EmbeddingLength = 2560,
        HeadCount = 8,
        HeadCountKv = 2,
        KeyLength = 512,
        ValueLength = 512,
        KeyLengthSwa = 256,
        ValueLengthSwa = 256,
        SlidingWindow = 512,
        SharedKvLayers = 18,
        SlidingWindowPattern = Enumerable.Range(0, 42).Select(layer => (layer + 1) % 6 != 0).ToArray(),
    };

    [Fact]
    public void GroupedQueryAttention_CountsKvHeadsNotAttentionHeads()
    {
        var geometry = KvCacheGeometry.FromMetadata(Qwen25_7B)!;

        // 28 layers × 4 KV heads × 128 dims × (K+V) × 2 bytes
        geometry.BytesPerToken(null, null).Should().Be(57_344);
        geometry.TotalBytes(16_384, null, null, sequences: 4, ubatch: 512).Should().Be(896 * MiB, "llama-server: 896.00 MiB");
    }

    [Fact]
    public void HybridModel_CountsOnlyFullAttentionLayers()
    {
        var geometry = KvCacheGeometry.FromMetadata(Qwen35_2B)!;

        geometry.TotalBytes(16_384, null, null, sequences: 4, ubatch: 512)
            .Should().Be(192 * MiB, "llama-server: 192.00 MiB over 6 layers");
    }

    [Fact]
    public void SlidingWindowAndSharedLayers_MatchTheServer()
    {
        var geometry = KvCacheGeometry.FromMetadata(Gemma4_E4B)!;

        geometry.BytesPerToken(null, null).Should().Be(16_384, "4 full-attention layers of their own");
        geometry.SlidingWindowBytes(16_384, null, null, sequences: 4, ubatch: 512)
            .Should().Be(100 * MiB, "llama-server: 100.00 MiB over 20 layers, 2560 cells");
        geometry.TotalBytes(16_384, null, null, sequences: 4, ubatch: 512).Should().Be(356 * MiB);
    }

    [Fact]
    public void SlidingWindow_NeverHoldsMoreCellsThanTheContext()
    {
        var geometry = KvCacheGeometry.FromMetadata(Gemma4_E4B)!;

        geometry.SlidingWindowBytes(1_024, null, null, sequences: 4, ubatch: 512)
            .Should().Be(geometry.SlidingWindowBytes(16_384, null, null, sequences: 4, ubatch: 512) * 1_024 / 2_560);
    }

    [Fact]
    public void QuantizedCache_ScalesByBlockSize()
    {
        var geometry = KvCacheGeometry.FromMetadata(Qwen25_7B)!;

        // q8_0 stores 32 values in 34 bytes.
        geometry.BytesPerToken("q8_0", "q8_0").Should().Be(57_344 / 2 * 34 / 32);
        geometry.BytesPerToken("q8_0", "f16").Should().Be(57_344 / 4 * 34 / 32 + 57_344 / 2);
    }

    [Fact]
    public void SlidingWindowWithoutAPattern_IsCountedAsFullAttention()
    {
        // A file that names a window but not which layers use it is sized as if every layer kept the whole
        // context — over, never under.
        var geometry = KvCacheGeometry.FromMetadata(Qwen25_7B with { SlidingWindow = 4096 })!;

        geometry.BytesPerToken(null, null).Should().Be(57_344);
        geometry.SlidingWindowBytes(16_384, null, null, 1, 512).Should().Be(0);
    }

    [Fact]
    public void PerLayerKvHeads_SkipLayersWithoutAttention()
    {
        var geometry = KvCacheGeometry.FromMetadata(Qwen25_7B with
        {
            LayerCount = 4,
            HeadCountKvPerLayer = [4, 0, 4, 0],
        })!;

        geometry.BytesPerToken(null, null).Should().Be(2 * 4 * 128 * 2 * 2);
    }

    [Fact]
    public void WithoutLayerOrHeadGeometry_ReturnsNull()
    {
        KvCacheGeometry.FromMetadata(null).Should().BeNull();
        KvCacheGeometry.FromMetadata(new GgufMetadata { LayerCount = 28 }).Should().BeNull();
        KvCacheGeometry.FromMetadata(new GgufMetadata { EmbeddingLength = 3584, HeadCount = 28 }).Should().BeNull();
    }
}

/// <summary>
/// The context fit uses the file's KV geometry. Reproduces a consumer report: an 8 GB GPU with a ~6.8 GB
/// budget fitted Qwen2.5-7B (IQ4_XS, 4.2 GB) to 6,163 tokens of a requested 16,384, which llama-server loads
/// at 16,384 fully offloaded. The file here is tiny, so the budget is set to what the KV cache needs on top of
/// the fixed 512 MB buffer.
/// </summary>
[Collection("VramBudget")]
public class ContextFitUsesKvGeometryTests : IDisposable
{
    private readonly string _tempFile;
    private readonly string? _originalBudgetEnv;

    private static readonly GgufMetadata Qwen25_7B = new()
    {
        Architecture = "qwen2",
        LayerCount = 28,
        EmbeddingLength = 3584,
        HeadCount = 28,
        HeadCountKv = 4,
    };

    public ContextFitUsesKvGeometryTests()
    {
        _tempFile = Path.GetTempFileName();
        File.WriteAllBytes(_tempFile, new byte[1024]);
        _originalBudgetEnv = Environment.GetEnvironmentVariable(VramBudget.BudgetOverrideEnvVar);
    }

    public void Dispose()
    {
        if (File.Exists(_tempFile))
            File.Delete(_tempFile);
        Environment.SetEnvironmentVariable(VramBudget.BudgetOverrideEnvVar, _originalBudgetEnv);
        GC.SuppressFinalize(this);
    }

    [Fact]
    public void GroupedQueryModel_GetsTheContextItsCacheFits()
    {
        // 512 MB buffer + 896 MB = the 16,384-token f16 cache of this model (+1 MB for the 1 KB file × 1.1).
        Environment.SetEnvironmentVariable(VramBudget.BudgetOverrideEnvVar, "1409");

        var (context, floored) = LlamaServerGeneratorModel.EstimateSafeContextLengthDetailed(
            _tempFile, requestedContext: 16_384, gpuLayerCount: -1, ggufMetadata: Qwen25_7B);

        floored.Should().BeFalse();
        context.Should().Be(16_384, "the cache of 4 KV heads × 28 layers fits");
    }

    [Fact]
    public void QuantizedCache_FitsMoreContext()
    {
        Environment.SetEnvironmentVariable(VramBudget.BudgetOverrideEnvVar, "1409");

        var (f16, _) = LlamaServerGeneratorModel.EstimateSafeContextLengthDetailed(
            _tempFile, requestedContext: 65_536, gpuLayerCount: -1, ggufMetadata: Qwen25_7B);
        var (q8, _) = LlamaServerGeneratorModel.EstimateSafeContextLengthDetailed(
            _tempFile, requestedContext: 65_536, gpuLayerCount: -1, ggufMetadata: Qwen25_7B,
            cacheTypeK: "q8_0", cacheTypeV: "q8_0");

        f16.Should().BeInRange(16_384, 16_410, "the budget holds 896 MB of cache plus the rounding margin");
        q8.Should().BeGreaterThan(30_000, "a q8_0 cache is about half the size");
    }
}
