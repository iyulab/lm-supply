using AwesomeAssertions;
using LMSupply.Hardware;

namespace LMSupply.Reranker.Tests;

/// <summary>
/// On a Medium host <c>auto</c> used to mean <c>quality</c> — an ONNX model trained on English and Chinese
/// that on a Korean corpus ranks worse than no reranking. When a llama-server binary is already cached,
/// <c>auto</c> now takes the GGUF build of the multilingual model instead (a fifth of the download, a
/// fraction of the CPU latency, every language). The binary is a precondition, never a consequence:
/// <c>auto</c> does not fetch one, so a host without it keeps the ONNX mapping unchanged.
/// </summary>
public sealed class AutoAliasGgufRouteTests
{
    [Fact]
    public void Medium_with_a_cached_server_resolves_to_the_gguf_alias()
    {
        LocalReranker.ResolveAutoAlias(PerformanceTier.Medium, llamaServerCached: true).Should().Be("multilingual-fast");
        LocalReranker.GgufAliases.Should().ContainKey("multilingual-fast", "the rewritten alias must enter the GGUF route");
    }

    [Fact]
    public void Medium_without_a_cached_server_stays_on_the_onnx_mapping()
    {
        LocalReranker.ResolveAutoAlias(PerformanceTier.Medium, llamaServerCached: false).Should().Be("auto");
    }

    [Theory]
    [InlineData(PerformanceTier.Low)]
    [InlineData(PerformanceTier.High)]
    [InlineData(PerformanceTier.Ultra)]
    public void Other_tiers_are_unchanged_whether_or_not_a_server_is_cached(PerformanceTier tier)
    {
        LocalReranker.ResolveAutoAlias(tier, llamaServerCached: true).Should().Be("auto");
        LocalReranker.ResolveAutoAlias(tier, llamaServerCached: false).Should().Be("auto");
    }
}
