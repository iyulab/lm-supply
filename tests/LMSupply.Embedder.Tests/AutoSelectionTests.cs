using AwesomeAssertions;
using LMSupply.Embedder.Utils;

namespace LMSupply.Embedder.Tests;

/// <summary><c>auto</c> weighs each candidate's size against the memory budget, so every catalog entry carries one.</summary>
public class AutoSelectionTests
{
    private const long Mb = 1024 * 1024;

    [Fact]
    public void EveryCatalogEntry_CarriesSizeAndParameterCount()
    {
        DefaultModels.All.Should().OnlyContain(m => m.SizeBytes > 0 && m.Parameters > 0);
    }

    [Theory]
    [InlineData(108, "intfloat/multilingual-e5-small")] // a CPU-only host's budget: nothing fits, the smallest is taken
    [InlineData(1_000, "nomic-ai/nomic-embed-text-v1.5")]
    [InlineData(4_000, "BAAI/bge-m3")]
    public void Auto_FollowsTheBudget(long budgetMb, string expected)
    {
        var selected = ModelRegistryBase<ModelInfo>.SelectLargestFitting(EmbedderModelRegistry.AutoCandidates, budgetMb * Mb, out _);

        selected.RepoId.Should().Be(expected);
    }

    [Fact]
    public void Auto_ResolvesOnThisHost()
    {
        EmbedderModelRegistry.Default.Resolve("auto").SizeBytes.Should().BeGreaterThan(0);
    }
}
