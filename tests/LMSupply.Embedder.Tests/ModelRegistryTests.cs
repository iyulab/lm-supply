using FluentAssertions;
using LMSupply.Embedder.Utils;

namespace LMSupply.Embedder.Tests;

public class ModelRegistryTests
{
    [Fact]
    public void TryGetModel_ReturnsTrue_ForKnownModel()
    {
        var result = ModelRegistry.TryGetModel("all-MiniLM-L6-v2", out var info);

        result.Should().BeTrue();
        info.Should().NotBeNull();
        info!.RepoId.Should().Be("sentence-transformers/all-MiniLM-L6-v2");
        info.Dimensions.Should().Be(384);
    }

    [Fact]
    public void TryGetModel_ReturnsFalse_ForUnknownModel()
    {
        var result = ModelRegistry.TryGetModel("unknown-model", out var info);

        result.Should().BeFalse();
        info.Should().BeNull();
    }

    [Fact]
    public void TryGetModel_IsCaseInsensitive()
    {
        var result = ModelRegistry.TryGetModel("ALL-MINILM-L6-V2", out var info);

        result.Should().BeTrue();
        info.Should().NotBeNull();
    }

    [Fact]
    public void GetAvailableModels_ReturnsNonEmptyList()
    {
        var models = ModelRegistry.GetAvailableModels().ToList();

        models.Should().NotBeEmpty();
        models.Should().Contain("all-MiniLM-L6-v2");
        models.Should().Contain("bge-small-en-v1.5");
    }

    [Theory]
    [InlineData("all-MiniLM-L6-v2", 384, PoolingMode.Mean)]
    [InlineData("all-mpnet-base-v2", 768, PoolingMode.Mean)]
    [InlineData("bge-small-en-v1.5", 384, PoolingMode.Cls)]
    [InlineData("multilingual-e5-small", 384, PoolingMode.Mean)]
    public void KnownModels_HaveCorrectConfiguration(string modelId, int dimensions, PoolingMode pooling)
    {
        ModelRegistry.TryGetModel(modelId, out var info);

        info.Should().NotBeNull();
        info!.Dimensions.Should().Be(dimensions);
        info.PoolingMode.Should().Be(pooling);
    }

    [Fact]
    public void DefaultAlias_PointsToBgeSmall()
    {
        ModelRegistry.TryGetModel("default", out var info);

        info.Should().NotBeNull();
        info!.RepoId.Should().Be("BAAI/bge-small-en-v1.5");
    }

    [Fact]
    public void FastAlias_PointsToMiniLM()
    {
        ModelRegistry.TryGetModel("fast", out var info);

        info.Should().NotBeNull();
        info!.RepoId.Should().Be("sentence-transformers/all-MiniLM-L6-v2");
    }

    [Fact]
    public void QualityAlias_PointsToBgeBase()
    {
        ModelRegistry.TryGetModel("quality", out var info);

        info.Should().NotBeNull();
        info!.RepoId.Should().Be("BAAI/bge-base-en-v1.5");
    }
}
