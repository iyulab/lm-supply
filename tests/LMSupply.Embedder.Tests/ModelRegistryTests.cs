using AwesomeAssertions;
using LMSupply.Embedder.Utils;
using LMSupply.Exceptions;

namespace LMSupply.Embedder.Tests;

public class ModelRegistryTests
{
    private readonly EmbedderModelRegistry _registry = EmbedderModelRegistry.Default;

    [Fact]
    public void TryResolve_ReturnsTrue_ForKnownShortName()
    {
        var result = _registry.TryResolve("multilingual-e5-small", out var info);

        result.Should().BeTrue();
        info.Should().NotBeNull();
        info!.RepoId.Should().Be("intfloat/multilingual-e5-small");
        info.Dimensions.Should().Be(384);
    }

    [Fact]
    public void TryResolve_ReturnsFalse_ForUnknownModel()
    {
        var result = _registry.TryResolve("unknown-model", out var info);

        result.Should().BeFalse();
        info.Should().BeNull();
    }

    [Fact]
    public void TryResolve_IsCaseInsensitive()
    {
        var result = _registry.TryResolve("MULTILINGUAL-E5-SMALL", out var info);

        result.Should().BeTrue();
        info.Should().NotBeNull();
    }

    [Fact]
    public void GetAliases_ReturnsNonEmptyList()
    {
        var aliases = _registry.GetAliases();

        aliases.Should().NotBeEmpty();
        aliases.Select(a => a.Name).Should().Contain("default");
        aliases.Select(a => a.Name).Should().Contain("fast");
        aliases.Select(a => a.Name).Should().Contain("auto");
    }

    [Theory]
    [InlineData("all-mpnet-base-v2", 768, PoolingMode.Mean)]
    [InlineData("bge-base-en-v1.5", 768, PoolingMode.Cls)]
    [InlineData("multilingual-e5-small", 384, PoolingMode.Mean)]
    [InlineData("multilingual-e5-large", 1024, PoolingMode.Mean)]
    public void KnownModels_HaveCorrectConfiguration(string modelId, int dimensions, PoolingMode pooling)
    {
        _registry.TryResolve(modelId, out var info);

        info.Should().NotBeNull();
        info!.Dimensions.Should().Be(dimensions);
        info.PoolingMode.Should().Be(pooling);
    }

    [Fact]
    public void DefaultAlias_PointsToBgeM3()
    {
        _registry.TryResolve("default", out var info);

        info.Should().NotBeNull();
        info!.RepoId.Should().Be("BAAI/bge-m3");
    }

    [Fact]
    public void FastAlias_PointsToMultilingualE5Small()
    {
        _registry.TryResolve("fast", out var info);

        info.Should().NotBeNull();
        info!.RepoId.Should().Be("intfloat/multilingual-e5-small");
    }

    [Fact]
    public void QualityAlias_PointsToBgeM3()
    {
        _registry.TryResolve("quality", out var info);

        info.Should().NotBeNull();
        info!.RepoId.Should().Be("BAAI/bge-m3");
    }

    [Fact]
    public void AutoAlias_ResolvesToModel()
    {
        var result = _registry.TryResolve("auto", out var info);

        result.Should().BeTrue();
        info.Should().NotBeNull();
        info!.RepoId.Should().NotBeNullOrEmpty();
    }

    [Fact]
    public void Resolve_ThrowsForUnknownModel()
    {
        var act = () => _registry.Resolve("completely-nonexistent-xyz");

        act.Should().Throw<ModelNotFoundException>();
    }

    [Fact]
    public void GetAvailableModels_ReturnsDeduplicatedModels()
    {
        var models = _registry.GetAvailableModels();

        models.Should().NotBeEmpty();
        // All models should have distinct RepoIds
        models.Select(m => m.RepoId).Should().OnlyHaveUniqueItems();
    }

    [Fact]
    public void FullRepoId_ResolvesCorrectly()
    {
        var result = _registry.TryResolve("BAAI/bge-base-en-v1.5", out var info);

        result.Should().BeTrue();
        info.Should().NotBeNull();
        info!.RepoId.Should().Be("BAAI/bge-base-en-v1.5");
    }

    [Fact]
    public void RegisterAlias_WorksForUserAlias()
    {
        var registry = new EmbedderModelRegistry(DefaultModels.All);
        registry.RegisterAlias("my-embed", "BAAI/bge-base-en-v1.5");

        var result = registry.TryResolve("my-embed", out var info);

        result.Should().BeTrue();
        info.Should().NotBeNull();
        info!.RepoId.Should().Be("BAAI/bge-base-en-v1.5");
    }

    [Fact]
    public void RegisterAlias_ThrowsForSystemAlias()
    {
        var registry = new EmbedderModelRegistry(DefaultModels.All);

        var act = () => registry.RegisterAlias("default", "some-model");

        act.Should().Throw<AliasConflictException>();
    }

    [Fact]
    public void RemoveAlias_WorksForUserAlias()
    {
        var registry = new EmbedderModelRegistry(DefaultModels.All);
        registry.RegisterAlias("my-embed", "BAAI/bge-base-en-v1.5");

        registry.RemoveAlias("my-embed").Should().BeTrue();
        registry.TryResolve("my-embed", out _).Should().BeFalse();
    }

    [Fact]
    public void RemoveAlias_ReturnsFalse_ForSystemAlias()
    {
        var registry = new EmbedderModelRegistry(DefaultModels.All);

        registry.RemoveAlias("default").Should().BeFalse();
    }

    // --- Query/passage prefix convention (docket iyulab/lm-supply#171) ---

    [Theory]
    [InlineData("fast", "query: ", "passage: ")]
    [InlineData("large", "query: ", "passage: ")]
    [InlineData("e5-small-v2", "query: ", "passage: ")]
    [InlineData("e5-base-v2", "query: ", "passage: ")]
    [InlineData("multilingual-e5-small", "query: ", "passage: ")]
    [InlineData("multilingual-e5-base", "query: ", "passage: ")]
    [InlineData("multilingual-e5-large", "query: ", "passage: ")]
    [InlineData("nomic-embed-text-v1.5", "search_query: ", "search_document: ")]
    public void E5AndNomicPresets_HaveQueryPassagePrefixes(string modelId, string queryPrefix, string passagePrefix)
    {
        _registry.TryResolve(modelId, out var info);

        info.Should().NotBeNull();
        info!.QueryPrefix.Should().Be(queryPrefix);
        info.PassagePrefix.Should().Be(passagePrefix);
    }

    [Theory]
    [InlineData("default")] // BGE-M3
    [InlineData("quality")] // BGE-M3
    [InlineData("all-mpnet-base-v2")]
    [InlineData("bge-base-en-v1.5")]
    [InlineData("bge-large-en-v1.5")]
    [InlineData("gte-large-en-v1.5")]
    public void NonPrefixModels_HaveNullQueryPassagePrefixes(string modelId)
    {
        _registry.TryResolve(modelId, out var info);

        info.Should().NotBeNull();
        info!.QueryPrefix.Should().BeNull();
        info.PassagePrefix.Should().BeNull();
    }
}
