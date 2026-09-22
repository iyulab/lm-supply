using AwesomeAssertions;
using LMSupply.Embedder.Utils;

namespace LMSupply.Embedder.Tests;

/// <summary>
/// <c>TryResolve</c> answers any <c>org/repo</c> with a fallback entry whose dimensions, pooling, length
/// and subfolder are placeholders. A loader that takes that for a declaration reads the model wrongly —
/// which is what happened: the repository-id branch that reads the model's own files was unreachable.
/// <c>TryResolveCatalog</c> says only what the catalog knows.
/// </summary>
public sealed class TryResolveCatalogTests
{
    private readonly EmbedderModelRegistry _registry = EmbedderModelRegistry.Default;

    [Theory]
    [InlineData("default")]
    [InlineData("multilingual-e5-small")]
    [InlineData("BAAI/bge-m3")]
    [InlineData("auto")]
    public void ACatalogedIdOrAlias_Resolves(string id)
    {
        _registry.TryResolveCatalog(id, out var info, out _).Should().BeTrue();
        info.Should().NotBeNull();
    }

    [Fact]
    public void ARepositoryTheCatalogDoesNotKnow_IsNotCataloged_ButTryResolveStillFabricatesAnEntry()
    {
        _registry.TryResolveCatalog("BAAI/bge-small-en-v1.5", out var info, out var resolvedId).Should().BeFalse();
        info.Should().BeNull();
        resolvedId.Should().Be("BAAI/bge-small-en-v1.5");

        _registry.TryResolve("BAAI/bge-small-en-v1.5", out var fallback).Should().BeTrue("the fallback still exists for callers that only need a repo id");
        fallback!.PoolingMode.Should().Be(PoolingMode.Mean, "a placeholder, not a declaration — the loader must not read it as one");
    }
}
