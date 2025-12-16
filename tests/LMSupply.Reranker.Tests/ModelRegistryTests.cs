using FluentAssertions;
using LMSupply.Exceptions;
using LMSupply.Reranker.Models;

namespace LMSupply.Reranker.Tests;

public class ModelRegistryTests
{
    private readonly ModelRegistry _registry = ModelRegistry.Default;

    [Theory]
    [InlineData("default")]
    [InlineData("DEFAULT")]
    [InlineData("Default")]
    public void Resolve_DefaultAlias_ShouldReturnMsMarcoMiniLM(string alias)
    {
        var model = _registry.Resolve(alias);

        model.Should().NotBeNull();
        model.Id.Should().Be("cross-encoder/ms-marco-MiniLM-L-6-v2");
        model.Alias.Should().Be("default");
    }

    [Theory]
    [InlineData("quality", "BAAI/bge-reranker-base")]
    [InlineData("fast", "cross-encoder/ms-marco-TinyBERT-L-2-v2")]
    [InlineData("multilingual", "BAAI/bge-reranker-v2-m3")]
    public void Resolve_BuiltInAliases_ShouldReturnCorrectModel(string alias, string expectedId)
    {
        var model = _registry.Resolve(alias);

        model.Should().NotBeNull();
        model.Id.Should().Be(expectedId);
    }

    [Fact]
    public void Resolve_FullModelId_ShouldReturnModel()
    {
        var model = _registry.Resolve("cross-encoder/ms-marco-MiniLM-L-6-v2");

        model.Should().NotBeNull();
        model.DisplayName.Should().Be("MS MARCO MiniLM L6");
    }

    [Fact]
    public void Resolve_UnknownHuggingFaceId_ShouldCreateGenericModel()
    {
        var model = _registry.Resolve("some-org/some-model");

        model.Should().NotBeNull();
        model.Id.Should().Be("some-org/some-model");
        model.OnnxFile.Should().Be("onnx/model.onnx");
    }

    [Fact]
    public void Resolve_LocalPath_ShouldCreateLocalModel()
    {
        var model = _registry.Resolve("./models/custom.onnx");

        model.Should().NotBeNull();
        model.Alias.Should().Be("local");
        model.OnnxFile.Should().Be("custom.onnx");
    }

    [Fact]
    public void Resolve_UnknownAlias_ShouldThrow()
    {
        var act = () => _registry.Resolve("nonexistent");

        act.Should().Throw<ModelNotFoundException>()
            .Where(e => e.ModelId == "nonexistent");
    }

    [Fact]
    public void TryResolve_ValidAlias_ShouldReturnTrue()
    {
        var success = _registry.TryResolve("default", out var model);

        success.Should().BeTrue();
        model.Should().NotBeNull();
    }

    [Fact]
    public void TryResolve_InvalidAlias_ShouldReturnFalse()
    {
        var success = _registry.TryResolve("nonexistent", out var model);

        success.Should().BeFalse();
        model.Should().BeNull();
    }

    [Fact]
    public void GetAll_ShouldReturnAllBuiltInModels()
    {
        var models = _registry.GetAll().ToList();

        models.Should().HaveCount(6);
    }

    [Fact]
    public void GetAliases_ShouldReturnAllAliases()
    {
        var aliases = _registry.GetAliases().ToList();

        aliases.Should().Contain(["default", "quality", "fast", "multilingual", "large"]);
    }
}
