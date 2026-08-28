using AwesomeAssertions;
using LMSupply;
using LMSupply.Exceptions;
using LMSupply.Reranker.Models;

namespace LMSupply.Reranker.Tests;

public class ModelRegistryTests
{
    private readonly RerankerModelRegistry _registry = RerankerModelRegistry.Default;

    [Theory]
    [InlineData("default")]
    [InlineData("DEFAULT")]
    [InlineData("Default")]
    public void Resolve_DefaultAlias_ShouldReturnMsMarcoMiniLM(string alias)
    {
        var model = _registry.Resolve(alias);

        model.Should().NotBeNull();
        model.Id.Should().Be("cross-encoder/ms-marco-MiniLM-L-6-v2");
        model.AliasName.Should().Be("default");
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
        model.AliasName.Should().Be("local");
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
    public void GetAvailableModels_ShouldReturnAllBuiltInModels()
    {
        var models = _registry.GetAvailableModels();

        models.Should().HaveCount(6);
    }

    [Fact]
    public void GetAliases_ShouldReturnAllAliasesAsAliasInfo()
    {
        var aliases = _registry.GetAliases();

        var aliasNames = aliases.Select(a => a.Name).ToList();
        aliasNames.Should().Contain(["auto", "default", "quality", "fast", "multilingual", "large"]);
        aliases.Should().AllSatisfy(a => a.Kind.Should().Be(AliasKind.System));
    }

    [Fact]
    public void RegisterAlias_ShouldBeResolvable()
    {
        var registry = new RerankerModelRegistry(DefaultModels.All);

        registry.RegisterAlias("my-model", "BAAI/bge-reranker-base");

        var model = registry.Resolve("my-model");
        model.Should().NotBeNull();
        model.Id.Should().Be("BAAI/bge-reranker-base");
    }

    [Fact]
    public void RegisterAlias_SystemAliasConflict_ShouldThrow()
    {
        var registry = new RerankerModelRegistry(DefaultModels.All);

        var act = () => registry.RegisterAlias("default", "BAAI/bge-reranker-base");

        act.Should().Throw<AliasConflictException>();
    }

    [Fact]
    public void RemoveAlias_UserAlias_ShouldSucceed()
    {
        var registry = new RerankerModelRegistry(DefaultModels.All);
        registry.RegisterAlias("temp-alias", "BAAI/bge-reranker-base");

        var removed = registry.RemoveAlias("temp-alias");

        removed.Should().BeTrue();
        var act = () => registry.Resolve("temp-alias");
        act.Should().Throw<ModelNotFoundException>();
    }

    [Fact]
    public void RemoveAlias_SystemAlias_ShouldReturnFalse()
    {
        var registry = new RerankerModelRegistry(DefaultModels.All);

        var removed = registry.RemoveAlias("default");

        removed.Should().BeFalse();
    }

    [Fact]
    public void GetAliases_WithUserAlias_ShouldIncludeBoth()
    {
        var registry = new RerankerModelRegistry(DefaultModels.All);
        registry.RegisterAlias("custom", "BAAI/bge-reranker-large");

        var aliases = registry.GetAliases();

        aliases.Should().Contain(a => a.Name == "custom" && a.Kind == AliasKind.User);
        aliases.Should().Contain(a => a.Name == "default" && a.Kind == AliasKind.System);
    }

    [Fact]
    public void Resolve_AutoAlias_ShouldReturnModel()
    {
        var model = _registry.Resolve("auto");

        model.Should().NotBeNull();
        model.AliasName.Should().Be("auto");
    }

    [Fact]
    public void LocalReranker_Registry_ShouldExpose()
    {
        var registry = LocalReranker.Registry;

        registry.Should().NotBeNull();
        registry.Should().BeAssignableTo<IModelRegistry<ModelInfo>>();
    }

    [Fact]
    public void LocalReranker_GetAvailableModels_ShouldReturnAliasNames()
    {
        var models = LocalReranker.GetAvailableModels().ToList();

        models.Should().Contain("default");
        models.Should().Contain("auto");
        models.Should().AllSatisfy(m => m.Should().BeOfType<string>());
    }

    [Fact]
    public void LocalReranker_CheckRuntimeAvailability_ReturnsWithoutCrash()
    {
        var (available, errorMessage) = LocalReranker.CheckRuntimeAvailability();

        if (available)
            errorMessage.Should().BeNull();
        else
            errorMessage.Should().NotBeNullOrEmpty();
    }

    [Fact]
    public void LocalReranker_IsModelDownloaded_ReturnsFalseForEmptyCache()
    {
        var tempDir = Path.Combine(Path.GetTempPath(), $"reranker-cache-{Guid.NewGuid():N}");
        try
        {
            var result = LocalReranker.IsModelDownloaded("default", tempDir);
            result.Should().BeFalse();
        }
        finally
        {
            if (Directory.Exists(tempDir))
                Directory.Delete(tempDir, recursive: true);
        }
    }

    [Fact]
    public void LocalReranker_IsModelDownloaded_ReturnsFalseForUnknownModel()
    {
        var tempDir = Path.Combine(Path.GetTempPath(), $"reranker-cache-{Guid.NewGuid():N}");
        try
        {
            var result = LocalReranker.IsModelDownloaded("nonexistent/model-xyz", tempDir);
            result.Should().BeFalse();
        }
        finally
        {
            if (Directory.Exists(tempDir))
                Directory.Delete(tempDir, recursive: true);
        }
    }
}
