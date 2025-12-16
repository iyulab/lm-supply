using FluentAssertions;
using LMSupply.Detector.Models;

namespace LMSupply.Detector.Tests;

public class DetectorModelRegistryTests
{
    private readonly DetectorModelRegistry _registry = DetectorModelRegistry.Default;

    [Theory]
    [InlineData("default")]
    [InlineData("DEFAULT")]
    [InlineData("Default")]
    public void Resolve_DefaultAlias_ShouldReturnRtDetrR18(string alias)
    {
        var model = _registry.Resolve(alias);

        model.Should().NotBeNull();
        model.Id.Should().Be("PekingU/rtdetr_r18vd");
        model.Alias.Should().Be("default");
    }

    [Theory]
    [InlineData("quality", "PekingU/rtdetr_r50vd")]
    [InlineData("fast", "Kalray/efficientdet-d0")]
    [InlineData("large", "PekingU/rtdetr_r101vd")]
    public void Resolve_BuiltInAliases_ShouldReturnCorrectModel(string alias, string expectedId)
    {
        var model = _registry.Resolve(alias);

        model.Should().NotBeNull();
        model.Id.Should().Be(expectedId);
    }

    [Fact]
    public void Resolve_FullModelId_ShouldReturnModel()
    {
        var model = _registry.Resolve("PekingU/rtdetr_r18vd");

        model.Should().NotBeNull();
        model.DisplayName.Should().Be("RT-DETR R18");
    }

    [Fact]
    public void Resolve_UnknownHuggingFaceId_ShouldCreateGenericModel()
    {
        var model = _registry.Resolve("some-org/some-detector");

        model.Should().NotBeNull();
        model.Id.Should().Be("some-org/some-detector");
        model.OnnxFile.Should().Be("model.onnx");
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

        models.Should().HaveCount(4);
    }

    [Fact]
    public void GetAliases_ShouldReturnAllAliases()
    {
        var aliases = _registry.GetAliases().ToList();

        aliases.Should().Contain(["default", "quality", "fast", "large"]);
    }

    [Theory]
    [InlineData("PekingU/rtdetr_r18vd", "RT-DETR", false)]
    [InlineData("Kalray/efficientdet-d0", "EfficientDet", true)]
    public void DefaultModels_ShouldHaveCorrectNmsSettings(string modelId, string architecture, bool requiresNms)
    {
        var model = _registry.Resolve(modelId);

        model.Architecture.Should().Be(architecture);
        model.RequiresNms.Should().Be(requiresNms);
    }

    [Fact]
    public void DefaultModels_ShouldAllHaveApache2License()
    {
        var models = _registry.GetAll();

        models.Should().OnlyContain(m => m.License == "Apache-2.0");
    }
}
