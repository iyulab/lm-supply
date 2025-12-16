using FluentAssertions;
using LMSupply.Segmenter.Models;

namespace LMSupply.Segmenter.Tests;

public class SegmenterModelRegistryTests
{
    private readonly SegmenterModelRegistry _registry = SegmenterModelRegistry.Default;

    [Theory]
    [InlineData("default")]
    [InlineData("DEFAULT")]
    [InlineData("Default")]
    public void Resolve_DefaultAlias_ShouldReturnSegFormerB0(string alias)
    {
        var model = _registry.Resolve(alias);

        model.Should().NotBeNull();
        model.Id.Should().Be("nvidia/segformer-b0-finetuned-ade-512-512");
        model.Alias.Should().Be("default");
    }

    [Theory]
    [InlineData("quality", "nvidia/segformer-b2-finetuned-ade-512-512")]
    [InlineData("fast", "nvidia/segformer-b1-finetuned-ade-512-512")]
    [InlineData("large", "nvidia/segformer-b5-finetuned-ade-640-640")]
    public void Resolve_BuiltInAliases_ShouldReturnCorrectModel(string alias, string expectedId)
    {
        var model = _registry.Resolve(alias);

        model.Should().NotBeNull();
        model.Id.Should().Be(expectedId);
    }

    [Fact]
    public void Resolve_FullModelId_ShouldReturnModel()
    {
        var model = _registry.Resolve("nvidia/segformer-b0-finetuned-ade-512-512");

        model.Should().NotBeNull();
        model.DisplayName.Should().Be("SegFormer-B0");
    }

    [Fact]
    public void Resolve_UnknownHuggingFaceId_ShouldCreateGenericModel()
    {
        var model = _registry.Resolve("some-org/some-segmenter");

        model.Should().NotBeNull();
        model.Id.Should().Be("some-org/some-segmenter");
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

        models.Should().HaveCount(5); // 4 SegFormer + 1 MobileSAM
    }

    [Fact]
    public void GetAliases_ShouldReturnAllAliases()
    {
        var aliases = _registry.GetAliases().ToList();

        aliases.Should().Contain(["default", "quality", "fast", "large", "interactive"]);
    }

    [Fact]
    public void DefaultModels_SegFormerModels_ShouldHaveCorrectArchitecture()
    {
        var models = _registry.GetAll()
            .Where(m => m.Architecture == "SegFormer");

        models.Should().HaveCount(4);
        models.Should().OnlyContain(m => m.License == "MIT");
    }

    [Fact]
    public void DefaultModels_MobileSAM_ShouldHaveCorrectProperties()
    {
        var model = _registry.Resolve("interactive");

        model.Architecture.Should().Be("MobileSAM");
        model.Id.Should().Be("ChaoningZhang/MobileSAM");
        model.License.Should().Be("Apache-2.0");
        model.IsInteractive.Should().BeTrue();
        model.EncoderFile.Should().Be("mobile_sam_image_encoder.onnx");
        model.DecoderFile.Should().Be("mobile_sam_mask_decoder.onnx");
    }

    [Fact]
    public void DefaultModels_NonInteractiveModels_ShouldNotHaveEncoderDecoder()
    {
        var models = _registry.GetAll()
            .Where(m => !m.IsInteractive);

        models.Should().OnlyContain(m => m.EncoderFile == null);
        models.Should().OnlyContain(m => m.DecoderFile == null);
    }
}
