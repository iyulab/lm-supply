using FluentAssertions;
using LMSupply.Translator.Models;

namespace LMSupply.Translator.Tests;

public class TranslatorModelRegistryTests
{
    private readonly TranslatorModelRegistry _registry = TranslatorModelRegistry.Default;

    [Fact]
    public void Default_ShouldContainAllDefaultModels()
    {
        var models = _registry.GetAll().ToList();
        models.Should().NotBeEmpty();
    }

    [Fact]
    public void GetAliases_ShouldContainExpectedAliases()
    {
        var aliases = _registry.GetAliases().ToList();

        aliases.Should().Contain("default");
        aliases.Should().Contain("ko-en");
        aliases.Should().Contain("en-ko");
        aliases.Should().Contain("ja-en");
        aliases.Should().Contain("zh-en");
    }

    [Theory]
    [InlineData("default")]
    [InlineData("ko-en")]
    [InlineData("en-ko")]
    [InlineData("ja-en")]
    [InlineData("zh-en")]
    public void Resolve_WithValidAlias_ShouldReturnModelInfo(string alias)
    {
        var model = _registry.Resolve(alias);

        model.Should().NotBeNull();
        model.Alias.Should().Be(alias);
    }

    [Fact]
    public void Resolve_WithDefaultAlias_ShouldReturnKoEnModel()
    {
        var model = _registry.Resolve("default");

        model.SourceLanguage.Should().Be("ko");
        model.TargetLanguage.Should().Be("en");
        model.Architecture.Should().Be("MarianMT");
    }

    [Theory]
    [InlineData("Helsinki-NLP/opus-mt-ko-en")]
    [InlineData("Helsinki-NLP/opus-mt-en-ko")]
    [InlineData("Helsinki-NLP/opus-mt-ja-en")]
    [InlineData("Helsinki-NLP/opus-mt-zh-en")]
    public void Resolve_WithHuggingFaceId_ShouldReturnModelInfo(string modelId)
    {
        var model = _registry.Resolve(modelId);

        model.Should().NotBeNull();
        model.Id.Should().Be(modelId);
    }

    [Theory]
    [InlineData("./model.onnx")]
    [InlineData("../model.onnx")]
    [InlineData(".\\model.onnx")]
    [InlineData("..\\model.onnx")]
    public void Resolve_WithRelativePath_ShouldCreateLocalModelInfo(string path)
    {
        var model = _registry.Resolve(path);

        model.Should().NotBeNull();
        model.Alias.Should().Be("local");
    }

    [Fact]
    public void Resolve_WithUnknownHuggingFaceId_ShouldCreateGenericModelInfo()
    {
        var model = _registry.Resolve("org/unknown-model");

        model.Should().NotBeNull();
        model.Id.Should().Be("org/unknown-model");
        model.Alias.Should().Be("org/unknown-model");
    }

    [Fact]
    public void Resolve_WithOpusMtModelId_ShouldDetectLanguagePair()
    {
        var model = _registry.Resolve("Helsinki-NLP/opus-mt-de-en");

        model.SourceLanguage.Should().Be("de");
        model.TargetLanguage.Should().Be("en");
    }

    [Fact]
    public void Resolve_WithNullOrEmpty_ShouldThrowException()
    {
        var act1 = () => _registry.Resolve(null!);
        var act2 = () => _registry.Resolve("");
        var act3 = () => _registry.Resolve("   ");

        act1.Should().Throw<ArgumentException>();
        act2.Should().Throw<ArgumentException>();
        act3.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void Resolve_WithUnknownAlias_ShouldThrowModelNotFoundException()
    {
        var act = () => _registry.Resolve("nonexistent");

        act.Should().Throw<ModelNotFoundException>()
            .Where(e => e.ModelId == "nonexistent");
    }

    [Theory]
    [InlineData("default", true)]
    [InlineData("ko-en", true)]
    [InlineData("nonexistent", false)]
    [InlineData("Helsinki-NLP/opus-mt-ko-en", true)]
    public void TryResolve_ShouldReturnExpectedResult(string modelIdOrAlias, bool expectedResult)
    {
        var result = _registry.TryResolve(modelIdOrAlias, out var modelInfo);

        result.Should().Be(expectedResult);
        if (expectedResult)
        {
            modelInfo.Should().NotBeNull();
        }
        else
        {
            modelInfo.Should().BeNull();
        }
    }

    [Fact]
    public void CustomRegistry_ShouldAcceptCustomModels()
    {
        var customModels = new List<TranslatorModelInfo>
        {
            new()
            {
                Id = "custom/model",
                Alias = "custom",
                DisplayName = "Custom Model",
                Architecture = "Custom",
                SourceLanguage = "en",
                TargetLanguage = "fr",
                EncoderFile = "encoder.onnx",
                DecoderFile = "decoder.onnx"
            }
        };

        var customRegistry = new TranslatorModelRegistry(customModels);
        var model = customRegistry.Resolve("custom");

        model.Id.Should().Be("custom/model");
        model.SourceLanguage.Should().Be("en");
        model.TargetLanguage.Should().Be("fr");
    }

    [Fact]
    public void ModelInfo_ShouldHaveCorrectEncoderDecoderFiles()
    {
        var model = _registry.Resolve("default");

        model.EncoderFile.Should().Be("encoder_model.onnx");
        model.DecoderFile.Should().Be("decoder_model.onnx");
    }

    [Fact]
    public void ModelInfo_ShouldHaveCorrectLicense()
    {
        var model = _registry.Resolve("default");

        model.License.Should().Be("Apache-2.0");
    }

    [Fact]
    public void AllDefaultModels_ShouldHaveApacheLicense()
    {
        var models = _registry.GetAll();

        foreach (var model in models)
        {
            model.License.Should().Be("Apache-2.0",
                because: $"Model {model.Id} should have Apache-2.0 license for commercial safety");
        }
    }

    [Fact]
    public void AllDefaultModels_ShouldHaveMarianMTArchitecture()
    {
        var models = _registry.GetAll();

        foreach (var model in models)
        {
            model.Architecture.Should().Be("MarianMT");
        }
    }
}
