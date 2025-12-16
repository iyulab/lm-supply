using FluentAssertions;
using LMSupply.Translator.Models;

namespace LMSupply.Translator.Tests;

public class LocalTranslatorTests
{
    [Fact]
    public void GetAvailableModels_ShouldReturnAliases()
    {
        var models = LocalTranslator.GetAvailableModels().ToList();

        models.Should().NotBeEmpty();
        models.Should().Contain("default");
        models.Should().Contain("ko-en");
        models.Should().Contain("en-ko");
    }

    [Fact]
    public void GetAllModels_ShouldReturnModelInfos()
    {
        var models = LocalTranslator.GetAllModels().ToList();

        models.Should().NotBeEmpty();
        models.Should().AllSatisfy(m =>
        {
            m.Id.Should().NotBeNullOrEmpty();
            m.Alias.Should().NotBeNullOrEmpty();
            m.SourceLanguage.Should().NotBeNullOrEmpty();
            m.TargetLanguage.Should().NotBeNullOrEmpty();
        });
    }

    [Fact]
    public void GetAllModels_ShouldContainExpectedLanguagePairs()
    {
        var models = LocalTranslator.GetAllModels().ToList();

        var koEnModels = models.Where(m => m.SourceLanguage == "ko" && m.TargetLanguage == "en");
        var enKoModels = models.Where(m => m.SourceLanguage == "en" && m.TargetLanguage == "ko");
        var jaEnModels = models.Where(m => m.SourceLanguage == "ja" && m.TargetLanguage == "en");
        var zhEnModels = models.Where(m => m.SourceLanguage == "zh" && m.TargetLanguage == "en");

        koEnModels.Should().NotBeEmpty();
        enKoModels.Should().NotBeEmpty();
        jaEnModels.Should().NotBeEmpty();
        zhEnModels.Should().NotBeEmpty();
    }

    [Fact]
    public void GetAllModels_DefaultModel_ShouldBeKoreanToEnglish()
    {
        var models = LocalTranslator.GetAllModels().ToList();
        var defaultModel = models.FirstOrDefault(m => m.Alias == "default");

        defaultModel.Should().NotBeNull();
        defaultModel!.SourceLanguage.Should().Be("ko");
        defaultModel.TargetLanguage.Should().Be("en");
    }

    [Theory]
    [InlineData("ko-en", "ko", "en")]
    [InlineData("en-ko", "en", "ko")]
    [InlineData("ja-en", "ja", "en")]
    [InlineData("zh-en", "zh", "en")]
    public void ResolveAlias_ShouldMatchExpectedLanguagePairs(
        string alias, string expectedSource, string expectedTarget)
    {
        // Use registry to resolve by alias since GetAllModels returns unique IDs
        var model = TranslatorModelRegistry.Default.Resolve(alias);

        model.Should().NotBeNull();
        model.SourceLanguage.Should().Be(expectedSource);
        model.TargetLanguage.Should().Be(expectedTarget);
    }

    [Fact]
    public void GetAllModels_AllModels_ShouldHaveValidEncoderDecoderFiles()
    {
        var models = LocalTranslator.GetAllModels();

        foreach (var model in models)
        {
            model.EncoderFile.Should().EndWith(".onnx");
            model.DecoderFile.Should().EndWith(".onnx");
        }
    }

    [Fact]
    public void GetAllModels_AllModels_ShouldHaveValidMaxLength()
    {
        var models = LocalTranslator.GetAllModels();

        foreach (var model in models)
        {
            model.MaxLength.Should().BeGreaterThan(0);
            model.MaxLength.Should().BeLessOrEqualTo(1024);
        }
    }

    [Fact]
    public void GetAllModels_AllModels_ShouldHaveCommerciallyCompatibleLicense()
    {
        var models = LocalTranslator.GetAllModels();
        var commercialLicenses = new[] { "Apache-2.0", "MIT", "BSD-3-Clause", "CC-BY-4.0" };

        foreach (var model in models)
        {
            commercialLicenses.Should().Contain(model.License,
                because: $"Model {model.Id} should have commercially compatible license");
        }
    }

    [Fact]
    public void GetAvailableModels_Count_ShouldBeGreaterOrEqualToGetAllModels()
    {
        // Aliases can be >= models because multiple aliases can point to the same model
        // (e.g., "default" and "ko-en" both point to the same Korean-English model)
        var aliases = LocalTranslator.GetAvailableModels().ToList();
        var models = LocalTranslator.GetAllModels().ToList();

        aliases.Count.Should().BeGreaterThanOrEqualTo(models.Count);
    }
}
