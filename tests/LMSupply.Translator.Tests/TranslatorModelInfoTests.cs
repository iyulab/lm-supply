using FluentAssertions;
using LMSupply.Translator.Models;

namespace LMSupply.Translator.Tests;

public class TranslatorModelInfoTests
{
    [Fact]
    public void Create_WithRequiredProperties_ShouldSucceed()
    {
        var modelInfo = new TranslatorModelInfo
        {
            Id = "test/model",
            Alias = "test",
            DisplayName = "Test Model",
            Architecture = "MarianMT",
            SourceLanguage = "en",
            TargetLanguage = "ko"
        };

        modelInfo.Id.Should().Be("test/model");
        modelInfo.Alias.Should().Be("test");
        modelInfo.DisplayName.Should().Be("Test Model");
        modelInfo.Architecture.Should().Be("MarianMT");
        modelInfo.SourceLanguage.Should().Be("en");
        modelInfo.TargetLanguage.Should().Be("ko");
    }

    [Fact]
    public void Create_DefaultValues_ShouldBeSet()
    {
        var modelInfo = new TranslatorModelInfo
        {
            Id = "test/model",
            Alias = "test",
            DisplayName = "Test",
            Architecture = "MarianMT",
            SourceLanguage = "en",
            TargetLanguage = "ko"
        };

        modelInfo.MaxLength.Should().Be(512);
        modelInfo.EncoderFile.Should().Be("encoder_model.onnx");
        modelInfo.DecoderFile.Should().Be("decoder_model.onnx");
        modelInfo.TokenizerFile.Should().Be("source.spm");
        modelInfo.Description.Should().BeEmpty();
        modelInfo.License.Should().Be("Unknown");
        modelInfo.ParametersM.Should().Be(0);
        modelInfo.SizeBytes.Should().Be(0);
        modelInfo.BleuScore.Should().Be(0);
        modelInfo.VocabSize.Should().Be(0);
    }

    [Fact]
    public void Create_WithAllProperties_ShouldSucceed()
    {
        var modelInfo = new TranslatorModelInfo
        {
            Id = "Helsinki-NLP/opus-mt-ko-en",
            Alias = "ko-en",
            DisplayName = "OPUS-MT Ko-En",
            Architecture = "MarianMT",
            SourceLanguage = "ko",
            TargetLanguage = "en",
            ParametersM = 74f,
            SizeBytes = 300_000_000,
            BleuScore = 35.5f,
            MaxLength = 512,
            VocabSize = 65000,
            EncoderFile = "encoder_model.onnx",
            DecoderFile = "decoder_model.onnx",
            TokenizerFile = "source.spm",
            Description = "Korean to English translation",
            License = "Apache-2.0"
        };

        modelInfo.ParametersM.Should().Be(74f);
        modelInfo.SizeBytes.Should().Be(300_000_000);
        modelInfo.BleuScore.Should().Be(35.5f);
        modelInfo.VocabSize.Should().Be(65000);
        modelInfo.TokenizerFile.Should().Be("source.spm");
        modelInfo.Description.Should().Be("Korean to English translation");
        modelInfo.License.Should().Be("Apache-2.0");
    }

    [Theory]
    [InlineData("MarianMT")]
    [InlineData("NLLB")]
    [InlineData("M2M-100")]
    public void Architecture_ShouldAcceptKnownValues(string architecture)
    {
        var modelInfo = new TranslatorModelInfo
        {
            Id = "test/model",
            Alias = "test",
            DisplayName = "Test",
            Architecture = architecture,
            SourceLanguage = "en",
            TargetLanguage = "ko"
        };

        modelInfo.Architecture.Should().Be(architecture);
    }

    [Theory]
    [InlineData("ko", "en")]
    [InlineData("en", "ko")]
    [InlineData("ja", "en")]
    [InlineData("zh", "en")]
    [InlineData("de", "en")]
    [InlineData("fr", "en")]
    public void LanguagePair_ShouldAcceptVariousLanguages(string source, string target)
    {
        var modelInfo = new TranslatorModelInfo
        {
            Id = "test/model",
            Alias = "test",
            DisplayName = "Test",
            Architecture = "MarianMT",
            SourceLanguage = source,
            TargetLanguage = target
        };

        modelInfo.SourceLanguage.Should().Be(source);
        modelInfo.TargetLanguage.Should().Be(target);
    }

    [Theory]
    [InlineData(256)]
    [InlineData(512)]
    [InlineData(1024)]
    public void MaxLength_ShouldAcceptValidValues(int maxLength)
    {
        var modelInfo = new TranslatorModelInfo
        {
            Id = "test/model",
            Alias = "test",
            DisplayName = "Test",
            Architecture = "MarianMT",
            SourceLanguage = "en",
            TargetLanguage = "ko",
            MaxLength = maxLength
        };

        modelInfo.MaxLength.Should().Be(maxLength);
    }

    [Fact]
    public void DefaultModels_OpusMtKoEn_ShouldHaveCorrectValues()
    {
        var model = DefaultModels.OpusMtKoEn;

        model.Id.Should().Be("Helsinki-NLP/opus-mt-ko-en");
        model.Alias.Should().Be("default");
        model.SourceLanguage.Should().Be("ko");
        model.TargetLanguage.Should().Be("en");
        model.Architecture.Should().Be("MarianMT");
        model.License.Should().Be("Apache-2.0");
    }

    [Fact]
    public void DefaultModels_All_ShouldContainExpectedCount()
    {
        DefaultModels.All.Should().HaveCountGreaterOrEqualTo(4);
    }

    [Fact]
    public void DefaultModels_All_ShouldContainKoEnModel()
    {
        DefaultModels.All.Should().Contain(m => m.Alias == "ko-en");
    }

    [Fact]
    public void DefaultModels_All_ShouldContainEnKoModel()
    {
        DefaultModels.All.Should().Contain(m => m.Alias == "en-ko");
    }
}
