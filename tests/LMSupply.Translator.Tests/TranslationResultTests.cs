using AwesomeAssertions;

namespace LMSupply.Translator.Tests;

public class TranslationResultTests
{
    [Fact]
    public void Create_WithRequiredProperties_ShouldSucceed()
    {
        var result = new TranslationResult
        {
            SourceText = "Hello world",
            TranslatedText = "안녕하세요 세계",
            SourceLanguage = "en",
            TargetLanguage = "ko"
        };

        result.SourceText.Should().Be("Hello world");
        result.TranslatedText.Should().Be("안녕하세요 세계");
        result.SourceLanguage.Should().Be("en");
        result.TargetLanguage.Should().Be("ko");
        result.Confidence.Should().BeNull();
        result.InferenceTimeMs.Should().BeNull();
    }

    [Fact]
    public void Create_WithAllProperties_ShouldSucceed()
    {
        var result = new TranslationResult
        {
            SourceText = "안녕하세요",
            TranslatedText = "Hello",
            SourceLanguage = "ko",
            TargetLanguage = "en",
            Confidence = 0.95f,
            InferenceTimeMs = 150.5
        };

        result.SourceText.Should().Be("안녕하세요");
        result.TranslatedText.Should().Be("Hello");
        result.SourceLanguage.Should().Be("ko");
        result.TargetLanguage.Should().Be("en");
        result.Confidence.Should().Be(0.95f);
        result.InferenceTimeMs.Should().Be(150.5);
    }

    [Theory]
    [InlineData("en", "ko")]
    [InlineData("ko", "en")]
    [InlineData("ja", "en")]
    [InlineData("zh", "en")]
    public void Create_WithDifferentLanguagePairs_ShouldSucceed(string source, string target)
    {
        var result = new TranslationResult
        {
            SourceText = "Test",
            TranslatedText = "Translated",
            SourceLanguage = source,
            TargetLanguage = target
        };

        result.SourceLanguage.Should().Be(source);
        result.TargetLanguage.Should().Be(target);
    }

    [Theory]
    [InlineData(0.0f)]
    [InlineData(0.5f)]
    [InlineData(1.0f)]
    public void Confidence_ShouldAcceptValidRange(float confidence)
    {
        var result = new TranslationResult
        {
            SourceText = "Test",
            TranslatedText = "Test",
            SourceLanguage = "en",
            TargetLanguage = "ko",
            Confidence = confidence
        };

        result.Confidence.Should().Be(confidence);
    }

    [Fact]
    public void InferenceTimeMs_ShouldAcceptPositiveValues()
    {
        var result = new TranslationResult
        {
            SourceText = "Test",
            TranslatedText = "Test",
            SourceLanguage = "en",
            TargetLanguage = "ko",
            InferenceTimeMs = 0.001
        };

        result.InferenceTimeMs.Should().BeGreaterThan(0);
    }

    [Fact]
    public void Create_WithEmptySourceText_ShouldSucceed()
    {
        var result = new TranslationResult
        {
            SourceText = "",
            TranslatedText = "",
            SourceLanguage = "en",
            TargetLanguage = "ko"
        };

        result.SourceText.Should().BeEmpty();
        result.TranslatedText.Should().BeEmpty();
    }

    [Fact]
    public void Create_WithLongText_ShouldSucceed()
    {
        var longText = string.Join(" ", Enumerable.Repeat("word", 1000));

        var result = new TranslationResult
        {
            SourceText = longText,
            TranslatedText = longText,
            SourceLanguage = "en",
            TargetLanguage = "ko"
        };

        result.SourceText.Length.Should().BeGreaterThan(4000);
    }

    [Fact]
    public void Create_WithUnicodeText_ShouldSucceed()
    {
        var result = new TranslationResult
        {
            SourceText = "Hello 世界 🌍",
            TranslatedText = "안녕 세계 🌍",
            SourceLanguage = "en",
            TargetLanguage = "ko"
        };

        result.SourceText.Should().Contain("🌍");
        result.TranslatedText.Should().Contain("🌍");
    }
}
