using AwesomeAssertions;
using LMSupply.Ocr.Models;

namespace LMSupply.Ocr.Tests;

/// <summary>
/// Which recognizer a language code reaches, and what happens when it reaches none.
///
/// <para>
/// The language lists here come from the recognizers' own character dictionaries, not from a guess:
/// the Chinese model's dictionary holds every hiragana and katakana, and the Latin model's holds the
/// letters that set Turkish, Hungarian, Slovak and the rest apart from English. Before they were
/// listed, those codes fell through to the English recognizer — whose dictionary has none of those
/// letters — without saying so. A caller following the OS locale got the wrong model and no signal.
/// </para>
/// </summary>
public class OcrLanguageCoverageTests
{
    [Theory]
    [InlineData("ja")]
    [InlineData("ja-JP")]
    public void Japanese_ReachesTheChineseRecognizer_WhoseDictionaryHoldsKana(string code)
    {
        OcrRecognitionModelRegistry.Default.ResolveForLanguage(code).AliasName.Should().Be("crnn-chinese-v5");
    }

    [Theory]
    [InlineData("tr")]
    [InlineData("tr-TR")]
    [InlineData("hu")]
    [InlineData("sk")]
    [InlineData("hr")]
    [InlineData("bs")]
    [InlineData("sl")]
    [InlineData("is")]
    [InlineData("et")]
    [InlineData("lt")]
    [InlineData("sq")]
    [InlineData("ga")]
    [InlineData("id")]
    [InlineData("ms")]
    [InlineData("sw")]
    [InlineData("tl")]
    [InlineData("fil-PH")]
    [InlineData("nb-NO")]
    [InlineData("nn")]
    [InlineData("lv")]
    [InlineData("cy")]
    [InlineData("af")]
    public void LatinScriptLanguages_ReachTheLatinRecognizer(string code)
    {
        OcrRecognitionModelRegistry.Default.ResolveForLanguage(code).AliasName.Should().Be("crnn-latin-v5");
    }

    [Fact]
    public void EveryLanguageCode_ReachesTheRecognizerThatOwnsIt()
    {
        var registry = OcrRecognitionModelRegistry.Default;
        var primaries = registry.GetAvailableModels()
            .Where(m => m.AliasName.StartsWith("crnn-", StringComparison.OrdinalIgnoreCase))
            .ToList();
        primaries.Should().HaveCountGreaterThan(1);

        foreach (var primary in primaries)
        {
            foreach (var code in primary.LanguageCodes)
            {
                registry.ResolveForLanguage(code).AliasName.Should().Be(primary.AliasName,
                    $"'{code}' is a language of {primary.AliasName}, and models that live in a repository of their own have no subfolder to tell them apart by");
            }
        }
    }

    [Theory]
    [InlineData("ta", "crnn-tamil-v5")]
    [InlineData("te", "crnn-telugu-v5")]
    [InlineData("hi", "crnn-devanagari-v5")]
    [InlineData("ar", "crnn-arabic-v5")]
    [InlineData("ru", "crnn-cyrillic-v5")]
    [InlineData("sr", "crnn-cyrillic-v5")]
    [InlineData("th", "crnn-thai-v5")]
    [InlineData("el", "crnn-greek-v5")]
    public void ALanguage_ReachesItsOwnRecognizer(string code, string alias)
    {
        OcrRecognitionModelRegistry.Default.ResolveForLanguage(code).AliasName.Should().Be(alias);
    }

    // How callers and operating systems write a language: BCP 47 with a hyphen, or a POSIX locale name with an
    // underscore, an encoding and a modifier. The one language must reach the one recognizer either way.
    [Theory]
    [InlineData("ko-KR", "crnn-korean-v5")]
    [InlineData("ko_KR", "crnn-korean-v5")]
    [InlineData("KO_kr", "crnn-korean-v5")]
    [InlineData("ko_KR.UTF-8", "crnn-korean-v5")]
    [InlineData("zh-CN", "crnn-chinese-v5")]
    [InlineData("zh_CN", "crnn-chinese-v5")]
    [InlineData("zh_TW", "crnn-chinese-v5")]
    [InlineData("zh-Hans", "crnn-chinese-v5")]
    [InlineData("zh-Hant-TW", "crnn-chinese-v5")]
    [InlineData("pt_PT", "crnn-latin-v5")]
    [InlineData("pt-BR", "crnn-latin-v5")]
    [InlineData("en_US", "crnn-en-v5")]
    [InlineData("  en  ", "crnn-en-v5")]
    public void ARegionedOrUnderscoredCode_ReachesTheRecognizerOfItsLanguage(string code, string alias)
    {
        OcrRecognitionModelRegistry.Default.ResolveForLanguage(code).AliasName.Should().Be(alias);
    }

    // Serbian is written in two scripts and the script, not the region, picks the recognizer: the bare code and the
    // Cyrillic tag are the Cyrillic recognizer's, the Latin tag is the Latin recognizer's (its dictionary holds
    // Croatian's č ć đ š ž, which are Serbian Latin's too).
    [Theory]
    [InlineData("sr", "crnn-cyrillic-v5")]
    [InlineData("sr-RS", "crnn-cyrillic-v5")]
    [InlineData("sr-Cyrl", "crnn-cyrillic-v5")]
    [InlineData("sr-Cyrl-RS", "crnn-cyrillic-v5")]
    [InlineData("sr-Latn", "crnn-latin-v5")]
    [InlineData("sr_Latn", "crnn-latin-v5")]
    [InlineData("sr-Latn-RS", "crnn-latin-v5")]
    [InlineData("sr_RS@latin", "crnn-latin-v5")]
    [InlineData("sr_RS.UTF-8@latin", "crnn-latin-v5")]
    [InlineData("sr_RS@cyrillic", "crnn-cyrillic-v5")]
    public void TheScriptOfALanguageWrittenInTwo_PicksTheRecognizer(string code, string alias)
    {
        OcrRecognitionModelRegistry.Default.ResolveForLanguage(code).AliasName.Should().Be(alias);
    }

    // A code that names a script its language's recognizer does not read is refused, not read with that recognizer:
    // "bs" is the Latin recognizer's, and Bosnian in Cyrillic would come back as text in letters it has no dictionary for.
    [Theory]
    [InlineData("bs-Cyrl")]
    [InlineData("mk-Latn")]
    [InlineData("ru-Latn")]
    [InlineData("hi-Latn")]
    [InlineData("zh-Latn")]
    [InlineData("ja-Latn")]
    [InlineData("ar-Latn")]
    public void ACodeNamingAScriptItsLanguagesRecognizerDoesNotRead_IsRefused(string code)
    {
        OcrRecognitionModelRegistry.Default.Invoking(r => r.ResolveForLanguage(code))
            .Should().Throw<UnsupportedOcrLanguageException>().Which.LanguageCode.Should().Be(code);
    }

    [Theory]
    [InlineData("en-Latn", "crnn-en-v5")]
    [InlineData("de-Latn-DE", "crnn-latin-v5")]
    [InlineData("ru-Cyrl", "crnn-cyrillic-v5")]
    [InlineData("mk-Cyrl-MK", "crnn-cyrillic-v5")]
    [InlineData("ko-Kore", "crnn-korean-v5")]
    [InlineData("ja-Jpan", "crnn-chinese-v5")]
    [InlineData("zh-Hant-TW", "crnn-chinese-v5")]
    [InlineData("ar-Arab", "crnn-arabic-v5")]
    [InlineData("th_TH.UTF-8", "crnn-thai-v5")]
    public void ACodeNamingTheScriptItsRecognizerReads_ReachesIt(string code, string alias)
    {
        OcrRecognitionModelRegistry.Default.ResolveForLanguage(code).AliasName.Should().Be(alias);
    }

    [Theory]
    [InlineData("cmn", "crnn-chinese-v5")]
    [InlineData("yue", "crnn-chinese-v5")]
    [InlineData("cmn-Hans-CN", "crnn-chinese-v5")]
    public void ACodeThatNamesOneSupportedLanguageUnderAnotherCode_ReachesItsRecognizer(string code, string alias)
    {
        OcrRecognitionModelRegistry.Default.ResolveForLanguage(code).AliasName.Should().Be(alias);
    }

    // The English recognizer reads another script's text as confident garbage and a caller cannot tell from the
    // result, so a language without a model is refused instead of being read with it. "vi" and "uz" are Latin-script
    // languages whose letters the dictionary lacks (Vietnamese tone marks, Uzbek's ʻ). "sh" and "mo" are legacy codes
    // with two readings each; "iw" is Hebrew's former code, and Hebrew has no model.
    [Theory]
    [InlineData("he")]
    [InlineData("iw")]
    [InlineData("kn")]
    [InlineData("ml")]
    [InlineData("bn")]
    [InlineData("km")]
    [InlineData("vi")]
    [InlineData("uz")]
    [InlineData("uz-Latn")]
    [InlineData("uz-Cyrl")]
    [InlineData("sh")]
    [InlineData("mo")]
    [InlineData("vi_VN")]
    [InlineData("xx-unsupported-test")]
    public void ALanguageWithoutAModel_IsRefused_NotReadByAnotherRecognizer(string code)
    {
        var act = () => OcrRecognitionModelRegistry.Default.ResolveForLanguage(code);

        var thrown = act.Should().Throw<UnsupportedOcrLanguageException>().Which;
        thrown.LanguageCode.Should().Be(code, "the message names the code as the caller wrote it");
        thrown.SupportedLanguages.Should().Contain(["en", "ko", "sr", "sr-latn"],
            "what the caller does next is pick a code that works, so the exception carries the list");
        thrown.Message.Should().Contain($"'{code}'").And.Contain("ko").And.Contain("\"en\"");
        thrown.Should().BeAssignableTo<ArgumentException>();
    }

    [Theory]
    [InlineData("en")]
    [InlineData("default")]
    public void TheEnglishRecognizer_IsReachedByAskingForItAndNotByAskingForSomethingElse(string code)
    {
        OcrRecognitionModelRegistry.Default.ResolveForLanguage(code).AliasName.Should().Be("crnn-en-v5");
    }

    [Fact]
    public void TryResolveForLanguage_AnswersFalseForALanguageWithoutAModel_AndNeverSubstitutes()
    {
        var registry = OcrRecognitionModelRegistry.Default;

        registry.TryResolveForLanguage("ko_KR", out var korean).Should().BeTrue();
        korean!.AliasName.Should().Be("crnn-korean-v5");

        registry.TryResolveForLanguage("km", out var none).Should().BeFalse();
        none.Should().BeNull();
    }

    [Theory]
    [InlineData("")]
    [InlineData("   ")]
    public void ABlankLanguage_IsAnArgumentError(string code)
    {
        var registry = OcrRecognitionModelRegistry.Default;

        registry.Invoking(r => r.ResolveForLanguage(code)).Should().Throw<ArgumentException>();
        registry.Invoking(r => r.TryResolveForLanguage(code, out _)).Should().Throw<ArgumentException>();
    }

    [Fact]
    public void SupportedLanguages_ListTheNewlyMappedCodes()
    {
        OcrRecognitionModelRegistry.Default.GetSupportedLanguages()
            .Should().Contain(["ja", "tr", "hu", "nb", "sr-latn"]);
    }
}
