using AwesomeAssertions;
using LMSupply.Transcriber.Core;
using LMSupply.Transcriber.Decoding;
using LMSupply.Transcriber.Models;

namespace LMSupply.Transcriber.Tests;

/// <summary>
/// <see cref="TranscribeOptions.Language"/> is settled once at the entry of a transcription. A code the model does not
/// decode used to be dropped with a trace line — the model then chose a language of its own and the result carried the
/// caller's wrong code — and an empty string behaved differently from null.
/// </summary>
public class TranscribeLanguageTests
{
    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData("   ")]
    public void NoHint_IsNull_LikeNotAskingForALanguage(string? hint) =>
        WhisperTokenizer.ResolveLanguage(hint).Should().BeNull();

    [Theory]
    [InlineData("ko", "ko")]
    [InlineData("KO", "ko")]
    [InlineData(" Korean ", "ko")]
    [InlineData("english", "en")]
    [InlineData("ko-KR", "ko")]
    [InlineData("ko_KR", "ko")]
    [InlineData("zh-TW", "zh")]
    [InlineData("yue", "yue")]
    public void ACodeANameOrACodeWithARegion_BecomesTheCode(string hint, string expected) =>
        WhisperTokenizer.ResolveLanguage(hint).Should().Be(expected);

    [Theory]
    [InlineData("xx")]
    [InlineData("zz")]
    [InlineData("kor")]
    [InlineData("klingon")]
    [InlineData("-KR")]
    public void ACodeWhisperDoesNotDecode_ThrowsAndNamesTheCodes(string hint)
    {
        var act = () => WhisperTokenizer.ResolveLanguage(hint);

        var thrown = act.Should().Throw<ArgumentException>().Which;
        thrown.Message.Should().Contain($"'{hint}'").And.Contain("ko").And.Contain("leave it null");
        thrown.ParamName.Should().Be("language");
    }

    [Fact]
    public void WithoutAnOptionsObject_ThereIsNothingToSettle() =>
        TranscriptionLanguage.Resolve(null, null).Should().BeNull();

    [Fact]
    public void ACodeAlreadyInPlace_ReturnsTheSameInstance()
    {
        var options = new TranscribeOptions { Language = "ko" };

        TranscriptionLanguage.Resolve(options, null).Should().BeSameAs(options);
    }

    [Theory]
    [InlineData("korean", "ko")]
    [InlineData("ko-KR", "ko")]
    [InlineData("", null)]
    [InlineData("  ", null)]
    public void ANameARegionOrABlank_ReplacesTheHintInACopy_AndLeavesTheCallersObjectAlone(string hint, string? expected)
    {
        var options = new TranscribeOptions { Language = hint, Translate = true, InitialPrompt = "p" };

        var resolved = TranscriptionLanguage.Resolve(options, null)!;

        resolved.Should().NotBeSameAs(options);
        resolved.Language.Should().Be(expected);
        resolved.Translate.Should().BeTrue("every other option is carried over");
        resolved.InitialPrompt.Should().Be("p");
        options.Language.Should().Be(hint, "the caller's object is not changed");
    }

    [Fact]
    public void AMultilingualModel_TakesEveryWhisperLanguage()
    {
        var model = TranscriberModelRegistry.Default.Resolve("default");
        model.SupportedLanguages.Should().BeNull("a multilingual export lists no subset");

        TranscriptionLanguage.Resolve(new TranscribeOptions { Language = "sw" }, model)!.Language.Should().Be("sw");
    }

    [Fact]
    public void AModelFineTunedForOneLanguage_RejectsAnotherAndSaysWhatItTranscribes()
    {
        var korean = TranscriberModelRegistry.Default.Resolve("large-ko");

        TranscriptionLanguage.Resolve(new TranscribeOptions { Language = "korean" }, korean)!.Language.Should().Be("ko");

        var act = () => TranscriptionLanguage.Resolve(new TranscribeOptions { Language = "en" }, korean);
        act.Should().Throw<ArgumentException>().Which.Message
            .Should().Contain(korean.Id).And.Contain("'en'").And.Contain("It transcribes: ko");
    }

    [Fact]
    public void AnEnglishOnlyModel_RejectsAForeignHint()
    {
        var english = TranscriberModelRegistry.Default.Resolve("english");

        var act = () => TranscriptionLanguage.Resolve(new TranscribeOptions { Language = "ko" }, english);

        act.Should().Throw<ArgumentException>().Which.Message.Should().Contain("It transcribes: en");
    }

    [Fact]
    public void AModelFamilyWithAShorterList_RejectsWhatItDoesNotTranscribe()
    {
        var parakeet = TranscriberModelRegistry.Default.Resolve("parakeet-tdt");

        TranscriptionLanguage.Resolve(new TranscribeOptions { Language = "de" }, parakeet)!.Language.Should().Be("de");

        var act = () => TranscriptionLanguage.Resolve(new TranscribeOptions { Language = "ko" }, parakeet);
        act.Should().Throw<ArgumentException>().Which.Message.Should().Contain(parakeet.Id).And.Contain("'ko'");
    }
}
