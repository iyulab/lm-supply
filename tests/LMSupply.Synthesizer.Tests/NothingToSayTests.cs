using AwesomeAssertions;
using LMSupply.Synthesizer.Core;

namespace LMSupply.Synthesizer.Tests;

/// <summary>
/// A call that has nothing the voice can say is refused instead of answered with an empty audio file the caller would take for
/// success — text in another language than the voice's is the usual cause, because the phonemizer drops what it cannot read.
/// </summary>
public class NothingToSayTests
{
    [Theory]
    [InlineData("")]
    [InlineData("   ")]
    [InlineData("\n\t")]
    public void NoText_IsSaidSo(string text)
    {
        var refusal = OnnxSynthesizerModel.NothingToSay(text, [], "", "ko-kr", "ko");

        refusal.ParamName.Should().Be("text");
        refusal.Message.Should().Contain("empty or white space");
    }

    [Fact]
    public void TextThePhonemizerLeavesNothingOf_NamesTheVoiceItsLanguageAndTheDroppedSymbols()
    {
        var refusal = OnnxSynthesizerModel.NothingToSay("こんにちは", [], "こ", "ko-kr", "ko");

        refusal.ParamName.Should().Be("text");
        refusal.Message.Should().Contain("'ko-kr'").And.Contain("5 character(s)").And.Contain("U+3053")
            .And.Contain("voice speaks 'ko'");
    }

    [Fact]
    public void TextWithoutAKnownLanguage_LeavesTheLanguageHintOut()
    {
        var refusal = OnnxSynthesizerModel.NothingToSay("!!!", [], "", "voice", language: null);

        refusal.Message.Should().NotContain("speaks").And.Contain("voice");
    }

    [Fact]
    public void Phonemes_ThePhonemeApiNamesTheSentencesParameter()
    {
        var refusal = OnnxSynthesizerModel.NothingToSay(null, ["xyz", "q"], "xyzq", "voice", "en");

        refusal.ParamName.Should().Be("sentences");
        refusal.Message.Should().Contain("2 sentence(s)").And.Contain("U+0078");
    }
}
