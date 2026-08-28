using AwesomeAssertions;
using LMSupply.Transcriber.Decoding;

namespace LMSupply.Transcriber.Tests;

/// <summary>
/// Tests for WhisperTokenizer helper methods not covered by WhisperLanguageDetectionTests.
/// Uses default (v2) token IDs.
/// </summary>
public class WhisperTokenizerStaticTests
{
    private readonly WhisperTokenizer _tokenizer = WhisperTokenizer.CreateDefault();

    // IsSpecialToken tests

    [Theory]
    [InlineData(0, false)]
    [InlineData(100, false)]
    [InlineData(50256, false)]    // Just below EndOfText
    [InlineData(50257, true)]     // EndOfText - first special token
    [InlineData(50258, true)]     // StartOfTranscript
    [InlineData(50362, true)]     // NoSpeech
    [InlineData(50363, true)]     // NoTimestamps
    [InlineData(50364, true)]     // TimestampBegin
    [InlineData(51000, true)]     // Arbitrary high timestamp
    public void IsSpecialToken_ShouldClassifyCorrectly(int tokenId, bool expected)
    {
        _tokenizer.IsSpecialToken(tokenId).Should().Be(expected);
    }

    // IsTimestampToken tests

    [Theory]
    [InlineData(0, false)]
    [InlineData(50257, false)]    // EndOfText - special but not timestamp
    [InlineData(50363, false)]    // NoTimestamps - special but not timestamp
    [InlineData(50364, true)]     // TimestampBegin - first timestamp
    [InlineData(50365, true)]     // Second timestamp (0.02s)
    [InlineData(51000, true)]     // Arbitrary timestamp
    [InlineData(51864, true)]     // 30.00s timestamp (1500 * 0.02)
    public void IsTimestampToken_ShouldClassifyCorrectly(int tokenId, bool expected)
    {
        _tokenizer.IsTimestampToken(tokenId).Should().Be(expected);
    }

    // TimestampTokenToSeconds tests

    [Theory]
    [InlineData(50364, 0.0f)]      // First timestamp = 0.00s
    [InlineData(50365, 0.02f)]     // Second = 0.02s
    [InlineData(50414, 1.0f)]      // 50 tokens * 0.02s = 1.0s
    [InlineData(51864, 30.0f)]     // 1500 * 0.02s = 30.0s
    public void TimestampTokenToSeconds_ValidTimestamps_ShouldConvert(int tokenId, float expectedSeconds)
    {
        _tokenizer.TimestampTokenToSeconds(tokenId)
            .Should().BeApproximately(expectedSeconds, 0.001f);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(50257)]    // EndOfText
    [InlineData(50363)]    // NoTimestamps
    public void TimestampTokenToSeconds_BelowBegin_ShouldReturnZero(int tokenId)
    {
        _tokenizer.TimestampTokenToSeconds(tokenId).Should().Be(0f);
    }

    // Default token ID tests

    [Fact]
    public void DefaultTokenIds_ShouldHaveExpectedV2Values()
    {
        _tokenizer.EndOfTextToken.Should().Be(50257);
        _tokenizer.StartOfTranscriptToken.Should().Be(50258);
        _tokenizer.TranslateToken.Should().Be(50358);
        _tokenizer.TranscribeToken.Should().Be(50359);
        _tokenizer.NoSpeechToken.Should().Be(50362);
        _tokenizer.NoTimestampsToken.Should().Be(50363);
        _tokenizer.TimestampBeginToken.Should().Be(50364);
    }

    [Fact]
    public void LanguageTokenRange_ShouldBeConsistent()
    {
        _tokenizer.LanguageTokenStart.Should().Be(50259);
        _tokenizer.LanguageTokenEnd.Should().Be(50357);
        (_tokenizer.LanguageTokenEnd - _tokenizer.LanguageTokenStart + 1)
            .Should().Be(99, "99 language slots in the standard range");
    }

    [Fact]
    public void SupportedLanguages_ShouldNotBeEmpty()
    {
        WhisperTokenizer.SupportedLanguages.Should().NotBeEmpty();
        WhisperTokenizer.SupportedLanguages.Count.Should().BeGreaterThan(90);
    }

    // Boundary: IsSpecialToken and IsTimestampToken relationship

    [Fact]
    public void AllTimestampTokens_ShouldAlsoBeSpecialTokens()
    {
        _tokenizer.IsSpecialToken(_tokenizer.TimestampBeginToken).Should().BeTrue();
    }

    // GetSotSequence tests — verifies translate/transcribe task token selection
    // Regression: TranscribeOptions.Translate used to be silently ignored (see
    // ISSUE-lm-supply-20260409-transcribe-translate-silent-failure.md).

    [Fact]
    public void GetSotSequence_Default_ShouldUseTranscribeToken()
    {
        var sot = _tokenizer.GetSotSequence();

        sot.Should().Contain(_tokenizer.TranscribeToken);
        sot.Should().NotContain(_tokenizer.TranslateToken);
    }

    [Fact]
    public void GetSotSequence_TranslateTrue_ShouldUseTranslateToken()
    {
        var sot = _tokenizer.GetSotSequence(language: "ko", timestamps: false, translate: true);

        sot.Should().Contain(_tokenizer.TranslateToken, "translate=true must wire through to the task token");
        sot.Should().NotContain(_tokenizer.TranscribeToken);
    }

    [Fact]
    public void GetSotSequence_TranslateFalse_ShouldUseTranscribeToken()
    {
        var sot = _tokenizer.GetSotSequence(language: "ko", timestamps: false, translate: false);

        sot.Should().Contain(_tokenizer.TranscribeToken);
        sot.Should().NotContain(_tokenizer.TranslateToken);
    }

    [Fact]
    public void GetSotSequence_TranslateWithTimestamps_ShouldIncludeTranslateAndOmitNoTimestamps()
    {
        var sot = _tokenizer.GetSotSequence(language: "ja", timestamps: true, translate: true);

        sot.Should().Contain(_tokenizer.TranslateToken);
        sot.Should().NotContain(_tokenizer.NoTimestampsToken);
    }
}
