using AwesomeAssertions;
using LMSupply.Transcriber.Decoding;

namespace LMSupply.Transcriber.Tests;

/// <summary>
/// Tests for Whisper language detection: token mapping, probability calculation, and SOT sequence.
/// </summary>
public class WhisperLanguageDetectionTests
{
    private readonly WhisperTokenizer _tokenizer = WhisperTokenizer.CreateDefault();

    [Fact]
    public void GetLanguageToken_KnownLanguages_ReturnValidTokenInRange()
    {
        string[] testLanguages = ["en", "zh", "de", "es", "ko", "ja", "fr"];

        foreach (var lang in testLanguages)
        {
            var token = _tokenizer.GetLanguageToken(lang);
            token.Should().NotBeNull($"'{lang}' should have a valid token");
            token!.Value.Should().BeInRange(
                _tokenizer.LanguageTokenStart,
                _tokenizer.LanguageTokenEnd,
                $"token for '{lang}' should be in language token range");
        }
    }

    [Fact]
    public void GetLanguageToken_FirstLanguageIsEnglish()
    {
        _tokenizer.GetLanguageToken("en").Should().Be(_tokenizer.LanguageTokenStart);
    }

    [Fact]
    public void GetLanguageFromToken_FirstToken_ReturnsEnglish()
    {
        _tokenizer.GetLanguageFromToken(_tokenizer.LanguageTokenStart)
            .Should().Be("en");
    }

    [Fact]
    public void GetLanguageFromToken_KnownTokens_ReturnCorrectLanguage()
    {
        var languages = WhisperTokenizer.SupportedLanguages.Keys.ToList();

        for (int i = 0; i < Math.Min(10, languages.Count); i++)
        {
            var token = _tokenizer.LanguageTokenStart + i;
            var language = _tokenizer.GetLanguageFromToken(token);
            language.Should().Be(languages[i]);
        }
    }

    [Fact]
    public void LanguageToken_RoundTrip_AllLanguagesInTokenRange()
    {
        foreach (var (code, _) in WhisperTokenizer.SupportedLanguages)
        {
            var token = _tokenizer.GetLanguageToken(code);
            token.Should().NotBeNull($"language '{code}' should have a valid token");

            if (!_tokenizer.IsLanguageToken(token!.Value))
                continue;

            var roundTripped = _tokenizer.GetLanguageFromToken(token.Value);
            roundTripped.Should().Be(code, $"round-trip for '{code}' should match");
        }
    }

    [Theory]
    [InlineData(50259, true)]   // en
    [InlineData(50357, true)]   // last language token (v2 default)
    [InlineData(50258, false)]  // StartOfTranscript
    [InlineData(50358, false)]  // past range (v2 default)
    [InlineData(0, false)]
    public void IsLanguageToken_CorrectlyIdentifiesRange(int tokenId, bool expected)
    {
        _tokenizer.IsLanguageToken(tokenId).Should().Be(expected);
    }

    [Fact]
    public void GetLanguageFromToken_NonLanguageToken_ReturnsNull()
    {
        _tokenizer.GetLanguageFromToken(0).Should().BeNull();
        _tokenizer.GetLanguageFromToken(50258).Should().BeNull();
        _tokenizer.GetLanguageFromToken(50358).Should().BeNull();
    }

    [Fact]
    public void GetLanguageToken_InvalidCode_ReturnsNull()
    {
        _tokenizer.GetLanguageToken("xx").Should().BeNull();
        _tokenizer.GetLanguageToken("invalid").Should().BeNull();
    }

    [Fact]
    public void GetSotSequence_NoLanguage_OmitsLanguageToken()
    {
        var sot = _tokenizer.GetSotSequence(language: null, timestamps: false);

        sot.Should().Contain(_tokenizer.StartOfTranscriptToken);
        sot.Should().Contain(_tokenizer.TranscribeToken);
        sot.Should().Contain(_tokenizer.NoTimestampsToken);
        sot.Should().NotContain(t => _tokenizer.IsLanguageToken(t));
    }

    [Fact]
    public void GetSotSequence_WithLanguage_IncludesLanguageToken()
    {
        var sot = _tokenizer.GetSotSequence(language: "en", timestamps: false);

        sot.Should().Contain(_tokenizer.StartOfTranscriptToken);
        sot.Should().Contain(50259); // English language token
        sot.Should().Contain(_tokenizer.TranscribeToken);
    }

    [Fact]
    public void ComputeLanguageTokenProbability_DominantToken_ReturnsHighProbability()
    {
        var decoder = CreateTestDecoder();
        var logits = new float[51000];
        var enToken = _tokenizer.LanguageTokenStart;

        for (int i = _tokenizer.LanguageTokenStart; i <= _tokenizer.LanguageTokenEnd; i++)
        {
            logits[i] = -10f;
        }

        logits[enToken] = 10f;

        var prob = decoder.ComputeLanguageTokenProbability(logits, enToken);

        prob.Should().BeGreaterThan(0.99f, "dominant token should have near-1.0 probability");
    }

    [Fact]
    public void ComputeLanguageTokenProbability_UniformLogits_ReturnsEqualProbability()
    {
        var decoder = CreateTestDecoder();
        var logits = new float[51000];

        for (int i = _tokenizer.LanguageTokenStart; i <= _tokenizer.LanguageTokenEnd; i++)
        {
            logits[i] = 5.0f;
        }

        var languageCount = _tokenizer.LanguageTokenEnd - _tokenizer.LanguageTokenStart + 1;
        var expectedProb = 1.0f / languageCount;

        var prob = decoder.ComputeLanguageTokenProbability(logits, _tokenizer.LanguageTokenStart);

        prob.Should().BeApproximately(expectedProb, 0.001f,
            "uniform logits should yield equal probability for each token");
    }

    [Fact]
    public void ComputeLanguageTokenProbability_OutOfRange_ReturnsZero()
    {
        var decoder = CreateTestDecoder();
        var logits = new float[51000];

        decoder.ComputeLanguageTokenProbability(logits, 0).Should().Be(0f);
        decoder.ComputeLanguageTokenProbability(logits, 50358).Should().Be(0f);
    }

    [Fact]
    public void ComputeLanguageTokenProbability_ShortLogitsArray_ReturnsZero()
    {
        var decoder = CreateTestDecoder();
        var logits = new float[100];

        decoder.ComputeLanguageTokenProbability(logits, _tokenizer.LanguageTokenStart)
            .Should().Be(0f);
    }

    [Fact]
    public void ComputeLanguageTokenProbability_TwoCompetingLanguages_SplitsProbability()
    {
        var decoder = CreateTestDecoder();
        var logits = new float[51000];

        for (int i = _tokenizer.LanguageTokenStart; i <= _tokenizer.LanguageTokenEnd; i++)
        {
            logits[i] = -100f;
        }

        var enToken = _tokenizer.LanguageTokenStart;
        var zhToken = _tokenizer.LanguageTokenStart + 1;
        logits[enToken] = 10f;
        logits[zhToken] = 10f;

        var enProb = decoder.ComputeLanguageTokenProbability(logits, enToken);
        var zhProb = decoder.ComputeLanguageTokenProbability(logits, zhToken);

        enProb.Should().BeApproximately(0.5f, 0.01f);
        zhProb.Should().BeApproximately(0.5f, 0.01f);
    }

    /// <summary>
    /// Creates a WhisperDecoder with a null session for testing probability computation only.
    /// </summary>
    private WhisperDecoder CreateTestDecoder()
    {
        return WhisperDecoder.CreateForTesting(_tokenizer);
    }
}
