using AwesomeAssertions;
using LMSupply.Transcriber.Decoding;

namespace LMSupply.Transcriber.Tests;

/// <summary>
/// The pure half of Whisper's language-identification step (<c>WhisperDecoder.SelectLanguage</c>):
/// given the logits of one <c>[SOT]</c>-only decoder step, pick the most likely language token and
/// its probability over the language range. No session — a synthetic vocab-sized vector.
/// </summary>
public class WhisperLanguageSelectionTests
{
    private readonly WhisperTokenizer _tokenizer = WhisperTokenizer.CreateDefault();

    [Fact]
    public void SelectLanguage_PicksArgmaxOverLanguageRange_WithSoftmaxProbability()
    {
        var decoder = WhisperDecoder.CreateForTesting(_tokenizer);
        var logits = new float[_tokenizer.LanguageTokenEnd + 100];
        Array.Fill(logits, -10f);
        logits[_tokenizer.GetLanguageToken("en")!.Value] = 2f;
        logits[_tokenizer.GetLanguageToken("ko")!.Value] = 6f;
        logits[_tokenizer.EndOfTextToken] = 50f; // outside the language range — must be ignored

        var detection = decoder.SelectLanguage(logits);

        detection.Should().NotBeNull();
        detection!.Language.Should().Be("ko");
        detection.Probability.Should().BeGreaterThan(0.9f, "ko leads en by 4 nats over an otherwise flat range");
        detection.Probability.Should().BeLessThanOrEqualTo(1f);
    }

    [Fact]
    public void SelectLanguage_LogitsWithoutLanguageRange_ReturnsNull()
    {
        var decoder = WhisperDecoder.CreateForTesting(_tokenizer);
        var logits = new float[_tokenizer.LanguageTokenStart - 1]; // an export whose vocab stops before the range

        decoder.SelectLanguage(logits).Should().BeNull();
    }
}
