using AwesomeAssertions;
using LMSupply.Translator.Tokenization;

namespace LMSupply.Translator.Tests.Tokenization;

/// <summary>
/// The wrapper owns the special tokens. The inner SentencePiece tokenizer used to add its own &lt;s&gt; to every input,
/// which a Marian vocabulary (no &lt;s&gt;) mapped to &lt;unk&gt;: opus-mt-ko-en encoded «오늘 날씨가 정말 좋습니다.» as
/// [1, 1114, …] where the Marian reference (source.spm pieces through vocab.json, &lt;/s&gt; appended) is [1114, …].
/// The fixture is a tiny unigram model with a Marian-style vocab.json (&lt;/s&gt; = 0, &lt;unk&gt; = 1, no &lt;s&gt;);
/// the expected ids come from the Python sentencepiece reference on the same files.
/// </summary>
public class TranslatorTokenizerSpecialTokenTests
{
    private static string FixtureDir => Path.Combine(AppContext.BaseDirectory, "Tokenization", "Fixtures", "TinyMarian");

    [Fact]
    public void EncodeSource_MatchesTheMarianReference_NoLeadingToken()
    {
        using var tokenizer = TranslatorTokenizer.Create(FixtureDir);

        var ids = tokenizer.EncodeSource("the meeting starts tomorrow");

        ids.Should().Equal(10, 7, 3, 21, 4, 15, 23, 18, 12, 20, 22, 2, 30, 0);
    }

    [Fact]
    public void EncodeSource_WithoutSpecialTokens_IsThePiecesAlone()
    {
        using var tokenizer = TranslatorTokenizer.Create(FixtureDir);

        tokenizer.EncodeSource("the meeting starts tomorrow", addSpecialTokens: false)
            .Should().Equal(10, 7, 3, 21, 4, 15, 23, 18, 12, 20, 22, 2, 30);
        tokenizer.EncodeSource("the meeting starts tomorrow", addSpecialTokens: false)
            .Should().NotContain(tokenizer.UnkTokenId, "every piece of this sentence is in the vocabulary");
    }
}
