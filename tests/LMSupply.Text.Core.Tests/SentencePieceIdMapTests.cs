using System.Buffers;
using Microsoft.ML.Tokenizers;

namespace LMSupply.Text.Tests;

/// <summary>
/// XLM-RoBERTa models (multilingual-e5-*, bge-m3) do not use the SentencePiece model's raw ids as-is —
/// fairseq reserves <c>0..3</c> for <c>&lt;s&gt; &lt;pad&gt; &lt;/s&gt; &lt;unk&gt;</c>, so content
/// tokens shift by one. The mapping is derived from the observed difference between the spm and HF vocabularies.
/// </summary>
public class SentencePieceIdMapTests
{
    // spm's own layout: <unk>=0, <s>=1, </s>=2, then content tokens.
    private static Dictionary<string, int> SpmVocab() => new(StringComparer.Ordinal)
    {
        ["<unk>"] = 0,
        ["<s>"] = 1,
        ["</s>"] = 2,
        ["▁refund"] = 40,
        ["▁policy"] = 1293,
        [":"] = 11,
    };

    // XLM-R layout: <s>=0, <pad>=1, </s>=2, <unk>=3; content tokens are spm id + 1.
    private static Dictionary<string, int> XlmRobertaVocab() => new(StringComparer.Ordinal)
    {
        ["<s>"] = 0,
        ["<pad>"] = 1,
        ["</s>"] = 2,
        ["<unk>"] = 3,
        ["▁refund"] = 41,
        ["▁policy"] = 1294,
        [":"] = 12,
    };

    private static SentencePieceIdMap XlmRobertaMap() => SentencePieceIdMap.Create(
        SpmVocab(), XlmRobertaVocab(), unknownToken: "<unk>", unknownId: 0,
        beginningOfSentenceToken: "<s>", beginningOfSentenceId: 1,
        endOfSentenceToken: "</s>", endOfSentenceId: 2);

    [Fact]
    public void Create_DerivesTheFairseqOffsetFromTheTwoVocabularies()
    {
        var map = XlmRobertaMap();

        map.IsIdentity.Should().BeFalse();
        map.Map(40).Should().Be(41);
        map.Map(1293).Should().Be(1294);
        map.Map(11).Should().Be(12);
    }

    /// <summary>
    /// Naively adding the offset to spm's <c>&lt;unk&gt;</c> (0) yields XLM-R's <c>&lt;pad&gt;</c> (1) —
    /// unknown pieces would silently turn into padding. Special tokens are looked up again by name.
    /// </summary>
    [Fact]
    public void Map_ResolvesSpecialTokensByNameInsteadOfShiftingThem()
    {
        var map = XlmRobertaMap();

        map.Map(0).Should().Be(3, "spm <unk> must land on the XLM-R <unk>, not on <pad>");
        map.Map(1).Should().Be(0, "spm <s> must land on the XLM-R <s>");
        map.Map(2).Should().Be(2, "spm </s> and XLM-R </s> already agree");
    }

    /// <summary>
    /// Pins the ids for `query: refund policy` —
    /// they must match the ids produced by the HF `tokenizers` reference implementation.
    /// </summary>
    [Fact]
    public void Map_ReproducesTheReferenceTokenizerIdsForAKnownSequence()
    {
        var map = XlmRobertaMap();
        int[] rawSpmIds = [40, 1293, 11];

        map.Map(rawSpmIds).Should().Equal(41, 1294, 12);
    }

    /// <summary>
    /// Models without an offset (spm and HF vocabularies share the same layout) must be left unchanged —
    /// the mapping must not affect non-XLM-R SentencePiece models.
    /// </summary>
    [Fact]
    public void Create_ReturnsIdentityWhenTheVocabulariesAlreadyAgree()
    {
        var spm = SpmVocab();

        var map = SentencePieceIdMap.Create(
            spm, spm, unknownToken: "<unk>", unknownId: 0,
            beginningOfSentenceToken: "<s>", beginningOfSentenceId: 1,
            endOfSentenceToken: "</s>", endOfSentenceId: 2);

        map.IsIdentity.Should().BeTrue();
        map.Map(40).Should().Be(40);
    }

    [Fact]
    public void Create_ReturnsIdentityWhenNoHuggingFaceVocabularyIsAvailable()
    {
        var map = SentencePieceIdMap.Create(
            SpmVocab(), new Dictionary<string, int>(), unknownToken: "<unk>", unknownId: 0,
            beginningOfSentenceToken: "<s>", beginningOfSentenceId: 1,
            endOfSentenceToken: "</s>", endOfSentenceId: 2);

        map.IsIdentity.Should().BeTrue();
    }

    /// <summary>
    /// If the difference varies from piece to piece, the shift-by-one premise does not hold — no shift is guessed.
    /// </summary>
    [Fact]
    public void Create_ReturnsIdentityWhenTheDifferenceIsNotConstant()
    {
        var inconsistent = new Dictionary<string, int>(StringComparer.Ordinal)
        {
            ["▁refund"] = 41,
            ["▁policy"] = 1300,
            [":"] = 12,
        };

        var map = SentencePieceIdMap.Create(
            SpmVocab(), inconsistent, unknownToken: "<unk>", unknownId: 0,
            beginningOfSentenceToken: "<s>", beginningOfSentenceId: 1,
            endOfSentenceToken: "</s>", endOfSentenceId: 2);

        map.IsIdentity.Should().BeTrue();
    }
}

/// <summary>
/// Pins the invariant that the wrapper owns the special tokens — if the inner SentencePiece tokenizer
/// prepended its own BOS, that id (1) would be <c>&lt;pad&gt;</c> rather than <c>&lt;s&gt;</c> in the XLM-R vocabulary.
/// </summary>
public class SentencePieceWrapperSpecialTokenTests
{
    // Special tokens in the XLM-R layout: <s>=0, <pad>=1, </s>=2, <unk>=3.
    private static SpecialTokens XlmRobertaSpecials() => new()
    {
        BosToken = "<s>",
        BosTokenId = 0,
        EosToken = "</s>",
        EosTokenId = 2,
        PadToken = "<pad>",
        PadTokenId = 1,
        UnkToken = "<unk>",
        UnkTokenId = 3,
    };

    [Fact]
    public void PairTokenizer_EmitsExactlyOneLeadingSpecialToken()
    {
        var tokenizer = new StubTokenizer([41, 1294]);
        var sut = new SentencePiecePairTokenizer(tokenizer, XlmRobertaSpecials(), maxSequenceLength: 16);

        var ids = sut.Encode("refund policy");

        ids.Should().Equal([0, 41, 1294, 2],
            "the wrapper adds <s> and </s>; nothing underneath may add its own");
    }

    [Fact]
    public void TextTokenizer_EmitsExactlyOneLeadingSpecialToken()
    {
        var tokenizer = new StubTokenizer([41, 1294]);
        var sut = new SentencePieceTextTokenizer(tokenizer, XlmRobertaSpecials());

        var ids = sut.Encode("refund policy");

        ids.Should().Equal(0, 41, 1294, 2);
    }

    /// <summary>Tokenizer that returns content tokens only — equivalent to a real spm with BOS emission disabled.</summary>
    private sealed class StubTokenizer(int[] contentIds) : Tokenizer
    {
        protected override EncodeResults<int> EncodeToIds(
            string? text, ReadOnlySpan<char> textSpan, EncodeSettings settings)
            => new() { Tokens = contentIds, NormalizedText = text, CharsConsumed = text?.Length ?? 0 };

        protected override EncodeResults<EncodedToken> EncodeToTokens(
            string? text, ReadOnlySpan<char> textSpan, EncodeSettings settings)
            => new() { Tokens = [], NormalizedText = text, CharsConsumed = text?.Length ?? 0 };

        public override OperationStatus Decode(
            IEnumerable<int> ids, Span<char> destination, out int idsConsumed, out int charsWritten)
        {
            idsConsumed = 0;
            charsWritten = 0;
            return OperationStatus.Done;
        }
    }
}
