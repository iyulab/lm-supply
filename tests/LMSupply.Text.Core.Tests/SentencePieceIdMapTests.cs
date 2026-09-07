using System.Buffers;
using Microsoft.ML.Tokenizers;

namespace LMSupply.Text.Tests;

/// <summary>
/// XLM-Roberta 계열(multilingual-e5-*, bge-m3)은 SentencePiece 모델의 raw id를 그대로 쓰지 않는다 —
/// fairseq가 <c>0..3</c>을 <c>&lt;s&gt; &lt;pad&gt; &lt;/s&gt; &lt;unk&gt;</c>로 예약했기 때문에
/// 내용 토큰이 한 칸씩 밀린다. 그 사상을 spm 어휘와 HF 어휘의 «실측 차이»에서 도출한다.
/// </summary>
public class SentencePieceIdMapTests
{
    // spm 자신의 배치: <unk>=0, <s>=1, </s>=2, 이후 내용 토큰.
    private static Dictionary<string, int> SpmVocab() => new(StringComparer.Ordinal)
    {
        ["<unk>"] = 0,
        ["<s>"] = 1,
        ["</s>"] = 2,
        ["▁refund"] = 40,
        ["▁policy"] = 1293,
        [":"] = 11,
    };

    // XLM-R 배치: <s>=0, <pad>=1, </s>=2, <unk>=3, 내용 토큰은 spm id + 1.
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
    /// spm의 <c>&lt;unk&gt;</c>(0)에 오프셋을 그냥 더하면 XLM-R의 <c>&lt;pad&gt;</c>(1)가 된다 —
    /// 알 수 없는 조각이 조용히 패딩으로 둔갑한다. 특수 토큰은 이름으로 다시 찾는다.
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
    /// 제보된 실측 벡터(`query: refund policy`)를 그대로 고정한다 —
    /// HF `tokenizers` 참조 구현이 낸 id와 같아야 한다.
    /// </summary>
    [Fact]
    public void Map_ReproducesTheReferenceTokenizerIdsForAKnownSequence()
    {
        var map = XlmRobertaMap();
        int[] rawSpmIds = [40, 1293, 11];

        map.Map(rawSpmIds).Should().Equal(41, 1294, 12);
    }

    /// <summary>
    /// 오프셋이 없는 모델(spm 어휘와 HF 어휘가 같은 배치)에서는 아무것도 하지 않아야 한다 —
    /// 이 수정이 XLM-R이 아닌 기존 SentencePiece 소비자를 건드리면 안 된다.
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
    /// 조각마다 차이가 제각각이면 «한 칸 밀림»이라는 전제 자체가 틀린 것이다 — 추측해서 밀지 않는다.
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
/// 래퍼가 특수 토큰을 소유한다는 불변식을 고정한다 — 하위 SentencePiece 토크나이저가 자기 BOS를
/// 덧붙이면 XLM-R 어휘에서 그 id(1)는 <c>&lt;s&gt;</c>가 아니라 <c>&lt;pad&gt;</c>다.
/// </summary>
public class SentencePieceWrapperSpecialTokenTests
{
    // XLM-R 배치의 특수 토큰: <s>=0, <pad>=1, </s>=2, <unk>=3.
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

    /// <summary>내용 토큰만 돌려주는 토크나이저 — 실제 spm이 BOS 방출을 끈 상태와 같다.</summary>
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
