using AwesomeAssertions;
using LMSupply.Transcriber.Decoding;

namespace LMSupply.Transcriber.Tests;

public sealed class SentencePieceVocabularyTests
{
    private static readonly string[] s_lines =
    [
        "<unk> 0",
        "▁Good 1",
        "▁morning 2",
        ", 3",
        "▁every 4",
        "one 5",
        ". 6",
        "▁o 7",
        "' 8",
        "clock 9",
        "<blk> 10"
    ];

    [Fact]
    public void Parse_FindsBlank_AndReplacesWordBoundaryMarker()
    {
        var vocab = SentencePieceVocabulary.Parse(s_lines);

        vocab.Size.Should().Be(11);
        vocab.BlankId.Should().Be(10);
        vocab.Piece(1).Should().Be(" Good");
        vocab.Piece(5).Should().Be("one");
        vocab.Piece(99).Should().BeEmpty();
    }

    [Fact]
    public void Decode_JoinsPieces_KeepsWordSpaces_DropsSpacesBeforePunctuation_AndTheLeadingSpace()
    {
        var vocab = SentencePieceVocabulary.Parse(s_lines);

        vocab.Decode([1, 2, 3, 4, 5, 6]).Should().Be("Good morning, everyone.");
    }

    [Fact]
    public void Decode_ApostropheInsideAWord_StaysAttached()
    {
        var vocab = SentencePieceVocabulary.Parse(s_lines);

        vocab.Decode([7, 8, 9]).Should().Be("o'clock");
    }

    [Fact]
    public void Decode_Empty_IsEmpty()
    {
        SentencePieceVocabulary.Parse(s_lines).Decode([]).Should().BeEmpty();
    }

    [Fact]
    public void Parse_WithoutBlank_Throws()
    {
        var act = () => SentencePieceVocabulary.Parse(["<unk> 0", "▁a 1"]);

        act.Should().Throw<InvalidDataException>().WithMessage("*<blk>*");
    }

    [Fact]
    public void Parse_NonDenseIds_Throws()
    {
        var act = () => SentencePieceVocabulary.Parse(["<unk> 0", "▁a 2", "<blk> 3"]);

        act.Should().Throw<InvalidDataException>().WithMessage("*dense*");
    }

    [Fact]
    public void Parse_MalformedLine_Throws()
    {
        var act = () => SentencePieceVocabulary.Parse(["<unk> 0", "no-id-here", "<blk> 2"]);

        act.Should().Throw<InvalidDataException>().WithMessage("*Malformed*");
    }
}
