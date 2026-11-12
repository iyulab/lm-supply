using System.Text;
using AwesomeAssertions;
using LMSupply.ImageGenerator.Tokenizers;

namespace LMSupply.ImageGenerator.Tests;

public sealed class ClipBpeTests
{
    // A vocabulary small enough to read: the symbols below with an end-of-word variant, a few merged symbols, the two markers.
    // Ids: a 0, b 1, c 2, t 3, ! 4, 1 5, 2 6, Ã 7, © 8 — then the end-of-word variants and merged symbols named in the table.
    private static readonly string[] Symbols =
    [
        "a", "b", "c", "t", "!", "1", "2", "Ã", "©",
        "a</w>", "b</w>", "c</w>", "t</w>", "!</w>", "1</w>", "2</w>", "Ã</w>", "©</w>",
        "at</w>", "ab</w>", "aab</w>", "aa", "<|startoftext|>", "<|endoftext|>",
    ];

    // Best first: "a t</w>" is tried before "a a", so "aat" merges to a, at</w> and then stays; "aab" takes "a b</w>"
    // first and then "a ab</w>".
    private static readonly string[] Merges =
    [
        "#version: 0.2",
        "a t</w>",
        "a b</w>",
        "a ab</w>",
        "a a",
    ];

    private const int Unknown = 999;

    private static int Id(string symbol) => Array.IndexOf(Symbols, symbol);

    private static ClipBpe Create()
    {
        var vocab = new StringBuilder("{");
        for (var i = 0; i < Symbols.Length; i++)
        {
            vocab.Append(i == 0 ? "" : ",").Append('"').Append(Symbols[i].Replace("\"", "\\\"")).Append("\":").Append(i);
        }

        vocab.Append('}');
        using var vocabStream = new MemoryStream(Encoding.UTF8.GetBytes(vocab.ToString()));
        using var mergesStream = new MemoryStream(Encoding.UTF8.GetBytes(string.Join('\n', Merges) + "\n"));
        return ClipBpe.Create(vocabStream, mergesStream, Unknown);
    }

    [Fact]
    public void Encode_GivesTheLastSymbolOfAWordItsEndOfWordVariant()
    {
        Create().Encode("cat").Should().Equal(Id("c"), Id("at</w>"));
    }

    [Fact]
    public void Encode_MergesTheBestRankedPairFirst()
    {
        // "aab": a, a, b</w> -> a, ab</w> (rank 1 beats "a a", rank 3) -> aab</w> (rank 2).
        Create().Encode("aab").Should().Equal(Id("aab</w>"));
        // "aaa": a, a, a</w> -> only the pair "a a" is ranked (the last symbol carries the suffix, so "a a</w>" is not a pair
        // in the list), so the first two merge and the word ends as aa, a</w>.
        Create().Encode("aaa").Should().Equal(Id("aa"), Id("a</w>"));
    }

    [Fact]
    public void Encode_SplitsWordsFromPunctuationAndDigitsOneByOne()
    {
        Create().Encode("cat!!12").Should().Equal(
            Id("c"), Id("at</w>"), Id("!"), Id("!</w>"), Id("1</w>"), Id("2</w>"));
    }

    [Fact]
    public void Encode_LowerCasesAndCollapsesWhitespace()
    {
        var bpe = Create();
        bpe.Encode("  CAT \t\n cat  ").Should().Equal(bpe.Encode("cat cat"));
        bpe.Encode("cat cat").Should().Equal(Id("c"), Id("at</w>"), Id("c"), Id("at</w>"));
    }

    [Fact]
    public void Encode_ReadsTheMarkersInTheTextAsTheirOwnIds()
    {
        Create().Encode("a<|startoftext|>b<|endoftext|>").Should().Equal(
            Id("a</w>"), Id("<|startoftext|>"), Id("b</w>"), Id("<|endoftext|>"));
    }

    [Fact]
    public void Encode_MapsBytesOfNonAsciiTextToTheirPrintableCharacters()
    {
        // "é" is the UTF-8 bytes C3 A9, which stand for the characters Ã and ©.
        Create().Encode("é").Should().Equal(Id("Ã"), Id("©</w>"));
    }

    [Fact]
    public void Encode_UsesTheUnknownIdForASymbolTheVocabularyLacks()
    {
        Create().Encode("z").Should().Equal(Unknown);
    }

    [Fact]
    public void Encode_OfNothingIsEmpty()
    {
        Create().Encode("").Should().BeEmpty();
        Create().Encode("   ").Should().BeEmpty();
    }

    [Fact]
    public void Decode_TurnsEndOfWordSuffixesIntoSpacesAndBytesBackIntoText()
    {
        var bpe = Create();
        bpe.Decode(bpe.Encode("cat ab é")).Should().Be("cat ab é");
    }

    [Fact]
    public void Decode_SkipsIdsTheVocabularyDoesNotHold()
    {
        var bpe = Create();
        bpe.Decode([Id("c"), Unknown, Id("at</w>")]).Should().Be("cat");
    }

    [Theory]
    [InlineData("ＡＢＣ １２３", "abc 123")]
    [InlineData("a ﬁne ﬂower", "a fine flower")]
    [InlineData("“quoted” ‘it’s’", "\"quoted\" 'it's'")]
    [InlineData("&amp;lt;b&amp;gt; &amp;amp;", "<b> &")]
    [InlineData("café", "café")]
    [InlineData("A　B\r\nC", "a b c")]
    public void Clean_AppliesTheNormalizationsTheReferenceMakes(string text, string expected)
    {
        ClipBpe.Clean(text).Should().Be(expected);
    }
}
