using System.Text;
using AwesomeAssertions;
using LMSupply.ImageGenerator.Tokenizers;

namespace LMSupply.ImageGenerator.Tests;

public sealed class ClipTokenizerTests : IDisposable
{
    private readonly string _directory = Path.Combine(Path.GetTempPath(), "clip-tokenizer-" + Guid.NewGuid().ToString("N"));

    public ClipTokenizerTests()
    {
        Directory.CreateDirectory(_directory);
        // Ids: a 0, b 1, a</w> 2, b</w> 3, <|startoftext|> 4, <|endoftext|> 5.
        File.WriteAllText(
            Path.Combine(_directory, "vocab.json"),
            "{\"a\":0,\"b\":1,\"a</w>\":2,\"b</w>\":3,\"<|startoftext|>\":4,\"<|endoftext|>\":5}",
            Encoding.UTF8);
        File.WriteAllText(Path.Combine(_directory, "merges.txt"), "#version: 0.2\na b</w>\n", Encoding.UTF8);
    }

    public void Dispose() => Directory.Delete(_directory, recursive: true);

    [Fact]
    public void Encode_PutsTheStartIdFirstTheEndIdLastAndPadsToTheMaximumLength()
    {
        using var tokenizer = ClipTokenizer.FromDirectory(_directory, maxLength: 8);

        // "ab" merges to the symbol that has no id of its own here, so use two words.
        tokenizer.Encode("a a").Should().Equal(4, 2, 2, 5, 5, 5, 5, 5);
    }

    [Fact]
    public void Encode_TruncatesALongPromptButKeepsTheEndId()
    {
        using var tokenizer = ClipTokenizer.FromDirectory(_directory, maxLength: 5);

        tokenizer.Encode("a a a a a a a").Should().Equal(4, 2, 2, 2, 5);
    }

    [Fact]
    public void Encode_WithoutSpecialTokensIsTheBareIds()
    {
        using var tokenizer = ClipTokenizer.FromDirectory(_directory);

        tokenizer.Encode("A a", addSpecialTokens: false).Should().Equal(2, 2);
    }

    [Fact]
    public void Decode_SkipsTheSpecialTokensByDefault()
    {
        using var tokenizer = ClipTokenizer.FromDirectory(_directory, maxLength: 6);

        tokenizer.Decode(tokenizer.Encode("a b")).Should().Be("a b");
    }

    [Fact]
    public void FromDirectory_WithoutTheFilesSaysWhatIsMissing()
    {
        var empty = Path.Combine(_directory, "empty");
        Directory.CreateDirectory(empty);

        var act = () => ClipTokenizer.FromDirectory(empty);

        act.Should().Throw<FileNotFoundException>().WithMessage("*vocab.json and merges.txt*");
    }
}
