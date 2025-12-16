namespace LMSupply.Text.Tests;

public class SpecialTokensTests
{
    [Fact]
    public void Bert_ShouldHaveCorrectDefaults()
    {
        // Act
        var tokens = SpecialTokens.Bert;

        // Assert
        tokens.PadToken.Should().Be("[PAD]");
        tokens.UnkToken.Should().Be("[UNK]");
        tokens.ClsToken.Should().Be("[CLS]");
        tokens.SepToken.Should().Be("[SEP]");
        tokens.MaskToken.Should().Be("[MASK]");
        tokens.PadTokenId.Should().Be(0);
        tokens.UnkTokenId.Should().Be(100);
        tokens.ClsTokenId.Should().Be(101);
        tokens.SepTokenId.Should().Be(102);
        tokens.MaskTokenId.Should().Be(103);
    }

    [Fact]
    public void Gpt2_ShouldHaveCorrectDefaults()
    {
        // Act
        var tokens = SpecialTokens.Gpt2;

        // Assert
        tokens.PadToken.Should().Be("<|endoftext|>");
        tokens.BosToken.Should().Be("<|endoftext|>");
        tokens.EosToken.Should().Be("<|endoftext|>");
        tokens.PadTokenId.Should().Be(50256);
        tokens.BosTokenId.Should().Be(50256);
        tokens.EosTokenId.Should().Be(50256);
    }

    [Fact]
    public void SentencePiece_ShouldHaveCorrectDefaults()
    {
        // Act
        var tokens = SpecialTokens.SentencePiece;

        // Assert
        tokens.PadToken.Should().Be("<pad>");
        tokens.UnkToken.Should().Be("<unk>");
        tokens.BosToken.Should().Be("<s>");
        tokens.EosToken.Should().Be("</s>");
        tokens.PadTokenId.Should().Be(0);
        tokens.UnkTokenId.Should().Be(3);
        tokens.BosTokenId.Should().Be(1);
        tokens.EosTokenId.Should().Be(2);
    }

    [Fact]
    public void FromVocabulary_ShouldExtractBertTokens()
    {
        // Arrange
        var vocab = new Dictionary<string, int>
        {
            ["[PAD]"] = 0,
            ["[UNK]"] = 100,
            ["[CLS]"] = 101,
            ["[SEP]"] = 102,
            ["[MASK]"] = 103,
            ["hello"] = 7592
        };

        // Act
        var tokens = SpecialTokens.FromVocabulary(vocab);

        // Assert
        tokens.PadToken.Should().Be("[PAD]");
        tokens.UnkToken.Should().Be("[UNK]");
        tokens.ClsToken.Should().Be("[CLS]");
        tokens.SepToken.Should().Be("[SEP]");
        tokens.MaskToken.Should().Be("[MASK]");
        tokens.PadTokenId.Should().Be(0);
        tokens.UnkTokenId.Should().Be(100);
        tokens.ClsTokenId.Should().Be(101);
        tokens.SepTokenId.Should().Be(102);
        tokens.MaskTokenId.Should().Be(103);
    }

    [Fact]
    public void FromVocabulary_ShouldExtractSentencePieceTokens()
    {
        // Arrange
        var vocab = new Dictionary<string, int>
        {
            ["<pad>"] = 0,
            ["<s>"] = 1,
            ["</s>"] = 2,
            ["<unk>"] = 3,
            ["hello"] = 100
        };

        // Act
        var tokens = SpecialTokens.FromVocabulary(vocab);

        // Assert
        tokens.PadToken.Should().Be("<pad>");
        tokens.BosToken.Should().Be("<s>");
        tokens.EosToken.Should().Be("</s>");
        tokens.UnkToken.Should().Be("<unk>");
        tokens.PadTokenId.Should().Be(0);
        tokens.BosTokenId.Should().Be(1);
        tokens.EosTokenId.Should().Be(2);
        tokens.UnkTokenId.Should().Be(3);
    }

    [Fact]
    public void FromVocabulary_WithMissingTokens_ShouldReturnDefaults()
    {
        // Arrange
        var vocab = new Dictionary<string, int>
        {
            ["hello"] = 0,
            ["world"] = 1
        };

        // Act
        var tokens = SpecialTokens.FromVocabulary(vocab);

        // Assert
        tokens.PadTokenId.Should().Be(0);
        tokens.UnkTokenId.Should().Be(0);
        tokens.ClsTokenId.Should().BeNull();
        tokens.SepTokenId.Should().BeNull();
    }
}
