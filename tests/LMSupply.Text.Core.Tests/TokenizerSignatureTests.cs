namespace LMSupply.Text.Tests;

/// <summary>
/// <see cref="ISequenceTokenizer.Signature"/> names what a tokenizer does to text on the way to ids —
/// algorithm, normalization convention, implementation epoch — so a consumer that stores what depends
/// on those ids can see a release move them. Same model files + same signature ⇒ same ids.
/// </summary>
public sealed class TokenizerSignatureTests : IDisposable
{
    private readonly string _modelDir = Path.Combine(Path.GetTempPath(), $"tok-sig-{Guid.NewGuid():N}");

    public TokenizerSignatureTests() => Directory.CreateDirectory(_modelDir);

    public void Dispose()
    {
        if (Directory.Exists(_modelDir))
            Directory.Delete(_modelDir, true);
        GC.SuppressFinalize(this);
    }

    [Fact]
    public void Uncased_and_cased_conventions_have_different_signatures()
    {
        var uncased = BertNormalization.Uncased;
        var cased = BertNormalization.Uncased with { Lowercase = false, StripAccents = false };

        uncased.Signature.Should().Be("clean=1;chinese=1;accents=1;lower=1;punct=1");
        cased.Signature.Should().Be("clean=1;chinese=1;accents=0;lower=0;punct=1");
    }

    [Fact]
    public void An_unresolved_accent_flag_is_written_as_the_value_actually_applied()
    {
        // StripAccents = null follows Lowercase — the signature must say what ran, not what was declared.
        (BertNormalization.Uncased with { StripAccents = null, Lowercase = false }).Signature
            .Should().Contain("accents=0");
        (BertNormalization.Uncased with { StripAccents = true, Lowercase = false }).Signature
            .Should().Contain("accents=1");
    }

    [Fact]
    public async Task A_WordPiece_tokenizer_signs_with_its_epoch_and_the_convention_it_loaded()
    {
        await File.WriteAllLinesAsync(Path.Combine(_modelDir, "vocab.txt"),
            ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "hello", "world", "Hello"], TestContext.Current.CancellationToken);
        await File.WriteAllTextAsync(Path.Combine(_modelDir, "tokenizer_config.json"), """{"do_lower_case": false}""", TestContext.Current.CancellationToken);

        var tokenizer = await TokenizerFactory.CreateWordPieceAsync(_modelDir, maxSequenceLength: 32);

        tokenizer.Signature.Should().Be($"wordpiece/{TokenizerEpochs.WordPiece};clean=1;chinese=1;accents=0;lower=0;punct=1");
    }

    [Fact]
    public async Task Two_tokenizers_from_the_same_files_have_the_same_signature()
    {
        await File.WriteAllLinesAsync(Path.Combine(_modelDir, "vocab.txt"),
            ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "hello", "world"], TestContext.Current.CancellationToken);

        var a = await TokenizerFactory.CreateWordPieceAsync(_modelDir, maxSequenceLength: 32);
        var b = await TokenizerFactory.CreateWordPieceAsync(_modelDir, maxSequenceLength: 512);

        a.Signature.Should().Be(b.Signature, "the sequence length is a truncation bound, not part of what the tokenizer does to text");
    }

    [Fact]
    public void The_epochs_are_the_values_the_changelog_records()
    {
        // 0.70.0 (WordPiece basic tokenization) and the XLM-R id map. Raising either is a release act
        // that the golden-vector facts in the integration suite check against the vectors themselves.
        TokenizerEpochs.WordPiece.Should().Be(2);
        TokenizerEpochs.SentencePiece.Should().Be(2);
    }
}
