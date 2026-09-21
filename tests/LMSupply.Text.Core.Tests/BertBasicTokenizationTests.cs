namespace LMSupply.Text.Tests;

/// <summary>
/// A BERT-family model is trained on text that went through basic tokenization before WordPiece:
/// lowercased and accent-stripped (uncased models), with punctuation and CJK ideographs separated into
/// words of their own. WordPiece alone only looks whitespace-separated words up in the vocabulary, so
/// without that step an uncased vocabulary turns "The", "dog?!" and "Café" into <c>[UNK]</c> and the
/// resulting embeddings and relevance scores drift away from the model's own.
/// </summary>
/// <remarks>
/// The expected token lists are not hand-written: they are what the reference implementation (the
/// HuggingFace <c>tokenizers</c> library — WordPiece model, BertNormalizer, BertPreTokenizer) produces
/// over this same vocabulary, for the uncased convention and for the cased one.
/// </remarks>
public class BertBasicTokenizationTests : IDisposable
{
    private static readonly string[] Vocab =
    [
        "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
        "the", "quick", "brown", "fox", "—", "didn", "'", "t", "jump", "over", "lazy", "dog", "?", "!",
        "cafe", "cr", "##eme", "in", "sao", "paulo", ",", "naive", "resume",
        "東", "京", "タ", "##ワ", "##ー", "is", "beautiful",
        "line", "one", "two", "tab", "##bed", "$", "5", "+", "3", "=", "8",
        "hello", "world", "paris", "capital", "of", "france", ".",
        "The", "Paris", "Café", "QUICK",
    ];

    private static readonly string LongWord = new('x', 120);

    private readonly string _modelDir = Path.Combine(Path.GetTempPath(), $"bert-basic-{Guid.NewGuid():N}");

    public BertBasicTokenizationTests() => Directory.CreateDirectory(_modelDir);

    public void Dispose()
    {
        if (Directory.Exists(_modelDir))
            Directory.Delete(_modelDir, true);
        GC.SuppressFinalize(this);
    }

    public static TheoryData<string, string[]> Uncased => new()
    {
        { "hello world", ["hello", "world"] },
        { "The QUICK brown fox — didn't jump over the lazy dog?!",
            ["the", "quick", "brown", "fox", "—", "didn", "'", "t", "jump", "over", "the", "lazy", "dog", "?", "!"] },
        { "Café crème in São Paulo, naïve résumé",
            ["cafe", "cr", "##eme", "in", "sao", "paulo", ",", "naive", "resume"] },
        { "東京タワー \U0001F680 is beautiful",
            ["東", "京", "タ", "##ワ", "##ー", "[UNK]", "is", "beautiful"] },
        { "line one\nline two\tTabbed $5 + 3 = 8",
            ["line", "one", "line", "two", "tab", "##bed", "$", "5", "+", "3", "=", "8"] },
        { "Paris is the capital of France.", ["paris", "is", "the", "capital", "of", "france", "."] },
        { "hel​lo world\u0000", ["hello", "world"] },
        { LongWord + " fox", ["[UNK]", "fox"] },
        { "", [] },
        { "   ", [] },
    };

    public static TheoryData<string, string[]> Cased => new()
    {
        { "The QUICK brown fox — didn't jump over the lazy dog?!",
            ["The", "QUICK", "brown", "fox", "—", "didn", "'", "t", "jump", "over", "the", "lazy", "dog", "?", "!"] },
        { "Café crème in São Paulo, naïve résumé",
            ["Café", "[UNK]", "in", "[UNK]", "[UNK]", ",", "[UNK]", "[UNK]"] },
        { "line one\nline two\tTabbed $5 + 3 = 8",
            ["line", "one", "line", "two", "[UNK]", "$", "5", "+", "3", "=", "8"] },
        { "Paris is the capital of France.", ["Paris", "is", "the", "capital", "of", "[UNK]", "."] },
    };

    [Theory]
    [MemberData(nameof(Uncased))]
    public async Task AnUncasedModel_TokenizesLikeTheReference(string text, string[] expected)
    {
        WriteVocab(Vocab.Where(t => t.StartsWith('[') || !t.Any(char.IsUpper)));
        WriteTokenizerConfig(doLowerCase: true);

        (await TokensOf(text)).Should().Equal(expected);
    }

    [Theory]
    [MemberData(nameof(Cased))]
    public async Task ACasedModel_KeepsCaseAndAccents(string text, string[] expected)
    {
        WriteVocab(Vocab);
        WriteTokenizerConfig(doLowerCase: false, stripAccents: false);

        (await TokensOf(text)).Should().Equal(expected);
    }

    [Fact]
    public async Task TokenizerJson_IsTheAuthority_OverTokenizerConfig()
    {
        WriteVocab(Vocab);
        WriteTokenizerConfig(doLowerCase: true);
        File.WriteAllText(Path.Combine(_modelDir, "tokenizer.json"), """
            {
              "normalizer": { "type": "BertNormalizer", "clean_text": true, "handle_chinese_chars": true, "strip_accents": false, "lowercase": false },
              "pre_tokenizer": { "type": "BertPreTokenizer" },
              "model": { "type": "WordPiece" }
            }
            """);

        (await TokensOf("Paris is the capital.")).Should().Equal("Paris", "is", "the", "capital", ".");
    }

    [Fact]
    public async Task AModelThatDeclaresNothing_IsJudgedByItsVocabulary()
    {
        // No tokenizer.json, no tokenizer_config.json. An uncased vocabulary holds no uppercase pieces…
        WriteVocab(Vocab.Where(t => t.StartsWith('[') || !t.Any(char.IsUpper)));
        (await TokensOf("Paris is the capital.")).Should().Equal("paris", "is", "the", "capital", ".");

        // …so one that does is cased, and lowercasing it would send "Paris" to a piece it does not have.
        WriteVocab(Vocab);
        (await TokensOf("Paris is the capital.")).Should().Equal("Paris", "is", "the", "capital", ".");
    }

    [Fact]
    public async Task ThePairTokenizer_AppliesTheSameStep_ToBothTexts()
    {
        WriteVocab(Vocab.Where(t => t.StartsWith('[') || !t.Any(char.IsUpper)));
        WriteTokenizerConfig(doLowerCase: true);

        var pair = await TokenizerFactory.CreateWordPiecePairAsync(_modelDir, maxSequenceLength: 32);
        var encoded = pair.EncodePair("The capital of France?", "Paris is the capital of France.");

        var tokens = encoded.InputIds.ToArray().Select(id => Vocab.Where(t => t.StartsWith('[') || !t.Any(char.IsUpper)).ElementAt((int)id)).ToArray();
        tokens.Should().Equal(
            "[CLS]", "the", "capital", "of", "france", "?", "[SEP]",
            "paris", "is", "the", "capital", "of", "france", ".", "[SEP]");
        tokens.Should().NotContain("[UNK]");
    }

    private async Task<string[]> TokensOf(string text)
    {
        var vocab = File.ReadAllLines(Path.Combine(_modelDir, "vocab.txt"));
        var tokenizer = await TokenizerFactory.CreateWordPieceAsync(_modelDir, maxSequenceLength: 256);
        return tokenizer.Encode(text, addSpecialTokens: false).Select(id => vocab[id]).ToArray();
    }

    private void WriteVocab(IEnumerable<string> tokens)
        => File.WriteAllText(Path.Combine(_modelDir, "vocab.txt"), string.Join('\n', tokens) + "\n");

    private void WriteTokenizerConfig(bool doLowerCase, bool? stripAccents = null)
        => File.WriteAllText(
            Path.Combine(_modelDir, "tokenizer_config.json"),
            $$"""{ "do_lower_case": {{(doLowerCase ? "true" : "false")}}, "tokenize_chinese_chars": true, "strip_accents": {{(stripAccents is null ? "null" : stripAccents.Value ? "true" : "false")}} }""");
}
