namespace LMSupply.Text.Tests;

public class VocabularyLoaderTests : IDisposable
{
    private readonly string _tempDir;

    public VocabularyLoaderTests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), $"vocab-tests-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_tempDir);
    }

    public void Dispose()
    {
        if (Directory.Exists(_tempDir))
            Directory.Delete(_tempDir, true);
        GC.SuppressFinalize(this);
    }

    #region LoadFromVocabTxtAsync

    [Fact]
    public async Task LoadFromVocabTxtAsync_BertFormat_ReturnsCorrectMapping()
    {
        var path = CreateFile("vocab.txt", "[PAD]\n[UNK]\n[CLS]\n[SEP]\nhello\nworld");

        var vocab = await VocabularyLoader.LoadFromVocabTxtAsync(path, cancellationToken: TestContext.Current.CancellationToken);

        vocab.Should().HaveCount(6);
        vocab["[PAD]"].Should().Be(0);
        vocab["[UNK]"].Should().Be(1);
        vocab["[CLS]"].Should().Be(2);
        vocab["[SEP]"].Should().Be(3);
        vocab["hello"].Should().Be(4);
        vocab["world"].Should().Be(5);
    }

    [Fact]
    public async Task LoadFromVocabTxtAsync_EmptyLines_SkipsEmpty()
    {
        var path = CreateFile("vocab.txt", "hello\n\n\nworld\n");

        var vocab = await VocabularyLoader.LoadFromVocabTxtAsync(path, cancellationToken: TestContext.Current.CancellationToken);

        vocab.Should().HaveCount(2);
        vocab["hello"].Should().Be(0);
        // Line index 3 because empty lines are at indices 1, 2
        vocab["world"].Should().Be(3);
    }

    [Fact]
    public async Task LoadFromVocabTxtAsync_DuplicateTokens_KeepsFirst()
    {
        var path = CreateFile("vocab.txt", "hello\nworld\nhello");

        var vocab = await VocabularyLoader.LoadFromVocabTxtAsync(path, cancellationToken: TestContext.Current.CancellationToken);

        vocab.Should().HaveCount(2);
        vocab["hello"].Should().Be(0); // First occurrence kept
    }

    [Fact]
    public async Task LoadFromVocabTxtAsync_TrailingWhitespace_Trimmed()
    {
        var path = CreateFile("vocab.txt", "hello   \nworld\t");

        var vocab = await VocabularyLoader.LoadFromVocabTxtAsync(path, cancellationToken: TestContext.Current.CancellationToken);

        vocab.Should().ContainKey("hello");
        vocab.Should().ContainKey("world");
    }

    [Fact]
    public async Task LoadFromVocabTxtAsync_EmptyFile_ReturnsEmpty()
    {
        var path = CreateFile("vocab.txt", "");

        var vocab = await VocabularyLoader.LoadFromVocabTxtAsync(path, cancellationToken: TestContext.Current.CancellationToken);

        vocab.Should().BeEmpty();
    }

    [Fact]
    public async Task LoadFromVocabTxtAsync_FileNotFound_ThrowsFileNotFoundException()
    {
        var path = Path.Combine(_tempDir, "nonexistent.txt");

        var act = () => VocabularyLoader.LoadFromVocabTxtAsync(path);

        await act.Should().ThrowAsync<FileNotFoundException>();
    }

    [Fact]
    public async Task LoadFromVocabTxtAsync_CaseSensitive_DistinguishesCase()
    {
        var path = CreateFile("vocab.txt", "Hello\nhello\nHELLO");

        var vocab = await VocabularyLoader.LoadFromVocabTxtAsync(path, cancellationToken: TestContext.Current.CancellationToken);

        vocab.Should().HaveCount(3);
        vocab["Hello"].Should().Be(0);
        vocab["hello"].Should().Be(1);
        vocab["HELLO"].Should().Be(2);
    }

    #endregion

    #region LoadFromVocabJsonAsync

    [Fact]
    public async Task LoadFromVocabJsonAsync_ValidJson_ReturnsCorrectMapping()
    {
        var path = CreateFile("vocab.json",
            """{"[PAD]": 0, "[UNK]": 1, "hello": 100, "world": 101}""");

        var vocab = await VocabularyLoader.LoadFromVocabJsonAsync(path, cancellationToken: TestContext.Current.CancellationToken);

        vocab.Should().HaveCount(4);
        vocab["[PAD]"].Should().Be(0);
        vocab["[UNK]"].Should().Be(1);
        vocab["hello"].Should().Be(100);
        vocab["world"].Should().Be(101);
    }

    [Fact]
    public async Task LoadFromVocabJsonAsync_NonIntegerValues_Skipped()
    {
        var path = CreateFile("vocab.json",
            """{"hello": 1, "invalid": "not_a_number", "world": 2}""");

        var vocab = await VocabularyLoader.LoadFromVocabJsonAsync(path, cancellationToken: TestContext.Current.CancellationToken);

        vocab.Should().HaveCount(2);
        vocab["hello"].Should().Be(1);
        vocab["world"].Should().Be(2);
    }

    [Fact]
    public async Task LoadFromVocabJsonAsync_FileNotFound_ThrowsFileNotFoundException()
    {
        var path = Path.Combine(_tempDir, "nonexistent.json");

        var act = () => VocabularyLoader.LoadFromVocabJsonAsync(path);

        await act.Should().ThrowAsync<FileNotFoundException>();
    }

    [Fact]
    public async Task LoadFromVocabJsonAsync_MalformedJson_FallsBackToManualParsing()
    {
        // Malformed JSON that System.Text.Json cannot parse but manual parser can handle
        // Adding trailing comma which is invalid JSON
        var path = CreateFile("vocab.json",
            """{"hello": 1, "world": 2,}""");

        // Manual parser should handle this (it splits by commas)
        // But System.Text.Json will throw, triggering fallback
        var vocab = await VocabularyLoader.LoadFromVocabJsonAsync(path, cancellationToken: TestContext.Current.CancellationToken);

        // Manual parser should extract these
        vocab.Should().ContainKey("hello");
        vocab["hello"].Should().Be(1);
    }

    #endregion

    #region LoadFromTokenizerJsonAsync

    [Fact]
    public async Task LoadFromTokenizerJsonAsync_ObjectVocab_ReturnsCorrectMapping()
    {
        var json = """
        {
            "model": {
                "vocab": {
                    "[PAD]": 0,
                    "[UNK]": 1,
                    "hello": 100
                }
            }
        }
        """;
        var path = CreateFile("tokenizer.json", json);

        var vocab = await VocabularyLoader.LoadFromTokenizerJsonAsync(path, cancellationToken: TestContext.Current.CancellationToken);

        vocab.Should().HaveCount(3);
        vocab["[PAD]"].Should().Be(0);
        vocab["[UNK]"].Should().Be(1);
        vocab["hello"].Should().Be(100);
    }

    [Fact]
    public async Task LoadFromTokenizerJsonAsync_ArrayObjectFormat_ReturnsCorrectMapping()
    {
        var json = """
        {
            "model": {
                "vocab": [
                    {"id": 0, "content": "[PAD]"},
                    {"id": 1, "content": "[UNK]"},
                    {"id": 100, "content": "hello"}
                ]
            }
        }
        """;
        var path = CreateFile("tokenizer.json", json);

        var vocab = await VocabularyLoader.LoadFromTokenizerJsonAsync(path, cancellationToken: TestContext.Current.CancellationToken);

        vocab.Should().HaveCount(3);
        vocab["[PAD]"].Should().Be(0);
        vocab["[UNK]"].Should().Be(1);
        vocab["hello"].Should().Be(100);
    }

    [Fact]
    public async Task LoadFromTokenizerJsonAsync_UnigramFormat_UsesIndexAsId()
    {
        var json = """
        {
            "model": {
                "vocab": [
                    ["<unk>", 0.0],
                    ["hello", -1.5],
                    ["world", -2.3]
                ]
            }
        }
        """;
        var path = CreateFile("tokenizer.json", json);

        var vocab = await VocabularyLoader.LoadFromTokenizerJsonAsync(path, cancellationToken: TestContext.Current.CancellationToken);

        vocab.Should().HaveCount(3);
        vocab["<unk>"].Should().Be(0);
        vocab["hello"].Should().Be(1);
        vocab["world"].Should().Be(2);
    }

    [Fact]
    public async Task LoadFromTokenizerJsonAsync_NoModelProperty_ReturnsEmpty()
    {
        var json = """{"version": "1.0"}""";
        var path = CreateFile("tokenizer.json", json);

        var vocab = await VocabularyLoader.LoadFromTokenizerJsonAsync(path, cancellationToken: TestContext.Current.CancellationToken);

        vocab.Should().BeEmpty();
    }

    [Fact]
    public async Task LoadFromTokenizerJsonAsync_NoVocabProperty_ReturnsEmpty()
    {
        var json = """{"model": {"type": "BPE"}}""";
        var path = CreateFile("tokenizer.json", json);

        var vocab = await VocabularyLoader.LoadFromTokenizerJsonAsync(path, cancellationToken: TestContext.Current.CancellationToken);

        vocab.Should().BeEmpty();
    }

    [Fact]
    public async Task LoadFromTokenizerJsonAsync_FileNotFound_ThrowsFileNotFoundException()
    {
        var path = Path.Combine(_tempDir, "nonexistent.json");

        var act = () => VocabularyLoader.LoadFromTokenizerJsonAsync(path);

        await act.Should().ThrowAsync<FileNotFoundException>();
    }

    [Fact]
    public async Task LoadFromTokenizerJsonAsync_InvalidJson_ReturnsEmpty()
    {
        var path = CreateFile("tokenizer.json", "not valid json at all");

        var vocab = await VocabularyLoader.LoadFromTokenizerJsonAsync(path, cancellationToken: TestContext.Current.CancellationToken);

        vocab.Should().BeEmpty();
    }

    [Fact]
    public async Task LoadFromTokenizerJsonAsync_ArrayObjectNullContent_Skipped()
    {
        var json = """
        {
            "model": {
                "vocab": [
                    {"id": 0, "content": null},
                    {"id": 1, "content": "valid"}
                ]
            }
        }
        """;
        var path = CreateFile("tokenizer.json", json);

        var vocab = await VocabularyLoader.LoadFromTokenizerJsonAsync(path, cancellationToken: TestContext.Current.CancellationToken);

        vocab.Should().HaveCount(1);
        vocab["valid"].Should().Be(1);
    }

    #endregion

    #region LoadFromModelDirectoryAsync

    [Fact]
    public async Task LoadFromModelDirectoryAsync_VocabTxtExists_LoadsTxt()
    {
        CreateFile("vocab.txt", "[PAD]\nhello");

        var vocab = await VocabularyLoader.LoadFromModelDirectoryAsync(_tempDir, cancellationToken: TestContext.Current.CancellationToken);

        vocab.Should().HaveCount(2);
        vocab["[PAD]"].Should().Be(0);
    }

    [Fact]
    public async Task LoadFromModelDirectoryAsync_OnlyVocabJson_LoadsJson()
    {
        CreateFile("vocab.json", """{"hello": 0, "world": 1}""");

        var vocab = await VocabularyLoader.LoadFromModelDirectoryAsync(_tempDir, cancellationToken: TestContext.Current.CancellationToken);

        vocab.Should().HaveCount(2);
        vocab["hello"].Should().Be(0);
    }

    [Fact]
    public async Task LoadFromModelDirectoryAsync_OnlyTokenizerJson_LoadsTokenizer()
    {
        var json = """{"model": {"vocab": {"token": 0}}}""";
        CreateFile("tokenizer.json", json);

        var vocab = await VocabularyLoader.LoadFromModelDirectoryAsync(_tempDir, cancellationToken: TestContext.Current.CancellationToken);

        vocab.Should().HaveCount(1);
        vocab["token"].Should().Be(0);
    }

    [Fact]
    public async Task LoadFromModelDirectoryAsync_VocabTxtPreferred_OverJson()
    {
        // Both vocab.txt and vocab.json exist — vocab.txt should be loaded
        CreateFile("vocab.txt", "from_txt");
        CreateFile("vocab.json", """{"from_json": 0}""");

        var vocab = await VocabularyLoader.LoadFromModelDirectoryAsync(_tempDir, cancellationToken: TestContext.Current.CancellationToken);

        vocab.Should().ContainKey("from_txt");
        vocab.Should().NotContainKey("from_json");
    }

    [Fact]
    public async Task LoadFromModelDirectoryAsync_NoVocabFiles_ThrowsFileNotFoundException()
    {
        // Empty temp dir — no vocab files

        var act = () => VocabularyLoader.LoadFromModelDirectoryAsync(_tempDir);

        await act.Should().ThrowAsync<FileNotFoundException>()
            .WithMessage("*No vocabulary file found*");
    }

    #endregion

    #region ExtractSpecialTokensFromJson

    [Fact]
    public void ExtractSpecialTokensFromJson_BertTokens_ExtractsCorrectly()
    {
        var json = """
        {
            "added_tokens": [
                {"content": "[CLS]", "id": 101},
                {"content": "[SEP]", "id": 102},
                {"content": "[PAD]", "id": 0},
                {"content": "[UNK]", "id": 100},
                {"content": "[MASK]", "id": 103}
            ]
        }
        """;
        var path = CreateFile("tokenizer.json", json);

        var tokens = VocabularyLoader.ExtractSpecialTokensFromJson(path);

        tokens.ClsTokenId.Should().Be(101);
        tokens.SepTokenId.Should().Be(102);
        tokens.PadTokenId.Should().Be(0);
        tokens.UnkTokenId.Should().Be(100);
        tokens.MaskTokenId.Should().Be(103);
    }

    [Fact]
    public void ExtractSpecialTokensFromJson_SentencePieceTokens_ExtractsCorrectly()
    {
        var json = """
        {
            "added_tokens": [
                {"content": "<s>", "id": 1},
                {"content": "</s>", "id": 2},
                {"content": "<pad>", "id": 0},
                {"content": "<unk>", "id": 3}
            ]
        }
        """;
        var path = CreateFile("tokenizer.json", json);

        var tokens = VocabularyLoader.ExtractSpecialTokensFromJson(path);

        tokens.BosTokenId.Should().Be(1);
        tokens.EosTokenId.Should().Be(2);
        tokens.PadTokenId.Should().Be(0);
        tokens.UnkTokenId.Should().Be(3);
    }

    [Fact]
    public void ExtractSpecialTokensFromJson_MixedBertAndSentencePiece_PadUsesBracketFirst()
    {
        // Both [PAD] and <pad> present — [PAD] should win for padId
        var json = """
        {
            "added_tokens": [
                {"content": "[PAD]", "id": 0},
                {"content": "<pad>", "id": 50}
            ]
        }
        """;
        var path = CreateFile("tokenizer.json", json);

        var tokens = VocabularyLoader.ExtractSpecialTokensFromJson(path);

        // [PAD] sets padId=0, <pad> uses ??= so it does NOT overwrite
        tokens.PadTokenId.Should().Be(0);
    }

    [Fact]
    public void ExtractSpecialTokensFromJson_NoAddedTokens_DefaultsPadUnkToZero()
    {
        var json = """{"model": {"type": "BPE"}}""";
        var path = CreateFile("tokenizer.json", json);

        var tokens = VocabularyLoader.ExtractSpecialTokensFromJson(path);

        tokens.PadTokenId.Should().Be(0);
        tokens.UnkTokenId.Should().Be(0);
        tokens.ClsTokenId.Should().BeNull();
        tokens.SepTokenId.Should().BeNull();
        tokens.BosTokenId.Should().BeNull();
        tokens.EosTokenId.Should().BeNull();
        tokens.MaskTokenId.Should().BeNull();
    }

    [Fact]
    public void ExtractSpecialTokensFromJson_MalformedJson_ReturnsBertDefaults()
    {
        var path = CreateFile("tokenizer.json", "not valid json");

        var tokens = VocabularyLoader.ExtractSpecialTokensFromJson(path);

        // Falls back to SpecialTokens.Bert
        tokens.ClsTokenId.Should().Be(101);
        tokens.SepTokenId.Should().Be(102);
        tokens.PadTokenId.Should().Be(0);
        tokens.UnkTokenId.Should().Be(100);
        tokens.MaskTokenId.Should().Be(103);
    }

    [Fact]
    public void ExtractSpecialTokensFromJson_FileNotFound_ReturnsBertDefaults()
    {
        var path = Path.Combine(_tempDir, "nonexistent.json");

        var tokens = VocabularyLoader.ExtractSpecialTokensFromJson(path);

        // catch block returns SpecialTokens.Bert
        tokens.PadTokenId.Should().Be(0);
        tokens.UnkTokenId.Should().Be(100);
    }

    #endregion

    #region Edge Cases

    [Fact]
    public async Task LoadFromVocabTxtAsync_SingleToken_ReturnsOne()
    {
        var path = CreateFile("vocab.txt", "[PAD]");

        var vocab = await VocabularyLoader.LoadFromVocabTxtAsync(path, cancellationToken: TestContext.Current.CancellationToken);

        vocab.Should().HaveCount(1);
        vocab["[PAD]"].Should().Be(0);
    }

    [Fact]
    public async Task LoadFromVocabJsonAsync_EmptyObject_ReturnsEmpty()
    {
        var path = CreateFile("vocab.json", "{}");

        var vocab = await VocabularyLoader.LoadFromVocabJsonAsync(path, cancellationToken: TestContext.Current.CancellationToken);

        vocab.Should().BeEmpty();
    }

    [Fact]
    public async Task LoadFromVocabJsonAsync_SpecialCharacters_PreservedInKeys()
    {
        var path = CreateFile("vocab.json",
            """{"##ing": 0, "Ġhello": 1, "▁world": 2}""");

        var vocab = await VocabularyLoader.LoadFromVocabJsonAsync(path, cancellationToken: TestContext.Current.CancellationToken);

        vocab.Should().HaveCount(3);
        vocab["##ing"].Should().Be(0);
        vocab["Ġhello"].Should().Be(1);
        vocab["▁world"].Should().Be(2);
    }

    [Fact]
    public async Task LoadFromTokenizerJsonAsync_LargeVocab_HandlesCorrectly()
    {
        // Simulate a moderately-sized vocab
        var entries = new List<string>();
        for (int i = 0; i < 1000; i++)
        {
            entries.Add($"\"token_{i}\": {i}");
        }
        var json = "{\"model\": {\"vocab\": {" + string.Join(", ", entries) + "}}}";
        var path = CreateFile("tokenizer.json", json);

        var vocab = await VocabularyLoader.LoadFromTokenizerJsonAsync(path, cancellationToken: TestContext.Current.CancellationToken);

        vocab.Should().HaveCount(1000);
        vocab["token_0"].Should().Be(0);
        vocab["token_999"].Should().Be(999);
    }

    [Fact]
    public async Task LoadFromVocabJsonAsync_EscapedCharactersInManualParser_HandledCorrectly()
    {
        // JSON with escaped quotes that will cause System.Text.Json to fail (trailing comma)
        // but manual parser should handle the escape sequences
        var path = CreateFile("vocab.json",
            "{\"hello\\\"world\": 1, \"normal\": 2,}");

        var vocab = await VocabularyLoader.LoadFromVocabJsonAsync(path, cancellationToken: TestContext.Current.CancellationToken);

        // Manual parser should handle escaped quotes
        vocab.Should().ContainKey("normal");
        vocab["normal"].Should().Be(2);
    }

    #endregion

    #region Helpers

    private string CreateFile(string name, string content)
    {
        var path = Path.Combine(_tempDir, name);
        File.WriteAllText(path, content);
        return path;
    }

    #endregion
}
