namespace LMSupply.Text;

/// <summary>
/// Factory for creating tokenizer instances from model directories.
/// </summary>
public static class TokenizerFactory
{
    /// <summary>
    /// Creates a WordPiece tokenizer (BERT-style) from model directory.
    /// </summary>
    /// <param name="modelDir">Path to model directory.</param>
    /// <param name="maxSequenceLength">Maximum sequence length.</param>
    /// <returns>A sequence tokenizer instance.</returns>
    public static async Task<ISequenceTokenizer> CreateWordPieceAsync(
        string modelDir,
        int maxSequenceLength = 512)
    {
        var vocabPath = Path.Combine(modelDir, "vocab.txt");
        var tokenizerJsonPath = Path.Combine(modelDir, "tokenizer.json");

        Tokenizer tokenizer;
        SpecialTokens specialTokens;

        if (File.Exists(vocabPath))
        {
            // Load from vocab.txt
            using var vocabStream = File.OpenRead(vocabPath);
            tokenizer = WordPieceTokenizer.Create(vocabStream);
            var vocab = await VocabularyLoader.LoadFromVocabTxtAsync(vocabPath);
            specialTokens = SpecialTokens.FromVocabulary(vocab);
        }
        else if (File.Exists(tokenizerJsonPath))
        {
            // Extract vocab from tokenizer.json and create WordPiece tokenizer
            tokenizer = CreateWordPieceFromJson(tokenizerJsonPath);
            specialTokens = VocabularyLoader.ExtractSpecialTokensFromJson(tokenizerJsonPath);
        }
        else
        {
            throw new FileNotFoundException(
                $"No vocabulary file found. Expected vocab.txt or tokenizer.json in: {modelDir}");
        }

        return new WordPieceSequenceTokenizer(tokenizer, specialTokens, maxSequenceLength);
    }

    /// <summary>
    /// Creates a WordPiece pair tokenizer (for cross-encoders/rerankers).
    /// </summary>
    /// <param name="modelDir">Path to model directory.</param>
    /// <param name="maxSequenceLength">Maximum sequence length.</param>
    /// <returns>A pair tokenizer instance.</returns>
    public static async Task<IPairTokenizer> CreateWordPiecePairAsync(
        string modelDir,
        int maxSequenceLength = 512)
    {
        var vocabPath = Path.Combine(modelDir, "vocab.txt");
        var tokenizerJsonPath = Path.Combine(modelDir, "tokenizer.json");

        Tokenizer tokenizer;
        SpecialTokens specialTokens;

        if (File.Exists(vocabPath))
        {
            using var vocabStream = File.OpenRead(vocabPath);
            tokenizer = WordPieceTokenizer.Create(vocabStream);
            var vocab = await VocabularyLoader.LoadFromVocabTxtAsync(vocabPath);
            specialTokens = SpecialTokens.FromVocabulary(vocab);
        }
        else if (File.Exists(tokenizerJsonPath))
        {
            tokenizer = CreateWordPieceFromJson(tokenizerJsonPath);
            specialTokens = VocabularyLoader.ExtractSpecialTokensFromJson(tokenizerJsonPath);
        }
        else
        {
            throw new FileNotFoundException(
                $"No vocabulary file found. Expected vocab.txt or tokenizer.json in: {modelDir}");
        }

        return new WordPiecePairTokenizer(tokenizer, specialTokens, maxSequenceLength);
    }

    /// <summary>
    /// Creates a SentencePiece tokenizer (for translation models).
    /// </summary>
    /// <param name="modelDir">Path to model directory.</param>
    /// <returns>A text tokenizer instance.</returns>
    public static ITextTokenizer CreateSentencePiece(string modelDir)
    {
        var spmPath = FindSentencePieceModel(modelDir);
        var vocab = LoadVocabularySync(modelDir);
        var specialTokens = SpecialTokens.FromVocabulary(vocab);

        Tokenizer tokenizer;
        if (spmPath != null)
        {
            using var stream = File.OpenRead(spmPath);
            tokenizer = LlamaTokenizer.Create(stream);
        }
        else
        {
            // Fallback to BPE if SentencePiece not found
            tokenizer = CreateBpeTokenizer(modelDir)
                ?? throw new FileNotFoundException(
                    $"No SentencePiece model found. Expected .spm or .model file in: {modelDir}");
        }

        return new SentencePieceTextTokenizer(tokenizer, specialTokens);
    }

    /// <summary>
    /// Creates a GPT-2 style BPE tokenizer.
    /// </summary>
    /// <param name="modelDir">Path to model directory.</param>
    /// <returns>A text tokenizer instance.</returns>
    public static ITextTokenizer CreateGpt2(string modelDir)
    {
        var vocabPath = Path.Combine(modelDir, "vocab.json");
        var mergesPath = Path.Combine(modelDir, "merges.txt");

        if (!File.Exists(vocabPath) || !File.Exists(mergesPath))
        {
            throw new FileNotFoundException(
                $"GPT-2 tokenizer requires vocab.json and merges.txt in: {modelDir}");
        }

        using var vocabStream = File.OpenRead(vocabPath);
        using var mergesStream = File.OpenRead(mergesPath);
        var tokenizer = CodeGenTokenizer.Create(vocabStream, mergesStream);

        return new Gpt2TextTokenizer(tokenizer);
    }

    /// <summary>
    /// Auto-detects and creates appropriate tokenizer from model directory.
    /// </summary>
    /// <param name="modelDir">Path to model directory.</param>
    /// <param name="maxSequenceLength">Maximum sequence length (for sequence tokenizers).</param>
    /// <returns>A tokenizer instance.</returns>
    public static async Task<ITextTokenizer> CreateAutoAsync(
        string modelDir,
        int maxSequenceLength = 512)
    {
        // Check for SentencePiece model
        if (FindSentencePieceModel(modelDir) != null)
        {
            return CreateSentencePiece(modelDir);
        }

        // Check for GPT-2 style (vocab.json + merges.txt)
        var mergesPath = Path.Combine(modelDir, "merges.txt");
        var vocabJsonPath = Path.Combine(modelDir, "vocab.json");
        if (File.Exists(mergesPath) && File.Exists(vocabJsonPath))
        {
            return CreateGpt2(modelDir);
        }

        // Check for BERT style (vocab.txt or tokenizer.json with WordPiece)
        var vocabTxtPath = Path.Combine(modelDir, "vocab.txt");
        var tokenizerJsonPath = Path.Combine(modelDir, "tokenizer.json");
        if (File.Exists(vocabTxtPath) || File.Exists(tokenizerJsonPath))
        {
            return await CreateWordPieceAsync(modelDir, maxSequenceLength);
        }

        throw new FileNotFoundException(
            $"Could not determine tokenizer type from: {modelDir}. " +
            "Expected vocab.txt, vocab.json + merges.txt, tokenizer.json, or .spm model");
    }

    private static Tokenizer CreateWordPieceFromJson(string tokenizerJsonPath)
    {
        var json = File.ReadAllText(tokenizerJsonPath);
        using var doc = JsonDocument.Parse(json);

        if (!doc.RootElement.TryGetProperty("model", out var model) ||
            !model.TryGetProperty("vocab", out var vocab))
        {
            throw new InvalidOperationException("Invalid tokenizer.json: missing model.vocab section");
        }

        // Build vocab dictionary sorted by ID
        var vocabDict = new SortedDictionary<int, string>();
        foreach (var property in vocab.EnumerateObject())
        {
            vocabDict[property.Value.GetInt32()] = property.Name;
        }

        // Create vocab.txt content
        var vocabLines = new StringBuilder();
        for (var i = 0; i < vocabDict.Count; i++)
        {
            vocabLines.AppendLine(vocabDict.TryGetValue(i, out var token) ? token : $"[unused{i}]");
        }

        var vocabBytes = Encoding.UTF8.GetBytes(vocabLines.ToString());
        using var vocabStream = new MemoryStream(vocabBytes);
        return WordPieceTokenizer.Create(vocabStream);
    }

    private static string? FindSentencePieceModel(string modelDir)
    {
        var patterns = new[]
        {
            "sentencepiece.bpe.model",
            "source.spm",
            "target.spm",
            "*.spm",
            "*.model"
        };

        foreach (var pattern in patterns)
        {
            var files = Directory.GetFiles(modelDir, pattern);
            if (files.Length > 0)
            {
                // Verify it's actually a SentencePiece model
                var file = files[0];
                if (IsSentencePieceModel(file))
                    return file;
            }
        }

        return null;
    }

    private static bool IsSentencePieceModel(string path)
    {
        try
        {
            using var stream = File.OpenRead(path);
            // Try to create a tokenizer - if it works, it's valid
            _ = LlamaTokenizer.Create(stream);
            return true;
        }
        catch
        {
            return false;
        }
    }

    private static Tokenizer? CreateBpeTokenizer(string modelDir)
    {
        var vocabPath = Path.Combine(modelDir, "vocab.json");
        var mergesPath = Path.Combine(modelDir, "merges.txt");

        if (File.Exists(vocabPath) && File.Exists(mergesPath))
        {
            using var vocabStream = File.OpenRead(vocabPath);
            using var mergesStream = File.OpenRead(mergesPath);
            return CodeGenTokenizer.Create(vocabStream, mergesStream);
        }

        return null;
    }

    private static Dictionary<string, int> LoadVocabularySync(string modelDir)
    {
        var vocabJsonPath = Path.Combine(modelDir, "vocab.json");
        if (File.Exists(vocabJsonPath))
        {
            var json = File.ReadAllText(vocabJsonPath);
            var vocab = new Dictionary<string, int>(StringComparer.Ordinal);

            try
            {
                using var doc = JsonDocument.Parse(json);
                foreach (var property in doc.RootElement.EnumerateObject())
                {
                    if (property.Value.TryGetInt32(out var id))
                    {
                        vocab[property.Name] = id;
                    }
                }
            }
            catch
            {
                // Return empty vocab on parse failure
            }

            return vocab;
        }

        var tokenizerJsonPath = Path.Combine(modelDir, "tokenizer.json");
        if (File.Exists(tokenizerJsonPath))
        {
            var json = File.ReadAllText(tokenizerJsonPath);
            try
            {
                using var doc = JsonDocument.Parse(json);
                if (doc.RootElement.TryGetProperty("model", out var model) &&
                    model.TryGetProperty("vocab", out var vocabElement))
                {
                    var vocab = new Dictionary<string, int>(StringComparer.Ordinal);
                    foreach (var property in vocabElement.EnumerateObject())
                    {
                        if (property.Value.TryGetInt32(out var id))
                        {
                            vocab[property.Name] = id;
                        }
                    }
                    return vocab;
                }
            }
            catch
            {
                // Fall through
            }
        }

        return [];
    }
}
