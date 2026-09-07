using System.Diagnostics;

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
            // SentencePieceTokenizer.Create accepts both BPE (LLaMA) and Unigram
            // (XLM-Roberta / multilingual-e5 / BGE-M3). LlamaTokenizer.Create only
            // accepts BPE and throws "model type is not Bpe" on Unigram models.
            // BOS/EOS emission is turned off here because the wrapper owns the special
            // tokens; leaving it on prepends the SentencePiece model's own <s> underneath
            // the wrapper's, which lands on a different token entirely in an XLM-Roberta
            // vocabulary (<pad>).
            tokenizer = SentencePieceTokenizer.Create(
                stream, addBeginningOfSentence: false, addEndOfSentence: false);
        }
        else
        {
            // Fallback to BPE if SentencePiece not found
            tokenizer = CreateBpeTokenizer(modelDir)
                ?? throw new FileNotFoundException(
                    $"No SentencePiece model found. Expected .spm or .model file in: {modelDir}");
        }

        return new SentencePieceTextTokenizer(
            tokenizer, specialTokens, idMap: BuildIdMap(tokenizer, vocab));
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

    /// <summary>
    /// Creates a SentencePiece/Unigram sequence tokenizer (for encoders with non-WordPiece models).
    /// Reuses <see cref="SentencePiecePairTokenizer"/> since <see cref="IPairTokenizer"/> implements
    /// <see cref="ISequenceTokenizer"/> — embedder-only callers ignore the pair methods.
    /// </summary>
    /// <param name="modelDir">Path to model directory.</param>
    /// <param name="maxSequenceLength">Maximum sequence length.</param>
    /// <returns>A sequence tokenizer instance.</returns>
    public static async Task<ISequenceTokenizer> CreateSentencePieceSequenceAsync(
        string modelDir,
        int maxSequenceLength = 512)
    {
        return await CreateSentencePiecePairAsync(modelDir, maxSequenceLength);
    }

    /// <summary>
    /// Auto-detects tokenizer type and creates appropriate sequence tokenizer from model directory.
    /// Supports WordPiece (vocab.txt or tokenizer.json with WordPiece type) and SentencePiece
    /// (Unigram/BPE tokenizer.json, sentencepiece.bpe.model, *.spm).
    /// </summary>
    /// <param name="modelDir">Path to model directory.</param>
    /// <param name="maxSequenceLength">Maximum sequence length.</param>
    /// <returns>A sequence tokenizer instance.</returns>
    public static async Task<ISequenceTokenizer> CreateAutoSequenceAsync(
        string modelDir,
        int maxSequenceLength = 512)
    {
        var vocabTxtPath = Path.Combine(modelDir, "vocab.txt");
        var tokenizerJsonPath = Path.Combine(modelDir, "tokenizer.json");

        // vocab.txt is a definitive WordPiece signal
        if (File.Exists(vocabTxtPath))
        {
            return await CreateWordPieceAsync(modelDir, maxSequenceLength);
        }

        // Inspect tokenizer.json model.type when present
        if (File.Exists(tokenizerJsonPath))
        {
            var tokenizerType = DetectTokenizerType(tokenizerJsonPath);

            return tokenizerType switch
            {
                "WordPiece" => await CreateWordPieceAsync(modelDir, maxSequenceLength),
                "Unigram" or "BPE" => await CreateSentencePieceSequenceAsync(modelDir, maxSequenceLength),
                // Unknown type: prefer SentencePiece if a model file is present, otherwise WordPiece
                _ => FindSentencePieceModel(modelDir) != null
                    ? await CreateSentencePieceSequenceAsync(modelDir, maxSequenceLength)
                    : await CreateWordPieceAsync(modelDir, maxSequenceLength)
            };
        }

        // No tokenizer.json: fall back to SentencePiece file probe
        if (FindSentencePieceModel(modelDir) != null)
        {
            return await CreateSentencePieceSequenceAsync(modelDir, maxSequenceLength);
        }

        throw new FileNotFoundException(
            $"Could not determine tokenizer type from: {modelDir}. " +
            "Expected vocab.txt, tokenizer.json, sentencepiece.bpe.model, or *.spm.");
    }

    /// <summary>
    /// Creates a SentencePiece/Unigram pair tokenizer (for cross-encoders/rerankers with non-WordPiece models).
    /// </summary>
    /// <param name="modelDir">Path to model directory.</param>
    /// <param name="maxSequenceLength">Maximum sequence length.</param>
    /// <returns>A pair tokenizer instance.</returns>
    public static async Task<IPairTokenizer> CreateSentencePiecePairAsync(
        string modelDir,
        int maxSequenceLength = 512)
    {
        var tokenizerJsonPath = Path.Combine(modelDir, "tokenizer.json");
        var spmPath = FindSentencePieceModel(modelDir);

        Tokenizer tokenizer;
        SpecialTokens specialTokens;
        var idMap = SentencePieceIdMap.Identity;

        if (spmPath != null)
        {
            using var stream = File.OpenRead(spmPath);
            // SentencePieceTokenizer.Create accepts both BPE and Unigram model types,
            // so it works for LLaMA-style as well as XLM-Roberta-style SPM files.
            // BOS/EOS emission is turned off here because the wrapper owns the special
            // tokens; leaving it on prepends the SentencePiece model's own <s> underneath
            // the wrapper's, which lands on a different token entirely in an XLM-Roberta
            // vocabulary (<pad>).
            tokenizer = SentencePieceTokenizer.Create(
                stream, addBeginningOfSentence: false, addEndOfSentence: false);
            var vocab = LoadVocabularySync(modelDir);
            specialTokens = SpecialTokens.FromVocabulary(vocab);
            idMap = BuildIdMap(tokenizer, vocab);
        }
        else if (File.Exists(tokenizerJsonPath))
        {
            // For Unigram models without .spm file, try to create from tokenizer.json
            tokenizer = await CreateTokenizerFromJsonAsync(tokenizerJsonPath);
            specialTokens = VocabularyLoader.ExtractSpecialTokensFromJson(tokenizerJsonPath);
        }
        else
        {
            // Fallback to BPE
            var bpeTokenizer = CreateBpeTokenizer(modelDir);
            if (bpeTokenizer == null)
            {
                throw new FileNotFoundException(
                    $"No SentencePiece/BPE model found in: {modelDir}");
            }
            tokenizer = bpeTokenizer;
            var vocab = LoadVocabularySync(modelDir);
            specialTokens = SpecialTokens.FromVocabulary(vocab);
        }

        return new SentencePiecePairTokenizer(
            tokenizer, specialTokens, maxSequenceLength, idMap: idMap);
    }

    /// <summary>
    /// Auto-detects tokenizer type and creates appropriate pair tokenizer from model directory.
    /// Supports WordPiece, Unigram, and BPE tokenizers.
    /// </summary>
    /// <param name="modelDir">Path to model directory.</param>
    /// <param name="maxSequenceLength">Maximum sequence length.</param>
    /// <returns>A pair tokenizer instance.</returns>
    public static async Task<IPairTokenizer> CreateAutoPairAsync(
        string modelDir,
        int maxSequenceLength = 512)
    {
        var tokenizerJsonPath = Path.Combine(modelDir, "tokenizer.json");
        var vocabTxtPath = Path.Combine(modelDir, "vocab.txt");

        // If vocab.txt exists, use WordPiece (BERT-style)
        if (File.Exists(vocabTxtPath))
        {
            return await CreateWordPiecePairAsync(modelDir, maxSequenceLength);
        }

        // Check tokenizer.json for model type
        if (File.Exists(tokenizerJsonPath))
        {
            var tokenizerType = DetectTokenizerType(tokenizerJsonPath);

            return tokenizerType switch
            {
                "WordPiece" => await CreateWordPiecePairAsync(modelDir, maxSequenceLength),
                "Unigram" or "BPE" => await CreateSentencePiecePairAsync(modelDir, maxSequenceLength),
                _ => await CreateSentencePiecePairAsync(modelDir, maxSequenceLength)
            };
        }

        // Check for SentencePiece model
        if (FindSentencePieceModel(modelDir) != null)
        {
            return await CreateSentencePiecePairAsync(modelDir, maxSequenceLength);
        }

        throw new FileNotFoundException(
            $"Could not determine tokenizer type from: {modelDir}. " +
            "Expected vocab.txt, tokenizer.json, or .spm model");
    }

    /// <summary>
    /// Detects the tokenizer type from tokenizer.json.
    /// </summary>
    private static string? DetectTokenizerType(string tokenizerJsonPath)
    {
        try
        {
            var json = File.ReadAllText(tokenizerJsonPath);
            using var doc = JsonDocument.Parse(json);

            if (doc.RootElement.TryGetProperty("model", out var model) &&
                model.TryGetProperty("type", out var typeElement))
            {
                return typeElement.GetString();
            }
        }
        catch (Exception ex)
        {
            Trace.TraceInformation($"[TokenizerFactory] Tokenizer type parsing failed: {ex.Message}");
        }

        return null;
    }

    /// <summary>
    /// Creates a tokenizer from tokenizer.json for non-WordPiece models.
    /// </summary>
    private static async Task<Tokenizer> CreateTokenizerFromJsonAsync(string tokenizerJsonPath)
    {
        var json = await File.ReadAllTextAsync(tokenizerJsonPath);
        using var doc = JsonDocument.Parse(json);

        if (!doc.RootElement.TryGetProperty("model", out var model))
        {
            throw new InvalidOperationException("Invalid tokenizer.json: missing 'model' section");
        }

        if (!model.TryGetProperty("vocab", out var vocab))
        {
            throw new InvalidOperationException("Invalid tokenizer.json: missing 'model.vocab' section");
        }

        // Build vocab dictionary sorted by ID
        var vocabDict = new SortedDictionary<int, string>();

        // Handle both Object and Array formats for vocab
        if (vocab.ValueKind == JsonValueKind.Object)
        {
            foreach (var property in vocab.EnumerateObject())
            {
                vocabDict[property.Value.GetInt32()] = property.Name;
            }
        }
        else if (vocab.ValueKind == JsonValueKind.Array)
        {
            // Handle array formats:
            // 1. [{"id": 0, "content": "[PAD]"}, ...] - Object items
            // 2. [["token", score], ...] - Unigram format (tuple-like arrays)
            var index = 0;
            foreach (var item in vocab.EnumerateArray())
            {
                if (item.ValueKind == JsonValueKind.Object)
                {
                    // Object format: {"id": 0, "content": "[PAD]"}
                    if (item.TryGetProperty("id", out var idProp) &&
                        item.TryGetProperty("content", out var contentProp))
                    {
                        vocabDict[idProp.GetInt32()] = contentProp.GetString() ?? string.Empty;
                    }
                }
                else if (item.ValueKind == JsonValueKind.Array)
                {
                    // Unigram format: ["token", score] - index is the token ID
                    var arr = item.EnumerateArray().ToArray();
                    if (arr.Length >= 1 && arr[0].ValueKind == JsonValueKind.String)
                    {
                        vocabDict[index] = arr[0].GetString() ?? string.Empty;
                    }
                }
                index++;
            }
        }
        else
        {
            throw new InvalidOperationException(
                $"Invalid tokenizer.json: 'model.vocab' has unexpected type '{vocab.ValueKind}'.");
        }

        if (vocabDict.Count == 0)
        {
            throw new InvalidOperationException("Invalid tokenizer.json: 'model.vocab' is empty");
        }

        // Try to detect tokenizer type and handle appropriately
        var tokenizerType = model.TryGetProperty("type", out var typeElement)
            ? typeElement.GetString()
            : null;

        // BPE models with vocab.json + merges.txt can be loaded via CodeGen tokenizer.
        var modelDir = Path.GetDirectoryName(tokenizerJsonPath)!;
        var vocabJsonPath = Path.Combine(modelDir, "vocab.json");
        var mergesPath = Path.Combine(modelDir, "merges.txt");

        if (File.Exists(vocabJsonPath) && File.Exists(mergesPath))
        {
            using var vocabStream = File.OpenRead(vocabJsonPath);
            using var mergesStream = File.OpenRead(mergesPath);
            return CodeGenTokenizer.Create(vocabStream, mergesStream);
        }

        // No viable construction path. Microsoft.ML.Tokenizers (2.0) cannot build a faithful
        // Unigram tokenizer from tokenizer.json alone — Unigram requires the SentencePiece
        // protobuf (`sentencepiece.bpe.model` / `*.spm`). The previous implementation papered
        // over this by constructing a WordPiece tokenizer from the Unigram vocab list, which
        // (a) crashed for XLM-Roberta-style models because their `<unk>` token does not match
        // the WordPiece default `[UNK]`, and (b) would have produced semantically wrong
        // tokenizations even if the validation were silenced (greedy-longest-match vs.
        // probabilistic Unigram).
        var typeLabel = string.IsNullOrEmpty(tokenizerType) ? "non-WordPiece" : tokenizerType;
        throw new InvalidOperationException(
            $"Cannot construct a {typeLabel} tokenizer from tokenizer.json alone. " +
            $"This model requires a SentencePiece protobuf file ('sentencepiece.bpe.model' or '*.spm') " +
            $"to be present in the same directory. Searched: '{modelDir}'. " +
            "If this is a SentencePiece-based model (e.g. XLM-Roberta, multilingual-e5, BGE-M3), " +
            "delete the cached model directory and re-download so the SentencePiece model file is " +
            "fetched alongside tokenizer.json.");
    }

    private static WordPieceTokenizer CreateWordPieceFromJson(string tokenizerJsonPath)
    {
        var json = File.ReadAllText(tokenizerJsonPath);
        using var doc = JsonDocument.Parse(json);

        if (!doc.RootElement.TryGetProperty("model", out var model))
        {
            throw new InvalidOperationException("Invalid tokenizer.json: missing 'model' section");
        }

        // Check model type - WordPiece tokenizer only works with WordPiece models
        if (model.TryGetProperty("type", out var modelType))
        {
            var typeStr = modelType.GetString();
            if (typeStr != null && !typeStr.Equals("WordPiece", StringComparison.OrdinalIgnoreCase))
            {
                throw new InvalidOperationException(
                    $"Tokenizer type mismatch: expected 'WordPiece' but found '{typeStr}'. " +
                    $"This model may require a different tokenizer (e.g., SentencePiece for BPE/Unigram models).");
            }
        }

        if (!model.TryGetProperty("vocab", out var vocab))
        {
            throw new InvalidOperationException("Invalid tokenizer.json: missing 'model.vocab' section");
        }

        // Build vocab dictionary sorted by ID
        var vocabDict = new SortedDictionary<int, string>();

        // Handle both Object and Array formats for vocab
        if (vocab.ValueKind == JsonValueKind.Object)
        {
            foreach (var property in vocab.EnumerateObject())
            {
                vocabDict[property.Value.GetInt32()] = property.Name;
            }
        }
        else if (vocab.ValueKind == JsonValueKind.Array)
        {
            // Handle array formats:
            // 1. [{"id": 0, "content": "[PAD]"}, ...] - Object items
            // 2. [["token", score], ...] - Unigram format (tuple-like arrays)
            var index = 0;
            foreach (var item in vocab.EnumerateArray())
            {
                if (item.ValueKind == JsonValueKind.Object)
                {
                    // Object format: {"id": 0, "content": "[PAD]"}
                    if (item.TryGetProperty("id", out var idProp) &&
                        item.TryGetProperty("content", out var contentProp))
                    {
                        vocabDict[idProp.GetInt32()] = contentProp.GetString() ?? string.Empty;
                    }
                }
                else if (item.ValueKind == JsonValueKind.Array)
                {
                    // Unigram format: ["token", score] - index is the token ID
                    var arr = item.EnumerateArray().ToArray();
                    if (arr.Length >= 1 && arr[0].ValueKind == JsonValueKind.String)
                    {
                        vocabDict[index] = arr[0].GetString() ?? string.Empty;
                    }
                }
                index++;
            }
        }
        else
        {
            throw new InvalidOperationException(
                $"Invalid tokenizer.json: 'model.vocab' has unexpected type '{vocab.ValueKind}'. " +
                "Expected Object (token → id) or Array ([{{id, content}}]).");
        }

        if (vocabDict.Count == 0)
        {
            throw new InvalidOperationException("Invalid tokenizer.json: 'model.vocab' is empty");
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
            // SentencePieceTokenizer.Create accepts both BPE and Unigram models;
            // LlamaTokenizer.Create only accepts BPE and rejects Unigram models like
            // XLM-Roberta / multilingual-e5 with "model type is not Bpe", which would
            // make this validator falsely report SPM files as invalid.
            _ = SentencePieceTokenizer.Create(stream);
            return true;
        }
        catch (Exception ex)
        {
            Trace.TraceInformation($"[TokenizerFactory] SentencePieceTokenizer validation failed: {ex.Message}");
            return false;
        }
    }

    private static CodeGenTokenizer? CreateBpeTokenizer(string modelDir)
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

    /// <summary>
    /// SentencePiece 모델의 raw id를, 그 모델과 함께 배포된 어휘가 실제로 쓰는 id로 옮기는
    /// 사상을 만든다. 두 어휘가 이미 일치하면 아무것도 하지 않는 사상이 나온다.
    /// </summary>
    /// <remarks>
    /// XLM-Roberta 계열(multilingual-e5-*, bge-m3)은 fairseq 예약 슬롯 때문에 내용 토큰이
    /// spm 자신의 id보다 한 칸 뒤에 있다 — 그 어긋남을 계열 목록으로 알아맞히지 않고
    /// 두 어휘를 대조해 도출한다(<see cref="SentencePieceIdMap"/>).
    /// </remarks>
    private static SentencePieceIdMap BuildIdMap(Tokenizer tokenizer, Dictionary<string, int> targetVocabulary)
    {
        if (tokenizer is not SentencePieceTokenizer spm || targetVocabulary.Count == 0)
            return SentencePieceIdMap.Identity;

        var map = SentencePieceIdMap.Create(
            spm.Vocabulary.ToDictionary(entry => entry.Key.ToString(), entry => entry.Value, StringComparer.Ordinal),
            targetVocabulary,
            spm.UnknownToken,
            spm.UnknownId,
            spm.BeginningOfSentenceToken,
            spm.BeginningOfSentenceId,
            spm.EndOfSentenceToken,
            spm.EndOfSentenceId);

        if (!map.IsIdentity)
        {
            Trace.TraceInformation(
                "[TokenizerFactory] SentencePiece ids are remapped onto the model's declared vocabulary.");
        }

        return map;
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
                ParseVocabElement(doc.RootElement, vocab);
            }
            catch (Exception ex)
            {
                Trace.TraceInformation($"[TokenizerFactory] vocab.json parse failed: {ex.Message}");
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
                    ParseVocabElement(vocabElement, vocab);
                    return vocab;
                }
            }
            catch (Exception ex)
            {
                Trace.TraceInformation($"[TokenizerFactory] tokenizer.json vocab parse failed: {ex.Message}");
            }
        }

        return [];
    }

    /// <summary>
    /// Parses vocab element handling both Object and Array formats.
    /// </summary>
    private static void ParseVocabElement(JsonElement element, Dictionary<string, int> vocab)
    {
        if (element.ValueKind == JsonValueKind.Object)
        {
            foreach (var property in element.EnumerateObject())
            {
                if (property.Value.TryGetInt32(out var id))
                {
                    vocab[property.Name] = id;
                }
            }
        }
        else if (element.ValueKind == JsonValueKind.Array)
        {
            // Handle array formats:
            // 1. [{"id": 0, "content": "[PAD]"}, ...] - Object items
            // 2. [["token", score], ...] - Unigram format (tuple-like arrays)
            var index = 0;
            foreach (var item in element.EnumerateArray())
            {
                if (item.ValueKind == JsonValueKind.Object)
                {
                    // Object format: {"id": 0, "content": "[PAD]"}
                    if (item.TryGetProperty("id", out var idProp) &&
                        item.TryGetProperty("content", out var contentProp) &&
                        idProp.TryGetInt32(out var id))
                    {
                        var content = contentProp.GetString();
                        if (content != null)
                        {
                            vocab[content] = id;
                        }
                    }
                }
                else if (item.ValueKind == JsonValueKind.Array)
                {
                    // Unigram format: ["token", score] - index is the token ID
                    var arr = item.EnumerateArray().ToArray();
                    if (arr.Length >= 1 && arr[0].ValueKind == JsonValueKind.String)
                    {
                        var token = arr[0].GetString();
                        if (token != null)
                        {
                            vocab[token] = index;
                        }
                    }
                }
                index++;
            }
        }
    }
}
