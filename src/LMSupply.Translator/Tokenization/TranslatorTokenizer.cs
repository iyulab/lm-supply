using System.Diagnostics;
using System.Globalization;
using System.Text;
using System.Text.Json;
using LMSupply.Text;
using Microsoft.ML.Tokenizers;

namespace LMSupply.Translator.Tokenization;

/// <summary>
/// SentencePiece-based tokenizer for machine translation models.
/// Supports OPUS-MT (MarianMT) models with source and target tokenizers.
/// </summary>
internal sealed class TranslatorTokenizer : IDisposable
{
    private readonly Tokenizer _sourceTokenizer;
    private readonly Tokenizer _targetTokenizer;
    private readonly Dictionary<string, int> _vocab;
    private readonly Dictionary<int, string> _reverseVocab;
    private readonly int _padTokenId;
    private readonly int _bosTokenId;
    private readonly int _eosTokenId;
    private readonly int _unkTokenId;
    private bool _disposed;

    /// <summary>
    /// Gets the vocabulary size.
    /// </summary>
    public int VocabSize => _vocab.Count;

    /// <summary>
    /// Gets the PAD token ID.
    /// </summary>
    public int PadTokenId => _padTokenId;

    /// <summary>
    /// Gets the BOS (beginning of sequence) token ID.
    /// </summary>
    public int BosTokenId => _bosTokenId;

    /// <summary>
    /// Gets the EOS (end of sequence) token ID.
    /// </summary>
    public int EosTokenId => _eosTokenId;

    /// <summary>
    /// Gets the UNK (unknown) token ID.
    /// </summary>
    public int UnkTokenId => _unkTokenId;

    private TranslatorTokenizer(
        Tokenizer sourceTokenizer,
        Tokenizer targetTokenizer,
        Dictionary<string, int> vocab,
        int padTokenId,
        int bosTokenId,
        int eosTokenId,
        int unkTokenId)
    {
        _sourceTokenizer = sourceTokenizer;
        _targetTokenizer = targetTokenizer;
        _vocab = vocab;
        _reverseVocab = vocab.ToDictionary(kv => kv.Value, kv => kv.Key);
        _padTokenId = padTokenId;
        _bosTokenId = bosTokenId;
        _eosTokenId = eosTokenId;
        _unkTokenId = unkTokenId;
    }

    /// <summary>
    /// Creates a tokenizer from model directory.
    /// </summary>
    /// <param name="modelDir">Path to the model directory.</param>
    /// <returns>A configured tokenizer instance.</returns>
    public static TranslatorTokenizer Create(string modelDir)
    {
        // Load vocabulary
        var vocab = LoadVocabulary(modelDir);

        // Determine special token IDs
        var padTokenId = GetTokenId(vocab, "<pad>", "▁<pad>", 0);
        var bosTokenId = GetTokenId(vocab, "<s>", "▁<s>", padTokenId);
        var eosTokenId = GetTokenId(vocab, "</s>", "▁</s>", 0);
        var unkTokenId = GetTokenId(vocab, "<unk>", "▁<unk>", 3);

        // Load SentencePiece tokenizers
        var sourceTokenizer = LoadSentencePieceTokenizer(modelDir, "source.spm", "sentencepiece.bpe.model");
        var targetTokenizer = LoadSentencePieceTokenizer(modelDir, "target.spm", "sentencepiece.bpe.model");

        return new TranslatorTokenizer(
            sourceTokenizer,
            targetTokenizer,
            vocab,
            padTokenId,
            bosTokenId,
            eosTokenId,
            unkTokenId);
    }

    /// <summary>
    /// Encodes source text to token IDs.
    /// </summary>
    /// <param name="text">Input text in source language.</param>
    /// <param name="addSpecialTokens">Whether to add BOS/EOS tokens.</param>
    /// <returns>Array of token IDs.</returns>
    public long[] EncodeSource(string text, bool addSpecialTokens = true)
    {
        var tokens = new List<long>();

        if (addSpecialTokens && _bosTokenId != _padTokenId)
        {
            tokens.Add(_bosTokenId);
        }

        // Get pieces from SentencePiece tokenizer and map to vocab.json IDs
        var encodedTokens = _sourceTokenizer.EncodeToTokens(text, out _);
        foreach (var token in encodedTokens)
        {
            // Map piece string to vocab.json ID
            if (_vocab.TryGetValue(token.Value, out var vocabId))
            {
                tokens.Add(vocabId);
            }
            else
            {
                // Use UNK token for unknown pieces
                tokens.Add(_unkTokenId);
            }
        }

        if (addSpecialTokens)
        {
            tokens.Add(_eosTokenId);
        }

        return tokens.ToArray();
    }

    /// <summary>
    /// Decodes token IDs to target text.
    /// </summary>
    /// <param name="tokenIds">Array of token IDs.</param>
    /// <param name="skipSpecialTokens">Whether to skip special tokens in output.</param>
    /// <returns>Decoded text string.</returns>
    public string DecodeTarget(IEnumerable<long> tokenIds, bool skipSpecialTokens = true)
    {
        var ids = tokenIds.Select(id => (int)id);

        if (skipSpecialTokens)
        {
            ids = ids.Where(id =>
                id != _padTokenId &&
                id != _bosTokenId &&
                id != _eosTokenId);
        }

        // Map vocab.json IDs back to pieces and concatenate
        var pieces = new StringBuilder();
        foreach (var id in ids)
        {
            if (_reverseVocab.TryGetValue(id, out var piece))
            {
                pieces.Append(piece);
            }
        }

        var result = pieces.ToString();

        // Replace SentencePiece whitespace marker (▁) with actual space and normalize
        result = result.Replace("▁", " ");
        // Normalize multiple consecutive spaces to single space and trim
        result = string.Join(" ", result.Split(default(char[]), StringSplitOptions.RemoveEmptyEntries));

        return result;
    }

    /// <summary>
    /// Gets decoder input IDs for autoregressive generation.
    /// </summary>
    /// <returns>Initial decoder input IDs.</returns>
    public long[] GetDecoderStartIds()
    {
        return [_padTokenId]; // MarianMT uses pad_token_id as decoder_start_token_id
    }

    /// <summary>
    /// Checks if a token ID is a special token.
    /// </summary>
    /// <param name="tokenId">Token ID to check.</param>
    /// <returns>True if the token is a special token.</returns>
    public bool IsSpecialToken(long tokenId)
    {
        return tokenId == _padTokenId ||
               tokenId == _bosTokenId ||
               tokenId == _eosTokenId ||
               tokenId == _unkTokenId;
    }

    private static Dictionary<string, int> LoadVocabulary(string modelDir)
    {
        try
        {
            // Use centralized VocabularyLoader from Text.Core
            return VocabularyLoader.LoadFromModelDirectoryAsync(modelDir).GetAwaiter().GetResult();
        }
        catch (FileNotFoundException)
        {
            // Create minimal vocabulary if none found
            return CreateMinimalVocabulary();
        }
    }

    private static Dictionary<string, int> CreateMinimalVocabulary()
    {
        // Create a minimal vocabulary for fallback
        return new Dictionary<string, int>(StringComparer.Ordinal)
        {
            ["<pad>"] = 0,
            ["<s>"] = 1,
            ["</s>"] = 2,
            ["<unk>"] = 3
        };
    }

    private static int GetTokenId(Dictionary<string, int> vocab, string primary, string alternative, int defaultId)
    {
        if (vocab.TryGetValue(primary, out var id))
            return id;
        if (vocab.TryGetValue(alternative, out id))
            return id;
        return defaultId;
    }

    private static Tokenizer LoadSentencePieceTokenizer(string modelDir, string primaryFile, string fallbackFile)
    {
        // Build list of candidate paths to search (modelDir and subdirectories)
        var searchPaths = new List<string> { modelDir };

        // Also search in common subdirectories where ONNX files may be located
        var subdirs = new[] { "onnx", "model" };
        foreach (var subdir in subdirs)
        {
            var subdirPath = Path.Combine(modelDir, subdir);
            if (Directory.Exists(subdirPath))
                searchPaths.Add(subdirPath);
        }

        // Search for SentencePiece tokenizer files
        string? tokenizerPath = null;
        foreach (var searchPath in searchPaths)
        {
            var primaryPath = Path.Combine(searchPath, primaryFile);
            var fallbackPath = Path.Combine(searchPath, fallbackFile);

            if (File.Exists(primaryPath))
            {
                tokenizerPath = primaryPath;
                break;
            }
            if (File.Exists(fallbackPath))
            {
                tokenizerPath = fallbackPath;
                break;
            }
        }

        if (tokenizerPath != null)
        {
            try
            {
                // Use SentencePieceTokenizer for .spm files (supports both BPE and Unigram).
                // BOS/EOS emission is off: EncodeSource owns the special tokens. Left on, the model's own <s> was
                // prepended to every input and, absent from a Marian vocab.json, mapped to <unk> (opus-mt-ko-en:
                // [1, 1114, ...] where the reference is [1114, ...]).
                using var stream = File.OpenRead(tokenizerPath);
                return SentencePieceTokenizer.Create(stream, addBeginningOfSentence: false, addEndOfSentence: false);
            }
            catch (Exception ex)
            {
                Trace.TraceInformation($"[TranslatorTokenizer] SentencePiece tokenizer failed: {ex.Message}");
            }
        }

        // Try loading from tokenizer.json (HuggingFace format)
        foreach (var searchPath in searchPaths)
        {
            var tokenizerJsonPath = Path.Combine(searchPath, "tokenizer.json");
            if (File.Exists(tokenizerJsonPath))
            {
                try
                {
                    using var stream = File.OpenRead(tokenizerJsonPath);
                    return LlamaTokenizer.Create(stream);
                }
                catch (Exception ex)
                {
                    Trace.TraceInformation($"[TranslatorTokenizer] tokenizer.json load failed: {ex.Message}");
                }
            }
        }

        // Create a fallback tokenizer using vocab.json if available
        return CreateFallbackTokenizer(modelDir, searchPaths);
    }

    private static Tokenizer CreateFallbackTokenizer(string modelDir, IEnumerable<string>? searchPaths = null)
    {
        var paths = searchPaths?.ToList() ?? [modelDir];

        // Try to create BPE tokenizer from vocab.json and merges.txt
        foreach (var searchPath in paths)
        {
            var vocabPath = Path.Combine(searchPath, "vocab.json");
            var mergesPath = Path.Combine(searchPath, "merges.txt");

            if (File.Exists(vocabPath) && File.Exists(mergesPath))
            {
                try
                {
                    using var vocabStream = File.OpenRead(vocabPath);
                    using var mergesStream = File.OpenRead(mergesPath);
                    return CodeGenTokenizer.Create(vocabStream, mergesStream);
                }
                catch (Exception ex)
                {
                    Trace.TraceInformation($"[TranslatorTokenizer] BPE tokenizer load failed: {ex.Message}");
                }
            }
        }

        // Try to create tokenizer from tokenizer.json (Unigram/BPE models)
        Exception? lastException = null;
        foreach (var searchPath in paths)
        {
            var tokenizerJsonPath = Path.Combine(searchPath, "tokenizer.json");
            if (File.Exists(tokenizerJsonPath))
            {
                try
                {
                    var tokenizer = CreateTokenizerFromJson(tokenizerJsonPath);
                    if (tokenizer != null)
                        return tokenizer;
                }
                catch (Exception ex)
                {
                    lastException = ex;
                    // Continue searching
                }
            }
        }

        if (lastException != null)
        {
            throw new InvalidOperationException(
                $"Failed to load tokenizer from tokenizer.json: {lastException.Message}",
                lastException);
        }

        var searchedPaths = string.Join(", ", paths);
        throw new InvalidOperationException(
            $"Could not load tokenizer from model directory: {modelDir}. " +
            $"Searched paths: [{searchedPaths}]. " +
            "Expected one of: source.spm, sentencepiece.bpe.model, tokenizer.json, or vocab.json + merges.txt");
    }

    private static WordPieceTokenizer? CreateTokenizerFromJson(string tokenizerJsonPath)
    {
        var json = File.ReadAllText(tokenizerJsonPath);
        using var doc = JsonDocument.Parse(json);

        if (!doc.RootElement.TryGetProperty("model", out var model))
            return null;

        if (!model.TryGetProperty("vocab", out var vocab))
            return null;

        // Build vocab dictionary sorted by ID
        var vocabDict = new SortedDictionary<int, string>();

        if (vocab.ValueKind == JsonValueKind.Object)
        {
            foreach (var property in vocab.EnumerateObject())
            {
                vocabDict[property.Value.GetInt32()] = property.Name;
            }
        }
        else if (vocab.ValueKind == JsonValueKind.Array)
        {
            // Unigram format: [["token", score], ...]
            var index = 0;
            foreach (var item in vocab.EnumerateArray())
            {
                if (item.ValueKind == JsonValueKind.Array)
                {
                    var arr = item.EnumerateArray().ToArray();
                    if (arr.Length >= 1 && arr[0].ValueKind == JsonValueKind.String)
                    {
                        vocabDict[index] = arr[0].GetString() ?? string.Empty;
                    }
                }
                index++;
            }
        }

        if (vocabDict.Count == 0)
            return null;

        // Create vocab.txt format for WordPieceTokenizer
        // Map special tokens to BERT-compatible format
        var vocabLines = new StringBuilder();
        for (var i = 0; i < vocabDict.Count; i++)
        {
            if (vocabDict.TryGetValue(i, out var token))
            {
                // Map MarianMT special tokens to BERT-compatible tokens
                var mappedToken = token switch
                {
                    "<unk>" => "[UNK]",
                    "<pad>" => "[PAD]",
                    "<s>" => "[CLS]",
                    "</s>" => "[SEP]",
                    _ => token
                };
                vocabLines.AppendLine(mappedToken);
            }
            else
            {
                vocabLines.AppendLine(CultureInfo.InvariantCulture, $"[unused{i}]");
            }
        }

        var vocabBytes = Encoding.UTF8.GetBytes(vocabLines.ToString());
        using var vocabStream = new MemoryStream(vocabBytes);
        return WordPieceTokenizer.Create(vocabStream);
    }

    public void Dispose()
    {
        if (_disposed)
            return;

        // Tokenizers don't implement IDisposable, but we keep the pattern for future compatibility
        _disposed = true;
    }
}
