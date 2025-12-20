using System.Text.Json;
using LMSupply.Text;
using Microsoft.ML.Tokenizers;

namespace LMSupply.ImageGenerator.Tokenizers;

/// <summary>
/// CLIP text tokenizer for encoding prompts.
/// Uses BPE tokenization with special handling for CLIP's vocabulary.
/// </summary>
internal sealed class ClipTokenizer : ITextTokenizer
{
    private readonly Tokenizer _tokenizer;
    private readonly int _bosTokenId; // <|startoftext|>
    private readonly int _eosTokenId; // <|endoftext|>
    private readonly int _padTokenId;
    private readonly int _unkTokenId;
    private readonly int _maxLength;

    /// <summary>
    /// Default CLIP vocabulary size.
    /// </summary>
    public const int DefaultVocabSize = 49408;

    /// <summary>
    /// Default maximum sequence length for CLIP.
    /// </summary>
    public const int DefaultMaxLength = 77;

    public int VocabSize { get; }
    public int PadTokenId => _padTokenId;
    public int UnkTokenId => _unkTokenId;
    public int? BosTokenId => _bosTokenId;
    public int? EosTokenId => _eosTokenId;
    public int? ClsTokenId => null;
    public int? SepTokenId => null;

    /// <summary>
    /// Maximum sequence length including special tokens.
    /// </summary>
    public int MaxLength => _maxLength;

    private ClipTokenizer(
        Tokenizer tokenizer,
        int vocabSize,
        int bosTokenId,
        int eosTokenId,
        int padTokenId,
        int unkTokenId,
        int maxLength)
    {
        _tokenizer = tokenizer;
        VocabSize = vocabSize;
        _bosTokenId = bosTokenId;
        _eosTokenId = eosTokenId;
        _padTokenId = padTokenId;
        _unkTokenId = unkTokenId;
        _maxLength = maxLength;
    }

    /// <summary>
    /// Creates a CLIP tokenizer from model directory.
    /// </summary>
    /// <param name="modelDir">Path to model directory containing vocab.json and merges.txt.</param>
    /// <param name="maxLength">Maximum sequence length (default: 77).</param>
    /// <returns>CLIP tokenizer instance.</returns>
    public static ClipTokenizer FromDirectory(string modelDir, int maxLength = DefaultMaxLength)
    {
        var vocabPath = Path.Combine(modelDir, "vocab.json");
        var mergesPath = Path.Combine(modelDir, "merges.txt");

        // Try tokenizer subdirectory if files not found
        if (!File.Exists(vocabPath) || !File.Exists(mergesPath))
        {
            var tokenizerDir = Path.Combine(modelDir, "tokenizer");
            if (Directory.Exists(tokenizerDir))
            {
                var altVocabPath = Path.Combine(tokenizerDir, "vocab.json");
                var altMergesPath = Path.Combine(tokenizerDir, "merges.txt");
                if (File.Exists(altVocabPath) && File.Exists(altMergesPath))
                {
                    vocabPath = altVocabPath;
                    mergesPath = altMergesPath;
                }
            }
        }

        if (!File.Exists(vocabPath) || !File.Exists(mergesPath))
        {
            throw new FileNotFoundException(
                $"CLIP tokenizer requires vocab.json and merges.txt in: {modelDir}");
        }

        // Load vocabulary to get special token IDs
        var (vocabSize, bosId, eosId, padId, unkId) = LoadVocabularyInfo(vocabPath);

        // Create BPE tokenizer
        using var vocabStream = File.OpenRead(vocabPath);
        using var mergesStream = File.OpenRead(mergesPath);
        var tokenizer = CodeGenTokenizer.Create(vocabStream, mergesStream);

        return new ClipTokenizer(
            tokenizer,
            vocabSize,
            bosId,
            eosId,
            padId,
            unkId,
            maxLength);
    }

    /// <summary>
    /// Encodes text for CLIP text encoder.
    /// Format: [BOS] tokens... [EOS] [PAD...] to maxLength
    /// </summary>
    public int[] Encode(string text, bool addSpecialTokens = true)
    {
        // Preprocess text (lowercase for CLIP)
        var processedText = text.ToLowerInvariant();

        // Tokenize
        var ids = _tokenizer.EncodeToIds(processedText);
        var tokenList = ids.ToList();

        if (addSpecialTokens)
        {
            // Add BOS at start
            tokenList.Insert(0, _bosTokenId);

            // Truncate if needed (leave room for EOS)
            if (tokenList.Count > _maxLength - 1)
            {
                tokenList = tokenList.Take(_maxLength - 1).ToList();
            }

            // Add EOS at end
            tokenList.Add(_eosTokenId);

            // Pad to maxLength
            while (tokenList.Count < _maxLength)
            {
                tokenList.Add(_padTokenId);
            }
        }

        return [.. tokenList];
    }

    /// <summary>
    /// Encodes text and returns input_ids as a 2D array for ONNX model.
    /// Shape: [1, maxLength] (int64)
    /// </summary>
    public long[] EncodeForModel(string text)
    {
        var ids = Encode(text, addSpecialTokens: true);
        return ids.Select(id => (long)id).ToArray();
    }

    /// <summary>
    /// Encodes a batch of prompts.
    /// </summary>
    /// <param name="texts">Texts to encode.</param>
    /// <returns>2D array of shape [batchSize, maxLength].</returns>
    public long[,] EncodeBatch(IReadOnlyList<string> texts)
    {
        var batch = new long[texts.Count, _maxLength];

        for (int i = 0; i < texts.Count; i++)
        {
            var ids = EncodeForModel(texts[i]);
            for (int j = 0; j < _maxLength; j++)
            {
                batch[i, j] = ids[j];
            }
        }

        return batch;
    }

    public string Decode(ReadOnlySpan<int> tokenIds, bool skipSpecialTokens = true)
    {
        var ids = skipSpecialTokens
            ? tokenIds.ToArray().Where(id => !IsSpecialToken(id))
            : tokenIds.ToArray().AsEnumerable();

        return _tokenizer.Decode(ids) ?? string.Empty;
    }

    public bool IsSpecialToken(int tokenId)
    {
        return tokenId == _bosTokenId ||
               tokenId == _eosTokenId ||
               tokenId == _padTokenId ||
               tokenId == _unkTokenId;
    }

    public void Dispose()
    {
        // Tokenizer doesn't implement IDisposable
    }

    private static (int vocabSize, int bosId, int eosId, int padId, int unkId) LoadVocabularyInfo(string vocabPath)
    {
        var json = File.ReadAllText(vocabPath);
        using var doc = JsonDocument.Parse(json);

        var vocab = doc.RootElement;
        var vocabSize = 0;
        var bosId = -1;
        var eosId = -1;
        var padId = -1;
        var unkId = -1;

        foreach (var prop in vocab.EnumerateObject())
        {
            var id = prop.Value.GetInt32();
            vocabSize = Math.Max(vocabSize, id + 1);

            // CLIP special tokens
            if (prop.Name == "<|startoftext|>")
                bosId = id;
            else if (prop.Name == "<|endoftext|>")
                eosId = id;
        }

        // For CLIP, use endoftext as pad and unk if not found
        if (padId < 0) padId = eosId >= 0 ? eosId : 0;
        if (unkId < 0) unkId = eosId >= 0 ? eosId : 0;
        if (bosId < 0) bosId = 49406; // Default CLIP BOS
        if (eosId < 0) eosId = 49407; // Default CLIP EOS

        return (vocabSize, bosId, eosId, padId, unkId);
    }
}
