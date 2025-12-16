namespace LMSupply.Text;

/// <summary>
/// WordPiece tokenizer for pair encoding (cross-encoders/rerankers).
/// </summary>
internal sealed class WordPiecePairTokenizer : IPairTokenizer
{
    private readonly Tokenizer _tokenizer;
    private readonly SpecialTokens _specialTokens;
    private readonly int _maxSequenceLength;

    public int VocabSize => 30522; // Default BERT vocab size
    public int PadTokenId => _specialTokens.PadTokenId;
    public int UnkTokenId => _specialTokens.UnkTokenId;
    public int? BosTokenId => _specialTokens.BosTokenId;
    public int? EosTokenId => _specialTokens.EosTokenId;
    public int? ClsTokenId => _specialTokens.ClsTokenId;
    public int? SepTokenId => _specialTokens.SepTokenId;
    public int MaxSequenceLength => _maxSequenceLength;

    public WordPiecePairTokenizer(Tokenizer tokenizer, SpecialTokens specialTokens, int maxSequenceLength)
    {
        _tokenizer = tokenizer;
        _specialTokens = specialTokens;
        _maxSequenceLength = maxSequenceLength;
    }

    public int[] Encode(string text, bool addSpecialTokens = true)
    {
        var ids = _tokenizer.EncodeToIds(text).ToArray();

        if (!addSpecialTokens)
            return ids;

        var result = new int[ids.Length + 2];
        result[0] = _specialTokens.ClsTokenId ?? _specialTokens.BosTokenId ?? 101;
        Array.Copy(ids, 0, result, 1, ids.Length);
        result[^1] = _specialTokens.SepTokenId ?? _specialTokens.EosTokenId ?? 102;

        return result;
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
        return tokenId == PadTokenId ||
               tokenId == UnkTokenId ||
               tokenId == ClsTokenId ||
               tokenId == SepTokenId ||
               tokenId == BosTokenId ||
               tokenId == EosTokenId;
    }

    public EncodedSequence EncodeSequence(string text, int? maxLength = null)
    {
        var length = maxLength ?? _maxSequenceLength;
        var tokens = _tokenizer.EncodeToIds(text).ToArray();

        var availableLength = length - 2;
        var contentLength = Math.Min(tokens.Length, availableLength);

        var inputIds = new long[length];
        var attentionMask = new long[length];

        inputIds[0] = _specialTokens.ClsTokenId ?? 101;
        attentionMask[0] = 1;

        for (int i = 0; i < contentLength; i++)
        {
            inputIds[i + 1] = tokens[i];
            attentionMask[i + 1] = 1;
        }

        inputIds[contentLength + 1] = _specialTokens.SepTokenId ?? 102;
        attentionMask[contentLength + 1] = 1;

        for (int i = contentLength + 2; i < length; i++)
        {
            inputIds[i] = PadTokenId;
        }

        return new EncodedSequence(inputIds, attentionMask, contentLength + 2);
    }

    public EncodedBatch EncodeBatch(IReadOnlyList<string> texts, int? maxLength = null)
    {
        var length = maxLength ?? _maxSequenceLength;
        var batch = new EncodedBatch(texts.Count, length);

        for (int i = 0; i < texts.Count; i++)
        {
            var encoded = EncodeSequence(texts[i], length);
            batch.SetSequence(i, encoded);
        }

        return batch;
    }

    public EncodedPair EncodePair(string text1, string text2, int? maxLength = null)
    {
        var length = maxLength ?? _maxSequenceLength;

        var tokens1 = _tokenizer.EncodeToIds(text1).ToArray();
        var tokens2 = _tokenizer.EncodeToIds(text2).ToArray();

        // Format: [CLS] text1 [SEP] text2 [SEP]
        var availableLength = length - 3; // Reserve 3 for special tokens
        var totalTokens = tokens1.Length + tokens2.Length;

        int len1, len2;
        if (totalTokens <= availableLength)
        {
            len1 = tokens1.Length;
            len2 = tokens2.Length;
        }
        else
        {
            // Truncate proportionally, but ensure at least some tokens from each
            var ratio = (double)availableLength / totalTokens;
            len1 = Math.Max(1, (int)(tokens1.Length * ratio));
            len2 = Math.Max(1, Math.Min(tokens2.Length, availableLength - len1));
            len1 = Math.Min(tokens1.Length, availableLength - len2);
        }

        var inputIds = new long[length];
        var attentionMask = new long[length];
        var tokenTypeIds = new long[length];

        var pos = 0;

        // [CLS]
        inputIds[pos] = _specialTokens.ClsTokenId ?? 101;
        attentionMask[pos] = 1;
        tokenTypeIds[pos] = 0;
        pos++;

        // text1 tokens
        for (int i = 0; i < len1; i++)
        {
            inputIds[pos] = tokens1[i];
            attentionMask[pos] = 1;
            tokenTypeIds[pos] = 0;
            pos++;
        }

        // [SEP]
        inputIds[pos] = _specialTokens.SepTokenId ?? 102;
        attentionMask[pos] = 1;
        tokenTypeIds[pos] = 0;
        pos++;

        // text2 tokens
        for (int i = 0; i < len2; i++)
        {
            inputIds[pos] = tokens2[i];
            attentionMask[pos] = 1;
            tokenTypeIds[pos] = 1;
            pos++;
        }

        // [SEP]
        inputIds[pos] = _specialTokens.SepTokenId ?? 102;
        attentionMask[pos] = 1;
        tokenTypeIds[pos] = 1;
        pos++;

        // Padding
        for (int i = pos; i < length; i++)
        {
            inputIds[i] = PadTokenId;
            // attentionMask and tokenTypeIds already 0
        }

        return new EncodedPair(inputIds, attentionMask, tokenTypeIds, pos);
    }

    public EncodedPairBatch EncodePairBatch(string text1, IReadOnlyList<string> texts2, int? maxLength = null)
    {
        var length = maxLength ?? _maxSequenceLength;
        var batch = new EncodedPairBatch(texts2.Count, length);

        for (int i = 0; i < texts2.Count; i++)
        {
            var encoded = EncodePair(text1, texts2[i], length);
            batch.SetPair(i, encoded);
        }

        return batch;
    }

    public void Dispose()
    {
        // Tokenizer doesn't implement IDisposable
    }
}
