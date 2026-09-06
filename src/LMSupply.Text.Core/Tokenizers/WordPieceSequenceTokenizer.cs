namespace LMSupply.Text;

/// <summary>
/// WordPiece tokenizer implementation for sequence encoding (BERT-style models).
/// </summary>
internal sealed class WordPieceSequenceTokenizer : ISequenceTokenizer
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

    public WordPieceSequenceTokenizer(Tokenizer tokenizer, SpecialTokens specialTokens, int maxSequenceLength)
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

        // Add [CLS] and [SEP]
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

        // Calculate available space (excluding [CLS] and [SEP]). maxLength is a truncation cap
        // only — the sequence is sized to real content; EncodeBatch pads to the longest member.
        var availableLength = length - 2;
        var contentLength = Math.Min(tokens.Length, availableLength);
        var totalLength = contentLength + 2;

        var inputIds = new long[totalLength];
        var attentionMask = new long[totalLength];

        // [CLS]
        inputIds[0] = _specialTokens.ClsTokenId ?? 101;
        attentionMask[0] = 1;

        // Content tokens
        for (int i = 0; i < contentLength; i++)
        {
            inputIds[i + 1] = tokens[i];
            attentionMask[i + 1] = 1;
        }

        // [SEP]
        inputIds[contentLength + 1] = _specialTokens.SepTokenId ?? 102;
        attentionMask[contentLength + 1] = 1;

        return new EncodedSequence(inputIds, attentionMask, totalLength);
    }

    public EncodedBatch EncodeBatch(IReadOnlyList<string> texts, int? maxLength = null)
    {
        var length = maxLength ?? _maxSequenceLength;
        var encoded = new EncodedSequence[texts.Count];
        int longest = 0;
        for (int i = 0; i < texts.Count; i++)
        {
            encoded[i] = EncodeSequence(texts[i], length);
            longest = Math.Max(longest, encoded[i].Length);
        }

        var batch = new EncodedBatch(texts.Count, longest);
        for (int i = 0; i < texts.Count; i++)
        {
            batch.SetSequence(i, encoded[i], PadTokenId);
        }

        return batch;
    }

    public void Dispose()
    {
        // Tokenizer doesn't implement IDisposable
    }
}
