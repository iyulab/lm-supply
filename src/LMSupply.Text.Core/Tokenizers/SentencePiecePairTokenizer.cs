namespace LMSupply.Text;

/// <summary>
/// SentencePiece/Unigram tokenizer for pair encoding (cross-encoders/rerankers).
/// Supports models that use Unigram or BPE tokenization instead of WordPiece.
/// </summary>
internal sealed class SentencePiecePairTokenizer : IPairTokenizer
{
    private readonly Tokenizer _tokenizer;
    private readonly SpecialTokens _specialTokens;
    private readonly int _maxSequenceLength;
    private readonly SentencePieceIdMap _idMap;

    public int VocabSize { get; }
    public int PadTokenId => _specialTokens.PadTokenId;
    public int UnkTokenId => _specialTokens.UnkTokenId;
    public int? BosTokenId => _specialTokens.BosTokenId;
    public int? EosTokenId => _specialTokens.EosTokenId;
    public int? ClsTokenId => _specialTokens.ClsTokenId;
    public int? SepTokenId => _specialTokens.SepTokenId;
    public int MaxSequenceLength => _maxSequenceLength;

    public SentencePiecePairTokenizer(
        Tokenizer tokenizer,
        SpecialTokens specialTokens,
        int maxSequenceLength,
        int vocabSize = 32000,
        SentencePieceIdMap? idMap = null)
    {
        _tokenizer = tokenizer;
        _specialTokens = specialTokens;
        _maxSequenceLength = maxSequenceLength;
        _idMap = idMap ?? SentencePieceIdMap.Identity;
        VocabSize = vocabSize;
    }

    /// <summary>
    /// 내용 토큰만 인코딩한다 — 특수 토큰은 이 클래스가 붙이므로 하위 토크나이저가 BOS/EOS를
    /// 덧붙이지 않는다는 전제(<see cref="TokenizerFactory"/>가 그렇게 만든다)에 기대고,
    /// 모델이 선언한 어휘 배치로 id를 옮긴다.
    /// </summary>
    private int[] EncodeContent(string text)
        => _idMap.Map(_tokenizer.EncodeToIds(text).ToArray());

    public int[] Encode(string text, bool addSpecialTokens = true)
    {
        var ids = EncodeContent(text);

        if (!addSpecialTokens)
            return ids;

        // Use CLS/SEP if available (BERT-style), otherwise BOS/EOS
        var startToken = _specialTokens.ClsTokenId ?? _specialTokens.BosTokenId;
        var endToken = _specialTokens.SepTokenId ?? _specialTokens.EosTokenId;

        var hasStart = startToken.HasValue;
        var hasEnd = endToken.HasValue;
        var extraTokens = (hasStart ? 1 : 0) + (hasEnd ? 1 : 0);

        if (extraTokens == 0)
            return ids;

        var result = new int[ids.Length + extraTokens];
        var pos = 0;

        if (hasStart)
        {
            result[pos++] = startToken!.Value;
        }

        Array.Copy(ids, 0, result, pos, ids.Length);
        pos += ids.Length;

        if (hasEnd)
        {
            result[pos] = endToken!.Value;
        }

        return result;
    }

    public string Decode(ReadOnlySpan<int> tokenIds, bool skipSpecialTokens = true)
    {
        var ids = skipSpecialTokens
            ? tokenIds.ToArray().Where(id => !IsSpecialToken(id))
            : tokenIds.ToArray().AsEnumerable();

        var decoded = _tokenizer.Decode(ids);

        // SentencePiece uses ▁ (U+2581) to mark word boundaries, replace with space
        return decoded?.Replace("▁", " ").Trim() ?? string.Empty;
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
        var tokens = EncodeContent(text);

        var startToken = _specialTokens.ClsTokenId ?? _specialTokens.BosTokenId ?? 0;
        var endToken = _specialTokens.SepTokenId ?? _specialTokens.EosTokenId ?? 2;

        // maxLength is a truncation cap only: the sequence is sized to its real content so a short
        // text is not inflated to the model maximum (which made every embedding cost a full
        // max-length pass — e.g. 8192 tokens for a 10-token sentence). Batches pad to the longest
        // member in EncodeBatch.
        var availableLength = length - 2;
        var contentLength = Math.Min(tokens.Length, availableLength);
        var totalLength = contentLength + 2;

        var inputIds = new long[totalLength];
        var attentionMask = new long[totalLength];

        inputIds[0] = startToken;
        attentionMask[0] = 1;

        for (int i = 0; i < contentLength; i++)
        {
            inputIds[i + 1] = tokens[i];
            attentionMask[i + 1] = 1;
        }

        inputIds[contentLength + 1] = endToken;
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

        // Dynamic padding: pad to the longest sequence in this batch, not to the cap.
        var batch = new EncodedBatch(texts.Count, longest);
        for (int i = 0; i < texts.Count; i++)
        {
            batch.SetSequence(i, encoded[i], PadTokenId);
        }

        return batch;
    }

    public EncodedPair EncodePair(string text1, string text2, int? maxLength = null)
    {
        var length = maxLength ?? _maxSequenceLength;

        var tokens1 = EncodeContent(text1);
        var tokens2 = EncodeContent(text2);

        var startToken = _specialTokens.ClsTokenId ?? _specialTokens.BosTokenId ?? 0;
        var sepToken = _specialTokens.SepTokenId ?? _specialTokens.EosTokenId ?? 2;

        // Format: [CLS/BOS] text1 [SEP/EOS] text2 [SEP/EOS]
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

        // Sized to real content (see EncodeSequence); EncodePairBatch pads to the longest pair.
        var totalLength = len1 + len2 + 3;
        var inputIds = new long[totalLength];
        var attentionMask = new long[totalLength];
        var tokenTypeIds = new long[totalLength];

        var pos = 0;

        // [CLS/BOS]
        inputIds[pos] = startToken;
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

        // [SEP/EOS]
        inputIds[pos] = sepToken;
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

        // [SEP/EOS]
        inputIds[pos] = sepToken;
        attentionMask[pos] = 1;
        tokenTypeIds[pos] = 1;
        pos++;

        return new EncodedPair(inputIds, attentionMask, tokenTypeIds, pos);
    }

    public EncodedPairBatch EncodePairBatch(string text1, IReadOnlyList<string> texts2, int? maxLength = null)
    {
        var length = maxLength ?? _maxSequenceLength;
        var encoded = new EncodedPair[texts2.Count];
        int longest = 0;
        for (int i = 0; i < texts2.Count; i++)
        {
            encoded[i] = EncodePair(text1, texts2[i], length);
            longest = Math.Max(longest, encoded[i].Length);
        }

        // Dynamic padding: pad to the longest pair in this batch, not to the cap.
        var batch = new EncodedPairBatch(texts2.Count, longest);
        for (int i = 0; i < texts2.Count; i++)
        {
            batch.SetPair(i, encoded[i], PadTokenId);
        }

        return batch;
    }

    public void Dispose()
    {
        // Tokenizer doesn't implement IDisposable
    }
}
