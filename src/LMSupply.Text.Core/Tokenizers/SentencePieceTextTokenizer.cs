namespace LMSupply.Text;

/// <summary>
/// SentencePiece tokenizer implementation for text encoding (translation models).
/// </summary>
internal sealed class SentencePieceTextTokenizer : ITextTokenizer
{
    private readonly Tokenizer _tokenizer;
    private readonly SpecialTokens _specialTokens;

    public int VocabSize { get; }
    public int PadTokenId => _specialTokens.PadTokenId;
    public int UnkTokenId => _specialTokens.UnkTokenId;
    public int? BosTokenId => _specialTokens.BosTokenId;
    public int? EosTokenId => _specialTokens.EosTokenId;
    public int? ClsTokenId => _specialTokens.ClsTokenId;
    public int? SepTokenId => _specialTokens.SepTokenId;

    public SentencePieceTextTokenizer(Tokenizer tokenizer, SpecialTokens specialTokens, int vocabSize = 32000)
    {
        _tokenizer = tokenizer;
        _specialTokens = specialTokens;
        VocabSize = vocabSize;
    }

    public int[] Encode(string text, bool addSpecialTokens = true)
    {
        var ids = _tokenizer.EncodeToIds(text).ToArray();

        if (!addSpecialTokens)
            return ids;

        // SentencePiece typically uses <s> and </s> for BOS/EOS
        var hasBos = _specialTokens.BosTokenId.HasValue;
        var hasEos = _specialTokens.EosTokenId.HasValue;
        var extraTokens = (hasBos ? 1 : 0) + (hasEos ? 1 : 0);

        if (extraTokens == 0)
            return ids;

        var result = new int[ids.Length + extraTokens];
        var pos = 0;

        if (hasBos)
        {
            result[pos++] = _specialTokens.BosTokenId!.Value;
        }

        Array.Copy(ids, 0, result, pos, ids.Length);
        pos += ids.Length;

        if (hasEos)
        {
            result[pos] = _specialTokens.EosTokenId!.Value;
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
               tokenId == BosTokenId ||
               tokenId == EosTokenId ||
               tokenId == ClsTokenId ||
               tokenId == SepTokenId;
    }

    public void Dispose()
    {
        // Tokenizer doesn't implement IDisposable
    }
}
