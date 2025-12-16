namespace LMSupply.Text;

/// <summary>
/// GPT-2 style BPE tokenizer implementation for text encoding.
/// </summary>
internal sealed class Gpt2TextTokenizer : ITextTokenizer
{
    private readonly Tokenizer _tokenizer;
    private readonly int _eotTokenId;

    public int VocabSize => 50257; // Default GPT-2 vocab size
    public int PadTokenId => _eotTokenId; // GPT-2 uses EOT as PAD
    public int UnkTokenId => _eotTokenId; // GPT-2 uses EOT as UNK (rarely used due to BPE)
    public int? BosTokenId => _eotTokenId;
    public int? EosTokenId => _eotTokenId;
    public int? ClsTokenId => null;
    public int? SepTokenId => null;

    public Gpt2TextTokenizer(Tokenizer tokenizer, int eotTokenId = 50256)
    {
        _tokenizer = tokenizer;
        _eotTokenId = eotTokenId;
    }

    public int[] Encode(string text, bool addSpecialTokens = true)
    {
        var ids = _tokenizer.EncodeToIds(text).ToArray();

        if (!addSpecialTokens)
            return ids;

        // GPT-2 style: optionally add EOT at end
        var result = new int[ids.Length + 1];
        Array.Copy(ids, 0, result, 0, ids.Length);
        result[^1] = _eotTokenId;

        return result;
    }

    public string Decode(ReadOnlySpan<int> tokenIds, bool skipSpecialTokens = true)
    {
        var ids = skipSpecialTokens
            ? tokenIds.ToArray().Where(id => id != _eotTokenId)
            : tokenIds.ToArray().AsEnumerable();

        return _tokenizer.Decode(ids) ?? string.Empty;
    }

    public bool IsSpecialToken(int tokenId)
    {
        return tokenId == _eotTokenId;
    }

    public void Dispose()
    {
        // Tokenizer doesn't implement IDisposable
    }
}
