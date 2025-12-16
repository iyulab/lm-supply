namespace LMSupply.Text;

/// <summary>
/// Configuration for special tokens used by tokenizers.
/// </summary>
public sealed class SpecialTokens
{
    /// <summary>
    /// Default BERT special tokens.
    /// </summary>
    public static SpecialTokens Bert { get; } = new()
    {
        PadToken = "[PAD]",
        UnkToken = "[UNK]",
        ClsToken = "[CLS]",
        SepToken = "[SEP]",
        MaskToken = "[MASK]",
        PadTokenId = 0,
        UnkTokenId = 100,
        ClsTokenId = 101,
        SepTokenId = 102,
        MaskTokenId = 103
    };

    /// <summary>
    /// Default GPT-2 special tokens.
    /// </summary>
    public static SpecialTokens Gpt2 { get; } = new()
    {
        PadToken = "<|endoftext|>",
        UnkToken = "<|endoftext|>",
        BosToken = "<|endoftext|>",
        EosToken = "<|endoftext|>",
        PadTokenId = 50256,
        UnkTokenId = 50256,
        BosTokenId = 50256,
        EosTokenId = 50256
    };

    /// <summary>
    /// Default SentencePiece/Marian special tokens.
    /// </summary>
    public static SpecialTokens SentencePiece { get; } = new()
    {
        PadToken = "<pad>",
        UnkToken = "<unk>",
        BosToken = "<s>",
        EosToken = "</s>",
        PadTokenId = 0,
        UnkTokenId = 3,
        BosTokenId = 1,
        EosTokenId = 2
    };

    /// <summary>
    /// PAD token string.
    /// </summary>
    public string? PadToken { get; init; }

    /// <summary>
    /// UNK token string.
    /// </summary>
    public string? UnkToken { get; init; }

    /// <summary>
    /// BOS token string.
    /// </summary>
    public string? BosToken { get; init; }

    /// <summary>
    /// EOS token string.
    /// </summary>
    public string? EosToken { get; init; }

    /// <summary>
    /// CLS token string.
    /// </summary>
    public string? ClsToken { get; init; }

    /// <summary>
    /// SEP token string.
    /// </summary>
    public string? SepToken { get; init; }

    /// <summary>
    /// MASK token string.
    /// </summary>
    public string? MaskToken { get; init; }

    /// <summary>
    /// PAD token ID.
    /// </summary>
    public int PadTokenId { get; init; }

    /// <summary>
    /// UNK token ID.
    /// </summary>
    public int UnkTokenId { get; init; } = 0;

    /// <summary>
    /// BOS token ID.
    /// </summary>
    public int? BosTokenId { get; init; }

    /// <summary>
    /// EOS token ID.
    /// </summary>
    public int? EosTokenId { get; init; }

    /// <summary>
    /// CLS token ID.
    /// </summary>
    public int? ClsTokenId { get; init; }

    /// <summary>
    /// SEP token ID.
    /// </summary>
    public int? SepTokenId { get; init; }

    /// <summary>
    /// MASK token ID.
    /// </summary>
    public int? MaskTokenId { get; init; }

    /// <summary>
    /// Creates special tokens from vocabulary dictionary.
    /// </summary>
    public static SpecialTokens FromVocabulary(IReadOnlyDictionary<string, int> vocab)
    {
        return new SpecialTokens
        {
            PadToken = FindToken(vocab, "[PAD]", "<pad>"),
            UnkToken = FindToken(vocab, "[UNK]", "<unk>"),
            BosToken = FindToken(vocab, "<s>", "[BOS]"),
            EosToken = FindToken(vocab, "</s>", "[EOS]"),
            ClsToken = FindToken(vocab, "[CLS]"),
            SepToken = FindToken(vocab, "[SEP]"),
            MaskToken = FindToken(vocab, "[MASK]"),
            PadTokenId = GetId(vocab, "[PAD]", "<pad>") ?? 0,
            UnkTokenId = GetId(vocab, "[UNK]", "<unk>") ?? 0,
            BosTokenId = GetId(vocab, "<s>", "[BOS]"),
            EosTokenId = GetId(vocab, "</s>", "[EOS]"),
            ClsTokenId = GetId(vocab, "[CLS]"),
            SepTokenId = GetId(vocab, "[SEP]"),
            MaskTokenId = GetId(vocab, "[MASK]")
        };
    }

    private static string? FindToken(IReadOnlyDictionary<string, int> vocab, params string[] candidates)
    {
        foreach (var candidate in candidates)
        {
            if (vocab.ContainsKey(candidate))
                return candidate;
        }
        return null;
    }

    private static int? GetId(IReadOnlyDictionary<string, int> vocab, params string[] candidates)
    {
        foreach (var candidate in candidates)
        {
            if (vocab.TryGetValue(candidate, out var id))
                return id;
        }
        return null;
    }
}
