namespace LMSupply.Text;

/// <summary>
/// Maps the raw ids produced by a SentencePiece model to the ids of the Hugging Face vocabulary shipped with that model.
/// </summary>
/// <remarks>
/// <para>
/// In the XLM-RoBERTa family (and models derived from it, such as multilingual-e5-* and bge-m3), fairseq reserves
/// <c>0..3</c> for <c>&lt;s&gt; &lt;pad&gt; &lt;/s&gt; &lt;unk&gt;</c>, so content tokens sit one position after
/// spm's own ids. Without correcting that offset the encoder sees a different sentence, and the
/// embeddings come out well-formed but meaningless.
/// </para>
/// <para>
/// The offset is not guessed from a list of model families; it is derived by comparing how far the two
/// vocabularies actually diverge. If there is no consistent difference, nothing is mapped (<see cref="IsIdentity"/>).
/// </para>
/// </remarks>
internal sealed class SentencePieceIdMap
{
    /// <summary>The mapping that changes nothing. This is the normal state for models without an offset.</summary>
    public static SentencePieceIdMap Identity { get; } = new(0, null);

    /// <summary>Maximum number of samples used to check consistency. Walking the whole vocabulary is wasteful for large vocabularies.</summary>
    private const int SampleSize = 64;

    private readonly int _offset;
    private readonly Dictionary<int, int>? _specialIds;

    private SentencePieceIdMap(int offset, Dictionary<int, int>? specialIds)
    {
        _offset = offset;
        _specialIds = specialIds;
    }

    /// <summary>Whether this mapping leaves every id unchanged.</summary>
    public bool IsIdentity => _offset == 0 && (_specialIds is null || _specialIds.Count == 0);

    /// <summary>Maps a single raw SentencePiece id to the target vocabulary's id.</summary>
    public int Map(int id)
        => _specialIds is not null && _specialIds.TryGetValue(id, out var mapped)
            ? mapped
            : id + _offset;

    /// <summary>Maps an array of raw SentencePiece ids. Returns the original array when <see cref="IsIdentity"/> is true.</summary>
    public int[] Map(int[] ids)
    {
        ArgumentNullException.ThrowIfNull(ids);

        if (IsIdentity)
            return ids;

        var mapped = new int[ids.Length];
        for (var i = 0; i < ids.Length; i++)
            mapped[i] = Map(ids[i]);

        return mapped;
    }

    /// <summary>
    /// Builds the mapping by comparing the spm vocabulary with the target (HF) vocabulary.
    /// </summary>
    /// <param name="spmVocabulary">The SentencePiece model's own piece → id table.</param>
    /// <param name="targetVocabulary">
    /// The piece → id table declared by <c>tokenizer.json</c>. If empty, there is nothing to compare against,
    /// so <see cref="Identity"/> is returned.
    /// </param>
    /// <param name="unknownToken">The spm unknown token string.</param>
    /// <param name="unknownId">The unknown id in spm.</param>
    /// <param name="beginningOfSentenceToken">The spm BOS token string.</param>
    /// <param name="beginningOfSentenceId">The BOS id in spm.</param>
    /// <param name="endOfSentenceToken">The spm EOS token string.</param>
    /// <param name="endOfSentenceId">The EOS id in spm.</param>
    public static SentencePieceIdMap Create(
        IReadOnlyDictionary<string, int> spmVocabulary,
        IReadOnlyDictionary<string, int> targetVocabulary,
        string? unknownToken,
        int unknownId,
        string? beginningOfSentenceToken,
        int beginningOfSentenceId,
        string? endOfSentenceToken,
        int endOfSentenceId)
    {
        ArgumentNullException.ThrowIfNull(spmVocabulary);
        ArgumentNullException.ThrowIfNull(targetVocabulary);

        if (spmVocabulary.Count == 0 || targetVocabulary.Count == 0)
            return Identity;

        var specialSpmIds = new HashSet<int> { unknownId, beginningOfSentenceId, endOfSentenceId };

        int? offset = null;
        var sampled = 0;

        foreach (var (piece, spmId) in spmVocabulary)
        {
            if (sampled >= SampleSize)
                break;

            // Special tokens are laid out differently, so they cannot be used to determine the offset; they are looked up by name below.
            if (specialSpmIds.Contains(spmId))
                continue;

            if (!targetVocabulary.TryGetValue(piece, out var targetId))
                continue;

            var delta = targetId - spmId;

            // If the difference varies between pieces, the single-offset premise is wrong. Do not shift on a guess.
            if (offset is not null && offset.Value != delta)
                return Identity;

            offset = delta;
            sampled++;
        }

        if (offset is null or 0)
            return Identity;

        var specialIds = new Dictionary<int, int>();
        AddSpecial(specialIds, targetVocabulary, unknownToken, unknownId, offset.Value);
        AddSpecial(specialIds, targetVocabulary, beginningOfSentenceToken, beginningOfSentenceId, offset.Value);
        AddSpecial(specialIds, targetVocabulary, endOfSentenceToken, endOfSentenceId, offset.Value);

        return new SentencePieceIdMap(offset.Value, specialIds.Count > 0 ? specialIds : null);
    }

    /// <summary>
    /// Looks up a special token by name in the target vocabulary and adds it as a fixed mapping. Simply adding
    /// the offset would turn spm's <c>&lt;unk&gt;</c> (0) into XLM-R's <c>&lt;pad&gt;</c> (1).
    /// </summary>
    private static void AddSpecial(
        Dictionary<int, int> specialIds,
        IReadOnlyDictionary<string, int> targetVocabulary,
        string? token,
        int spmId,
        int offset)
    {
        if (string.IsNullOrEmpty(token) || !targetVocabulary.TryGetValue(token, out var targetId))
            return;

        // No override is needed when the target id already equals the offset result.
        if (targetId == spmId + offset)
            return;

        specialIds[spmId] = targetId;
    }
}
