namespace LMSupply.Text;

/// <summary>
/// SentencePiece 모델이 내는 raw id를 그 모델과 함께 배포된 Hugging Face 어휘의 id로 옮긴다.
/// </summary>
/// <remarks>
/// <para>
/// XLM-Roberta 계열(그리고 그것에서 파생된 multilingual-e5-*, bge-m3)은 fairseq가 <c>0..3</c>을
/// <c>&lt;s&gt; &lt;pad&gt; &lt;/s&gt; &lt;unk&gt;</c>로 예약했기 때문에 내용 토큰이 spm 자신의
/// id보다 한 칸 뒤에 있다. 그 어긋남을 보정하지 않으면 인코더는 «다른 문장»을 보게 되고,
/// 임베딩은 형태만 멀쩡한 채로 무의미해진다.
/// </para>
/// <para>
/// 오프셋을 모델 계열 목록으로 «알아맞히지» 않는다 — 두 어휘가 실제로 얼마나 어긋나 있는지
/// 대조해서 도출한다. 일관된 차이가 없으면 아무것도 하지 않는다(<see cref="IsIdentity"/>).
/// </para>
/// </remarks>
internal sealed class SentencePieceIdMap
{
    /// <summary>아무것도 옮기지 않는 사상. 오프셋이 없는 모델의 정상 상태다.</summary>
    public static SentencePieceIdMap Identity { get; } = new(0, null);

    /// <summary>일관성 판정에 쓸 최대 표본 수. 전량 순회는 큰 어휘에서 낭비다.</summary>
    private const int SampleSize = 64;

    private readonly int _offset;
    private readonly Dictionary<int, int>? _specialIds;

    private SentencePieceIdMap(int offset, Dictionary<int, int>? specialIds)
    {
        _offset = offset;
        _specialIds = specialIds;
    }

    /// <summary>이 사상이 id를 전혀 바꾸지 않는지 여부.</summary>
    public bool IsIdentity => _offset == 0 && (_specialIds is null || _specialIds.Count == 0);

    /// <summary>raw SentencePiece id 하나를 대상 어휘의 id로 옮긴다.</summary>
    public int Map(int id)
        => _specialIds is not null && _specialIds.TryGetValue(id, out var mapped)
            ? mapped
            : id + _offset;

    /// <summary>raw SentencePiece id 배열을 옮긴다. <see cref="IsIdentity"/>면 원본을 그대로 돌려준다.</summary>
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
    /// spm 어휘와 대상(HF) 어휘를 대조해 사상을 만든다.
    /// </summary>
    /// <param name="spmVocabulary">SentencePiece 모델 자신의 조각 → id.</param>
    /// <param name="targetVocabulary">
    /// <c>tokenizer.json</c>이 선언한 조각 → id. 비어 있으면 대조할 근거가 없으므로
    /// <see cref="Identity"/>를 돌려준다.
    /// </param>
    /// <param name="unknownToken">spm의 unknown 토큰 문자열.</param>
    /// <param name="unknownId">spm에서의 unknown id.</param>
    /// <param name="beginningOfSentenceToken">spm의 BOS 토큰 문자열.</param>
    /// <param name="beginningOfSentenceId">spm에서의 BOS id.</param>
    /// <param name="endOfSentenceToken">spm의 EOS 토큰 문자열.</param>
    /// <param name="endOfSentenceId">spm에서의 EOS id.</param>
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

            // 특수 토큰은 배치가 다르므로 오프셋 판정의 근거가 될 수 없다 — 아래에서 이름으로 다시 찾는다.
            if (specialSpmIds.Contains(spmId))
                continue;

            if (!targetVocabulary.TryGetValue(piece, out var targetId))
                continue;

            var delta = targetId - spmId;

            // 조각마다 차이가 다르면 «한 칸 밀림»이라는 전제 자체가 틀린 것이다. 추측해서 밀지 않는다.
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
    /// 특수 토큰을 «이름»으로 대상 어휘에서 찾아 고정 사상에 넣는다. 오프셋을 그냥 더하면
    /// spm의 <c>&lt;unk&gt;</c>(0)가 XLM-R의 <c>&lt;pad&gt;</c>(1)로 둔갑한다.
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

        // 오프셋을 더한 결과와 같다면 예외를 둘 이유가 없다.
        if (targetId == spmId + offset)
            return;

        specialIds[spmId] = targetId;
    }
}
