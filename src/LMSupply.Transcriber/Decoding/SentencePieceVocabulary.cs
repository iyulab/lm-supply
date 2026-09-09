using System.Text;
using System.Text.RegularExpressions;

namespace LMSupply.Transcriber.Decoding;

/// <summary>
/// SentencePiece 어휘(NeMo `vocab.txt` 형식: 줄마다 <c>piece id</c>)와 그 역토큰화.
/// Parakeet TDT 계열이 쓴다 — Whisper의 byte-level BPE(<see cref="WhisperTokenizer"/>)와는 별개다.
/// </summary>
internal sealed partial class SentencePieceVocabulary
{
    private const char WordBoundary = '▁'; // ▁

    private readonly string[] _pieces;

    private SentencePieceVocabulary(string[] pieces, int blankId)
    {
        _pieces = pieces;
        BlankId = blankId;
    }

    /// <summary>어휘 크기(blank 포함).</summary>
    public int Size => _pieces.Length;

    /// <summary>Transducer blank 토큰 id(<c>&lt;blk&gt;</c>).</summary>
    public int BlankId { get; }

    /// <summary>
    /// <c>vocab.txt</c>를 읽는다. 각 줄은 <c>piece id</c>이며 piece 자체가 공백을 담을 수 있으므로
    /// 마지막 공백에서 나눈다. <c>&lt;blk&gt;</c>가 없으면 이 파일은 transducer 어휘가 아니다.
    /// </summary>
    public static async Task<SentencePieceVocabulary> LoadAsync(string vocabPath, CancellationToken cancellationToken = default)
    {
        var lines = await File.ReadAllLinesAsync(vocabPath, Encoding.UTF8, cancellationToken);
        return Parse(lines, vocabPath);
    }

    internal static SentencePieceVocabulary Parse(IReadOnlyList<string> lines, string source = "vocab.txt")
    {
        var pieces = new List<string>(lines.Count);
        var blankId = -1;
        foreach (var raw in lines)
        {
            if (raw.Length == 0) continue;
            var split = raw.LastIndexOf(' ');
            if (split <= 0 || !int.TryParse(raw.AsSpan(split + 1), out var id))
                throw new InvalidDataException($"Malformed vocabulary line in {source}: '{raw}' (expected '<piece> <id>').");
            if (id != pieces.Count)
                throw new InvalidDataException($"Vocabulary ids must be dense and ascending in {source}: got {id} at line {pieces.Count}.");

            var piece = raw[..split];
            if (piece == "<blk>") blankId = id;
            pieces.Add(piece.Replace(WordBoundary, ' '));
        }

        if (blankId < 0)
            throw new InvalidDataException($"Vocabulary {source} has no <blk> token — not a transducer vocabulary.");

        return new SentencePieceVocabulary([.. pieces], blankId);
    }

    /// <summary>토큰 id 하나의 piece(▁는 이미 공백으로 바뀌어 있다).</summary>
    public string Piece(int id) => (uint)id < (uint)_pieces.Length ? _pieces[id] : string.Empty;

    /// <summary>
    /// 토큰 id 열을 텍스트로 잇는다. 이어 붙인 뒤 선행 공백과 «단어 경계 앞이 아닌 공백»(구두점 앞 등)을 지우고
    /// 단어 경계의 공백만 하나 남긴다 — onnx-asr의 <c>re.sub(r"\A\s|\s\B|(\s)\b", …)</c>와 같은 규칙.
    /// </summary>
    public string Decode(IEnumerable<int> tokenIds)
    {
        var sb = new StringBuilder();
        foreach (var id in tokenIds)
            sb.Append(Piece(id));
        return Normalize(sb.ToString());
    }

    internal static string Normalize(string joined)
        => SpaceCleanup().Replace(joined, static m => m.Groups[1].Success ? " " : string.Empty);

    [GeneratedRegex(@"\A\s|\s\B|(\s)\b")]
    private static partial Regex SpaceCleanup();
}
