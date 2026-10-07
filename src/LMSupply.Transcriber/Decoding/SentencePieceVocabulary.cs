using System.Text;
using System.Text.RegularExpressions;

namespace LMSupply.Transcriber.Decoding;

/// <summary>
/// SentencePiece vocabulary (NeMo `vocab.txt` format: one <c>piece id</c> per line) and its detokenization.
/// Used by the Parakeet TDT family; separate from Whisper's byte-level BPE (<see cref="WhisperTokenizer"/>).
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

    /// <summary>Vocabulary size (including blank).</summary>
    public int Size => _pieces.Length;

    /// <summary>Transducer blank token id (<c>&lt;blk&gt;</c>).</summary>
    public int BlankId { get; }

    /// <summary>
    /// Reads <c>vocab.txt</c>. Each line is <c>piece id</c>; since the piece itself may contain spaces,
    /// lines are split at the last space. A file without <c>&lt;blk&gt;</c> is not a transducer vocabulary.
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
            throw new InvalidDataException($"Vocabulary {source} has no <blk> token - not a transducer vocabulary.");

        return new SentencePieceVocabulary([.. pieces], blankId);
    }

    /// <summary>The piece for a single token id (▁ has already been replaced with a space).</summary>
    public string Piece(int id) => (uint)id < (uint)_pieces.Length ? _pieces[id] : string.Empty;

    /// <summary>
    /// Joins a sequence of token ids into text. After concatenation, removes the leading space and any space not
    /// followed by a word boundary (e.g. before punctuation), keeping a single space at word boundaries — the same rule
    /// as onnx-asr's <c>re.sub(r"\A\s|\s\B|(\s)\b", …)</c>.
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
