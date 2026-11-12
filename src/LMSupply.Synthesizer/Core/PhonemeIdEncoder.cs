using System.Buffers;
using System.Text;
using LMSupply.Exceptions;
using LMSupply.Synthesizer.Models;

namespace LMSupply.Synthesizer.Core;

/// <summary>
/// Encodes one sentence of phonemes as the ids a Piper voice was trained on: the start symbol, a pad, then every phoneme
/// followed by a pad, then the end symbol (<c>^ _ p1 _ p2 _ ... pn _ $</c>).
/// </summary>
internal sealed class PhonemeIdEncoder
{
    private const int Pad = '_';
    private const int Start = '^';
    private const int End = '$';

    // Keyed by Unicode code point: a phoneme is one code point, whatever its UTF-16 length.
    private readonly Dictionary<int, long[]> _ids;
    private readonly long[] _pad;
    private readonly long[] _start;
    private readonly long[] _end;

    // Pairs of phonemes the voice was trained to see as one symbol (a diphthong), with that symbol's ids.
    private readonly Dictionary<(int First, int Second), long[]> _clusters;

    private PhonemeIdEncoder(Dictionary<int, long[]> ids, long[] pad, long[] start, long[] end, Dictionary<(int First, int Second), long[]> clusters)
    {
        _ids = ids;
        _clusters = clusters;
        _pad = pad;
        _start = start;
        _end = end;
    }

    /// <param name="idMap">The voice's <c>phoneme_id_map</c>.</param>
    /// <param name="catalog">The catalog entry, for messages.</param>
    /// <param name="vowelClusters">
    /// The voice's <c>vowel_clusters</c>: pairs of phonemes (<c>["e", "ɪ"]</c>) that the voice takes as the one symbol their
    /// concatenation names in <paramref name="idMap"/> (<c>"eɪ"</c>). Newer voices are trained on that symbol, not on its two
    /// parts; a config without the list is encoded one phoneme at a time, as before.
    /// </param>
    public static PhonemeIdEncoder FromIdMap(
        IReadOnlyDictionary<string, long[]> idMap,
        SynthesizerModelInfo catalog,
        IReadOnlyList<IReadOnlyList<string>>? vowelClusters = null)
    {
        var ids = new Dictionary<int, long[]>(idMap.Count);
        foreach (var (symbol, symbolIds) in idMap)
        {
            // A key of several code points can never be a phoneme: phonemes are looked up one code point at a time.
            // Some published configs carry such keys (diphthongs); they are dead entries, not a reason to refuse the voice.
            if (symbolIds is not { Length: > 0 } || !TryGetSingleCodePoint(symbol, out var codePoint))
                continue;

            ids[codePoint] = symbolIds;
        }

        return new PhonemeIdEncoder(ids, Required(Pad), Required(Start), Required(End), ReadClusters(idMap, vowelClusters));

        long[] Required(int symbol) => ids.TryGetValue(symbol, out var value)
            ? value
            : throw new ModelLoadException(
                $"The voice config '{catalog.ConfigFile}' has no id for '{(char)symbol}' in its phoneme_id_map; " +
                "a Piper voice needs the pad '_', start '^' and end '$' symbols.",
                catalog.Id);
    }

    /// <summary>Whether the voice has an id for <paramref name="codePoint"/>.</summary>
    public bool Has(int codePoint) => _ids.ContainsKey(codePoint);

    /// <summary>
    /// Encodes one sentence. The phonemes are decomposed first (NFD), as Piper does to its phonemizer's output: a voice
    /// has ids for a base letter and its combining mark, not for every precomposed letter. A phoneme the voice has no id
    /// for is skipped and reported in <see cref="EncodedSentence.Unknown"/>.
    /// </summary>
    public EncodedSentence Encode(string phonemes)
    {
        ArgumentNullException.ThrowIfNull(phonemes);

        var ids = new List<long>(_start.Length + _end.Length + (phonemes.Length + 1) * 2);
        ids.AddRange(_start);
        ids.AddRange(_pad);

        var count = 0;
        StringBuilder? unknown = null;
        var runes = phonemes.Normalize(NormalizationForm.FormD).EnumerateRunes().ToArray();
        for (var i = 0; i < runes.Length; i++)
        {
            // Two phonemes the voice takes as one symbol (a diphthong) are one phoneme here too.
            if (_clusters.Count > 0 && i + 1 < runes.Length && _clusters.TryGetValue((runes[i].Value, runes[i + 1].Value), out var clusterIds))
            {
                ids.AddRange(clusterIds);
                ids.AddRange(_pad);
                count++;
                i++;
                continue;
            }

            if (!_ids.TryGetValue(runes[i].Value, out var phonemeIds))
            {
                unknown ??= new StringBuilder();
                unknown.Append(runes[i].ToString());
                continue;
            }

            ids.AddRange(phonemeIds);
            ids.AddRange(_pad);
            count++;
        }

        ids.AddRange(_end);
        return new EncodedSentence([.. ids], count, unknown?.ToString() ?? string.Empty);
    }

    /// <summary>
    /// The pairs of <paramref name="vowelClusters"/> that name a symbol in <paramref name="idMap"/>. An entry that is not a
    /// pair of single phonemes, or whose joined symbol the voice has no id for, is left out: the pair is then encoded as its
    /// two phonemes, which is all such a voice can do with it.
    /// </summary>
    private static Dictionary<(int First, int Second), long[]> ReadClusters(
        IReadOnlyDictionary<string, long[]> idMap,
        IReadOnlyList<IReadOnlyList<string>>? vowelClusters)
    {
        var clusters = new Dictionary<(int First, int Second), long[]>();
        if (vowelClusters is null)
            return clusters;

        foreach (var pair in vowelClusters)
        {
            if (pair is not { Count: 2 }
                || !TryGetSingleCodePoint(pair[0], out var first)
                || !TryGetSingleCodePoint(pair[1], out var second)
                || !idMap.TryGetValue(pair[0] + pair[1], out var symbolIds)
                || symbolIds is not { Length: > 0 })
                continue;

            clusters[(first, second)] = symbolIds;
        }

        return clusters;
    }

    private static bool TryGetSingleCodePoint(string symbol, out int codePoint)
    {
        codePoint = 0;
        if (Rune.DecodeFromUtf16(symbol, out var rune, out var consumed) != OperationStatus.Done || consumed != symbol.Length)
            return false;

        codePoint = rune.Value;
        return true;
    }
}

/// <summary>One sentence as model input: its ids, how many phonemes they hold, and the phonemes that had no id.</summary>
internal readonly record struct EncodedSentence(long[] Ids, int PhonemeCount, string Unknown);
