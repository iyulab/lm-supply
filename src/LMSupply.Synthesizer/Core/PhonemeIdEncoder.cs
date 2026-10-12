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

    private PhonemeIdEncoder(Dictionary<int, long[]> ids, long[] pad, long[] start, long[] end)
    {
        _ids = ids;
        _pad = pad;
        _start = start;
        _end = end;
    }

    public static PhonemeIdEncoder FromIdMap(IReadOnlyDictionary<string, long[]> idMap, SynthesizerModelInfo catalog)
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

        return new PhonemeIdEncoder(ids, Required(Pad), Required(Start), Required(End));

        long[] Required(int symbol) => ids.TryGetValue(symbol, out var value)
            ? value
            : throw new ModelLoadException(
                $"The voice config '{catalog.ConfigFile}' has no id for '{(char)symbol}' in its phoneme_id_map; " +
                "a Piper voice needs the pad '_', start '^' and end '$' symbols.",
                catalog.Id);
    }

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
        foreach (var phoneme in phonemes.Normalize(NormalizationForm.FormD).EnumerateRunes())
        {
            if (!_ids.TryGetValue(phoneme.Value, out var phonemeIds))
            {
                unknown ??= new StringBuilder();
                unknown.Append(phoneme.ToString());
                continue;
            }

            ids.AddRange(phonemeIds);
            ids.AddRange(_pad);
            count++;
        }

        ids.AddRange(_end);
        return new EncodedSentence([.. ids], count, unknown?.ToString() ?? string.Empty);
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
