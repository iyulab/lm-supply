using System.Diagnostics.CodeAnalysis;
using System.IO.Compression;

namespace LMSupply.Synthesizer.Core.English;

/// <summary>
/// The CMU Pronouncing Dictionary as a lookup from a lower-case word to its first-listed pronunciation. The dictionary
/// ships inside the assembly (see <c>Data/cmudict</c>), so a lookup needs no file and no network.
/// </summary>
/// <remarks>
/// The entries are held as two byte arrays (the words, the pronunciations) and their offsets, sorted by word, and a
/// lookup is a binary search: about 3 MB for 126,000 words, where a dictionary of strings would take several times that.
/// </remarks>
internal sealed class PronunciationDictionary
{
    private const string ResourceName = "LMSupply.Synthesizer.cmudict.dict.gz";
    private const int MaxWordLength = 64;

    private static readonly Lazy<PronunciationDictionary> Shipped = new(LoadShipped);

    // Entry i is the word _words[_wordEnds[i - 1].._wordEnds[i]] with the phonemes _phones[_phoneEnds[i - 1].._phoneEnds[i]].
    private readonly byte[] _words;
    private readonly int[] _wordEnds;
    private readonly byte[] _phones;
    private readonly int[] _phoneEnds;

    private PronunciationDictionary(byte[] words, int[] wordEnds, byte[] phones, int[] phoneEnds)
    {
        _words = words;
        _wordEnds = wordEnds;
        _phones = phones;
        _phoneEnds = phoneEnds;
    }

    /// <summary>The dictionary that ships with the package, read on first use.</summary>
    public static PronunciationDictionary Default => Shipped.Value;

    public int Count => _wordEnds.Length;

    /// <summary>Looks up a word; the dictionary's words are lower-case.</summary>
    public bool TryGetValue(ReadOnlySpan<char> word, [NotNullWhen(true)] out Phone[]? phones)
    {
        phones = null;
        if (word.IsEmpty || word.Length > MaxWordLength)
            return false;

        Span<byte> key = stackalloc byte[word.Length];
        for (var i = 0; i < word.Length; i++)
        {
            if (word[i] > 0x7F)
                return false;
            key[i] = (byte)word[i];
        }

        var low = 0;
        var high = Count - 1;
        while (low <= high)
        {
            var middle = (low + high) >>> 1;
            var order = Word(middle).SequenceCompareTo(key);
            if (order == 0)
            {
                var packed = _phones.AsSpan(Start(_phoneEnds, middle), _phoneEnds[middle] - Start(_phoneEnds, middle));
                phones = new Phone[packed.Length];
                for (var i = 0; i < packed.Length; i++)
                    phones[i] = Phone.Unpack(packed[i]);
                return true;
            }

            if (order < 0)
                low = middle + 1;
            else
                high = middle - 1;
        }

        return false;
    }

    private ReadOnlySpan<byte> Word(int index) => _words.AsSpan(Start(_wordEnds, index), _wordEnds[index] - Start(_wordEnds, index));

    private static int Start(int[] ends, int index) => index == 0 ? 0 : ends[index - 1];

    /// <summary>The shipped dictionary's text, as the CMU file has it.</summary>
    internal static Stream OpenShipped()
    {
        var resource = typeof(PronunciationDictionary).Assembly.GetManifestResourceStream(ResourceName)
            ?? throw new InvalidOperationException($"The pronunciation dictionary '{ResourceName}' is missing from the assembly.");
        return new GZipStream(resource, CompressionMode.Decompress);
    }

    private static PronunciationDictionary LoadShipped()
    {
        using var gzip = OpenShipped();
        using var text = new MemoryStream(capacity: 4 * 1024 * 1024);
        gzip.CopyTo(text);
        return Parse(text.GetBuffer().AsSpan(0, (int)text.Length));
    }

    /// <summary>
    /// Parses the dictionary's text: one entry per line, the word, then its phonemes separated by spaces, then
    /// optionally <c>#</c> and a comment. A further pronunciation of a word is listed as <c>word(2)</c>; only the first
    /// one is kept.
    /// </summary>
    public static PronunciationDictionary Parse(ReadOnlySpan<byte> text)
    {
        var entries = new List<(int WordStart, int WordLength, int PhoneStart, int PhoneLength)>(140_000);
        var phones = new List<byte>(1_000_000);
        var sorted = true;
        ReadOnlySpan<byte> previous = default;
        Span<char> token = stackalloc char[8];

        var lineStart = 0;
        while (lineStart < text.Length)
        {
            var lineLength = text[lineStart..].IndexOf((byte)'\n');
            if (lineLength < 0)
                lineLength = text.Length - lineStart;
            var line = text.Slice(lineStart, lineLength);
            var lineOffset = lineStart;
            lineStart += lineLength + 1;

            var comment = line.IndexOf((byte)'#');
            if (comment >= 0)
                line = line[..comment];
            line = line.TrimEnd((byte)'\r').TrimEnd((byte)' ');

            var wordLength = line.IndexOf((byte)' ');
            if (wordLength <= 0 || line[wordLength - 1] == (byte)')')
                continue;

            var word = line[..wordLength];
            var phoneStart = phones.Count;
            foreach (var range in line[(wordLength + 1)..].Split((byte)' '))
            {
                var bytes = line[(wordLength + 1)..][range];
                if (bytes.IsEmpty)
                    continue;
                if (bytes.Length > token.Length)
                    throw new FormatException($"The pronunciation dictionary has an unreadable phoneme at byte {lineOffset}.");

                for (var i = 0; i < bytes.Length; i++)
                    token[i] = (char)bytes[i];
                if (!Arpabet.TryParse(token[..bytes.Length], out var phone))
                    throw new FormatException($"The pronunciation dictionary has an unknown phoneme '{token[..bytes.Length]}' at byte {lineOffset}.");
                phones.Add(phone.Pack());
            }

            if (phones.Count == phoneStart)
                continue;

            sorted &= previous.SequenceCompareTo(word) < 0;
            previous = word;
            entries.Add((lineOffset, wordLength, phoneStart, phones.Count - phoneStart));
        }

        if (!sorted)
        {
            // The words are almost in order already; where the file's order is not the byte order, or a word is listed
            // twice, sort and keep the first listing.
            var source = text.ToArray();
            var ordered = entries
                .Select((entry, index) => (entry, index))
                .OrderBy(e => e, Comparer<((int WordStart, int WordLength, int, int) entry, int index)>.Create((a, b) =>
                {
                    var order = source.AsSpan(a.entry.WordStart, a.entry.WordLength)
                        .SequenceCompareTo(source.AsSpan(b.entry.WordStart, b.entry.WordLength));
                    return order != 0 ? order : a.index.CompareTo(b.index);
                }))
                .Select(e => e.entry)
                .ToList();

            entries.Clear();
            foreach (var entry in ordered)
            {
                if (entries.Count > 0)
                {
                    var last = entries[^1];
                    if (text.Slice(last.WordStart, last.WordLength).SequenceEqual(text.Slice(entry.WordStart, entry.WordLength)))
                        continue;
                }

                entries.Add(entry);
            }
        }

        var words = new byte[entries.Sum(e => e.WordLength)];
        var wordEnds = new int[entries.Count];
        var packed = new byte[entries.Sum(e => e.PhoneLength)];
        var phoneEnds = new int[entries.Count];
        var wordOffset = 0;
        var phoneOffset = 0;
        var allPhones = System.Runtime.InteropServices.CollectionsMarshal.AsSpan(phones);
        for (var i = 0; i < entries.Count; i++)
        {
            var entry = entries[i];
            text.Slice(entry.WordStart, entry.WordLength).CopyTo(words.AsSpan(wordOffset));
            wordOffset += entry.WordLength;
            wordEnds[i] = wordOffset;

            allPhones.Slice(entry.PhoneStart, entry.PhoneLength).CopyTo(packed.AsSpan(phoneOffset));
            phoneOffset += entry.PhoneLength;
            phoneEnds[i] = phoneOffset;
        }

        return new PronunciationDictionary(words, wordEnds, packed, phoneEnds);
    }
}
