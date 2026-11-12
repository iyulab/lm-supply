using System.Collections.Concurrent;
using System.Net;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace LMSupply.ImageGenerator.Tokenizers;

/// <summary>
/// The byte-pair encoding CLIP's text encoder was trained with. It differs from a GPT-2 style byte-level
/// encoding in two ways that change every token id: a word is split from its neighbours before merging (there is
/// no leading-space marker), and the last symbol of every word carries the end-of-word suffix <c>&lt;/w&gt;</c>, so
/// <c>cat</c> is the single token <c>cat&lt;/w&gt;</c> and not <c>c</c>, <c>at</c>.
/// </summary>
/// <remarks>
/// Steps, in the order the reference tokenizer applies them: clean the text (normalization, entities, whitespace,
/// lower case), split it into words and punctuation runs with a fixed pattern, map each word's UTF-8 bytes to printable
/// characters, merge adjacent symbols by the rank listed in <c>merges.txt</c>, and look each resulting symbol up in
/// <c>vocab.json</c>. The reference repairs mojibake with a library this one does not use; the other repairs it makes
/// that change tokens (character width, ligatures, curly quotes, Unicode composition) are applied.
/// </remarks>
internal sealed class ClipBpe
{
    private const string EndOfWord = "</w>";
    private const string StartOfText = "<|startoftext|>";
    private const string EndOfText = "<|endoftext|>";
    private const int MaxCachedWords = 50_000;

    // Words and runs, as the reference splits them: the two specials, English contractions, a run of letters, one digit,
    // a run of anything that is none of space, letter or digit.
    private static readonly Regex Pattern = new(
        @"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|\p{L}+|\p{N}|[^\s\p{L}\p{N}]+",
        RegexOptions.IgnoreCase | RegexOptions.CultureInvariant | RegexOptions.Compiled);

    private static readonly Regex Whitespace = new(@"\s+", RegexOptions.CultureInvariant | RegexOptions.Compiled);

    private static readonly char[] ByteToChar = BuildByteToChar();
    private static readonly Dictionary<char, byte> CharToByte = BuildCharToByte();

    private readonly Dictionary<string, int> _vocab;
    private readonly Dictionary<int, string> _tokens;
    private readonly Dictionary<(string, string), int> _ranks;
    private readonly int _unknownId;
    private readonly ConcurrentDictionary<string, int[]> _cache = new();

    private ClipBpe(Dictionary<string, int> vocab, Dictionary<(string, string), int> ranks, int unknownId)
    {
        _vocab = vocab;
        _ranks = ranks;
        _unknownId = unknownId;
        _tokens = new Dictionary<int, string>(vocab.Count);
        foreach (var (token, id) in vocab)
        {
            _tokens[id] = token;
        }
    }

    /// <summary>
    /// Reads <c>vocab.json</c> (symbol to id) and <c>merges.txt</c> (one "left right" pair per line, best first, after an
    /// optional <c>#version</c> header line).
    /// </summary>
    /// <param name="vocabJson">The vocabulary.</param>
    /// <param name="mergesText">The merge list.</param>
    /// <param name="unknownId">The id for a symbol the vocabulary does not hold.</param>
    public static ClipBpe Create(Stream vocabJson, Stream mergesText, int unknownId)
    {
        var vocab = new Dictionary<string, int>();
        using (var document = JsonDocument.Parse(vocabJson))
        {
            foreach (var entry in document.RootElement.EnumerateObject())
            {
                vocab[entry.Name] = entry.Value.GetInt32();
            }
        }

        if (vocab.Count == 0)
        {
            throw new InvalidDataException("vocab.json is empty.");
        }

        var ranks = new Dictionary<(string, string), int>();
        using var reader = new StreamReader(mergesText, Encoding.UTF8);
        while (reader.ReadLine() is { } line)
        {
            if (line.Length == 0 || line.StartsWith("#version", StringComparison.Ordinal))
            {
                continue;
            }

            var parts = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            if (parts.Length == 2)
            {
                ranks.TryAdd((parts[0], parts[1]), ranks.Count);
            }
        }

        if (ranks.Count == 0)
        {
            throw new InvalidDataException("merges.txt holds no merges.");
        }

        return new ClipBpe(vocab, ranks, unknownId);
    }

    /// <summary>
    /// Encodes <paramref name="text"/> to token ids without the start and end markers.
    /// </summary>
    public int[] Encode(string text)
    {
        var ids = new List<int>();
        foreach (Match match in Pattern.Matches(Clean(text)))
        {
            var word = match.Value;
            if (word is StartOfText or EndOfText)
            {
                ids.Add(_vocab.GetValueOrDefault(word, _unknownId));
                continue;
            }

            ids.AddRange(EncodeWord(word));
        }

        return [.. ids];
    }

    /// <summary>
    /// Decodes token ids back to text: the symbols are joined, their characters turned back into bytes, and an
    /// end-of-word suffix becomes a space. Ids the vocabulary does not hold are skipped.
    /// </summary>
    public string Decode(IEnumerable<int> ids)
    {
        var joined = new StringBuilder();
        foreach (var id in ids)
        {
            if (_tokens.TryGetValue(id, out var token))
            {
                joined.Append(token);
            }
        }

        var bytes = new List<byte>(joined.Length);
        var text = new StringBuilder();
        void Flush()
        {
            if (bytes.Count > 0)
            {
                text.Append(Encoding.UTF8.GetString(bytes.ToArray()));
                bytes.Clear();
            }
        }

        var symbols = joined.ToString();
        for (var i = 0; i < symbols.Length; i++)
        {
            if (string.CompareOrdinal(symbols, i, EndOfWord, 0, EndOfWord.Length) == 0)
            {
                Flush();
                text.Append(' ');
                i += EndOfWord.Length - 1;
            }
            else if (symbols[i] == '<' && string.CompareOrdinal(symbols, i, StartOfText, 0, StartOfText.Length) == 0)
            {
                Flush();
                text.Append(StartOfText);
                i += StartOfText.Length - 1;
            }
            else if (symbols[i] == '<' && string.CompareOrdinal(symbols, i, EndOfText, 0, EndOfText.Length) == 0)
            {
                Flush();
                text.Append(EndOfText);
                i += EndOfText.Length - 1;
            }
            else if (CharToByte.TryGetValue(symbols[i], out var b))
            {
                bytes.Add(b);
            }
        }

        Flush();
        return text.ToString().TrimEnd();
    }

    private int[] EncodeWord(string word)
    {
        if (_cache.TryGetValue(word, out var cached))
        {
            return cached;
        }

        var encoded = new StringBuilder();
        foreach (var b in Encoding.UTF8.GetBytes(word))
        {
            encoded.Append(ByteToChar[b]);
        }

        var symbols = new List<string>(encoded.Length);
        for (var i = 0; i < encoded.Length; i++)
        {
            symbols.Add(i == encoded.Length - 1 ? encoded[i] + EndOfWord : encoded[i].ToString());
        }

        Merge(symbols);

        var ids = new int[symbols.Count];
        for (var i = 0; i < ids.Length; i++)
        {
            ids[i] = _vocab.GetValueOrDefault(symbols[i], _unknownId);
        }

        if (_cache.Count < MaxCachedWords)
        {
            _cache[word] = ids;
        }

        return ids;
    }

    // Repeatedly merges the adjacent pair with the best (lowest) rank, every occurrence of it from left to right, until no
    // adjacent pair is ranked.
    private void Merge(List<string> symbols)
    {
        while (symbols.Count > 1)
        {
            var best = int.MaxValue;
            (string, string) bestPair = default;
            for (var i = 0; i < symbols.Count - 1; i++)
            {
                if (_ranks.TryGetValue((symbols[i], symbols[i + 1]), out var rank) && rank < best)
                {
                    best = rank;
                    bestPair = (symbols[i], symbols[i + 1]);
                }
            }

            if (best == int.MaxValue)
            {
                return;
            }

            var merged = new List<string>(symbols.Count);
            for (var i = 0; i < symbols.Count; i++)
            {
                if (i < symbols.Count - 1 && symbols[i] == bestPair.Item1 && symbols[i + 1] == bestPair.Item2)
                {
                    merged.Add(bestPair.Item1 + bestPair.Item2);
                    i++;
                }
                else
                {
                    merged.Add(symbols[i]);
                }
            }

            symbols.Clear();
            symbols.AddRange(merged);
        }
    }

    // The reference's text cleaning, minus mojibake repair: composition, full-width forms, ligatures and curly quotes (what
    // its normalizer changes before tokenizing), entities decoded twice, whitespace collapsed to one space, lower case.
    internal static string Clean(string text)
    {
        var sb = new StringBuilder(text.Length);
        foreach (var c in text.Normalize(NormalizationForm.FormC))
        {
            sb.Append(c switch
            {
                '　' => " ",
                >= '！' and <= '～' => ((char)(c - 0xFEE0)).ToString(),
                >= 'ﬀ' and <= 'ﬆ' => c.ToString().Normalize(NormalizationForm.FormKC),
                '‘' or '’' or '‚' or '‛' => "'",
                '“' or '”' or '„' or '‟' => "\"",
                _ => c.ToString(),
            });
        }

        var decoded = WebUtility.HtmlDecode(WebUtility.HtmlDecode(sb.ToString())).Trim();
        return Whitespace.Replace(decoded, " ").Trim().ToLowerInvariant();
    }

    // The printable character each byte stands for: bytes that are already printable (and not a space or control) keep
    // their own code point, the rest are moved above 255 in order.
    private static char[] BuildByteToChar()
    {
        var map = new char[256];
        var next = 256;
        for (var b = 0; b < 256; b++)
        {
            var printable = (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF);
            map[b] = printable ? (char)b : (char)next++;
        }

        return map;
    }

    private static Dictionary<char, byte> BuildCharToByte()
    {
        var map = new Dictionary<char, byte>(256);
        for (var b = 0; b < 256; b++)
        {
            map[ByteToChar[b]] = (byte)b;
        }

        return map;
    }
}
