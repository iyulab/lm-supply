using System.Diagnostics.CodeAnalysis;
using System.Globalization;
using System.Text;

namespace LMSupply.Synthesizer.Core.English;

/// <summary>
/// The accents the English voices were transcribed in, named by the phonemizer voice a voice's config declares.
/// </summary>
internal enum EnglishAccent
{
    /// <summary><c>en-us</c>: General American, as the dictionary has it.</summary>
    American,

    /// <summary><c>en</c>, <c>en-gb</c>: British.</summary>
    British,

    /// <summary><c>en-gb-x-rp</c>: Received Pronunciation.</summary>
    ReceivedPronunciation
}

/// <summary>
/// Writes a pronunciation in the dictionary's phonemes as the IPA an English voice was trained on: eSpeak NG's way of
/// writing English (a stress mark directly before the stressed vowel, length marks, <c>ɹ</c> for r, <c>ɡ</c> for g), in
/// the accent of the voice, and only in symbols the voice has an id for.
/// </summary>
/// <remarks>
/// <para>
/// The dictionary is American. For <see cref="EnglishAccent.American"/> the mapping is direct, with the vowels before r
/// and the flapped t of that accent. For the British accents the pronunciation is converted by rule: r is said only before
/// a vowel (in the word, or at the start of the next one), the vowels before a dropped r become long vowels or centring
/// diphthongs, and where American merges two British vowels the word's spelling picks between them (<c>o</c> for the
/// vowel of «lot», <c>a</c> for the vowel of «father»). What no rule can tell is left as the dictionary has it: the
/// broad <c>a</c> of «bath» stays the short one.
/// </para>
/// <para>
/// A symbol the voice's <c>phoneme_id_map</c> lacks is never written. A few have a stand-in (see
/// <see cref="StandIns"/>); stress and length marks and punctuation the voice does not know are left out; and a voice
/// that lacks any other symbol of its accent is refused when the writer is made.
/// </para>
/// </remarks>
internal sealed class EspeakEnglishWriter
{
    // What is written in place of a symbol the voice has no id for.
    private static readonly Dictionary<char, string> StandIns = new()
    {
        ['ɡ'] = "g",  // the IPA letter and the ASCII one
        ['ɾ'] = "t",  // the flap is a t
        ['ɚ'] = "ə",  // the r-coloured schwa without its colour
        ['ɐ'] = "ə",  // the article's reduced vowel
    };

    // Marks a voice can do without: they are left out where it has no id for them.
    private const string Optional = "ˈˌː,;:.?!";

    private const string Consonants = "bdfhjklmnpstvwzðŋɡɹʃʒθ";
    private const string AmericanVowels = "aeiouæɑɔəɚɛɜɪʊʌɐɾ";
    private const string BritishVowels = "aeiuɑɒɔəɛɜɪʊʌɐ";

    private readonly EnglishAccent _accent;
    private readonly bool _british;
    private readonly string _trap;
    private readonly Dictionary<char, string> _written = [];

    /// <param name="accent">The accent to write.</param>
    /// <param name="hasSymbol">Whether the voice has an id for a symbol (a Unicode code point).</param>
    /// <param name="voiceName">The voice, for the message when it cannot be written for.</param>
    /// <exception cref="NotSupportedException">The voice has no id for a phoneme of the accent, and there is no stand-in.</exception>
    public EspeakEnglishWriter(EnglishAccent accent, Func<int, bool> hasSymbol, string voiceName)
    {
        _accent = accent;
        _british = accent != EnglishAccent.American;
        _trap = accent == EnglishAccent.British ? "a" : "æ";

        var inventory = " " + Optional + Consonants + (_british ? BritishVowels : AmericanVowels) + _trap;
        var missing = new StringBuilder();
        foreach (var symbol in inventory)
        {
            if (_written.ContainsKey(symbol))
                continue;

            if (hasSymbol(symbol))
                _written[symbol] = symbol.ToString();
            else if (StandIns.TryGetValue(symbol, out var standIn) && standIn.All(c => hasSymbol(c)))
                _written[symbol] = standIn;
            else if (Optional.Contains(symbol))
                _written[symbol] = string.Empty;
            else
                missing.Append(missing.Length > 0 ? ", " : string.Empty).Append(CultureInfo.InvariantCulture, $"U+{(int)symbol:X4}");
        }

        if (missing.Length > 0)
            ThrowUnwritable(voiceName, missing);
    }

    [DoesNotReturn]
    private static void ThrowUnwritable(string voiceName, StringBuilder missing) =>
        throw new NotSupportedException(
            $"The voice '{voiceName}' declares English phonemes, but its phoneme_id_map has no id for {missing}. " +
            "The built-in English phonemizer cannot write for it. Set SynthesizerOptions.Phonemizer to an IPhonemizer " +
            "for this voice, or pass phonemes to SynthesizePhonemesAsync.");

    // The methods that hold IPA literals do not throw themselves: operational text is scanned method by method.
    private static string NotA(string kind, Phone phone) =>
        throw new InvalidOperationException($"{phone.Symbol} is not a {kind}.");

    /// <summary>Appends a pause, if the voice has an id for the mark that makes it.</summary>
    public void WritePause(char mark, StringBuilder ipa) => Append(ipa, mark);

    public void WriteSpace(StringBuilder ipa) => Append(ipa, ' ');

    /// <summary>Appends one word.</summary>
    /// <param name="word">The word.</param>
    /// <param name="beforeVowel">Whether the next word follows without a pause and starts with a vowel.</param>
    /// <param name="ipa">Where the symbols go.</param>
    public void WriteWord(Pronunciation word, bool beforeVowel, StringBuilder ipa)
    {
        var phones = word.Phones;
        var letters = _british ? VowelLetters.Align(word.Spelling, phones) : null;

        for (var i = 0; i < phones.Length; i++)
        {
            var phone = phones[i];
            Phone? next = i + 1 < phones.Length ? phones[i + 1] : null;

            if (!phone.IsVowel)
            {
                WriteConsonant(word, i, letters, ipa);
                continue;
            }

            // In the British accents an r after a vowel is part of how the vowel is said: it colours or lengthens the
            // vowel, and is itself heard only before another vowel.
            var rFollows = next?.Symbol == Arpa.R;
            var vowelAfterR = rFollows && (i + 2 < phones.Length ? phones[i + 2].IsVowel : beforeVowel);

            if (!word.Weak && phone.Stress == 1)
                Append(ipa, 'ˈ');
            else if (!word.Weak && phone.Stress == 2)
                Append(ipa, 'ˌ');

            var group = letters is null ? default : letters[i];
            Append(ipa, _british
                ? BritishVowel(phone, rFollows, vowelAfterR, word, group)
                : AmericanVowel(phone, rFollows, word));

            if (phone.Symbol == Arpa.ER && !rFollows)
            {
                // The dictionary's ER has its r in it; before a vowel the r is said as well ("hurry", "history").
                if (next?.IsVowel ?? (_british && beforeVowel))
                    Append(ipa, 'ɹ');
            }
            else if (rFollows && _british)
            {
                if (vowelAfterR)
                    Append(ipa, 'ɹ');
                i++;
            }
        }
    }

    private void WriteConsonant(Pronunciation word, int index, VowelLetters.Group[]? letters, StringBuilder ipa)
    {
        var phones = word.Phones;
        var phone = phones[index];
        Phone? previous = index > 0 ? phones[index - 1] : null;
        Phone? next = index + 1 < phones.Length ? phones[index + 1] : null;

        // American: a t between a vowel (or r) and an unstressed vowel is a flap ("water", "party").
        if (!_british && phone.Symbol == Arpa.T && next is { IsVowel: true, Stress: 0 }
            && previous is { } before && (before.IsVowel || before.Symbol == Arpa.R))
        {
            Append(ipa, 'ɾ');
            return;
        }

        Append(ipa, phone.Symbol switch
        {
            Arpa.B => "b", Arpa.CH => "tʃ", Arpa.D => "d", Arpa.DH => "ð", Arpa.F => "f", Arpa.G => "ɡ", Arpa.HH => "h",
            Arpa.JH => "dʒ", Arpa.K => "k", Arpa.L => "l", Arpa.M => "m", Arpa.N => "n", Arpa.NG => "ŋ", Arpa.P => "p",
            Arpa.R => "ɹ", Arpa.S => "s", Arpa.SH => "ʃ", Arpa.T => "t", Arpa.TH => "θ", Arpa.V => "v", Arpa.W => "w",
            Arpa.Y => "j", Arpa.Z => "z", Arpa.ZH => "ʒ",
            _ => NotA("consonant", phone)
        });

        // British: after t, d and n the vowel of «new» and «tune» keeps the glide American drops. The spelling tells it
        // from the vowel of «noon» and «two».
        if (_british && phone.Symbol is Arpa.T or Arpa.D or Arpa.N && next?.Symbol == Arpa.UW && letters is not null
            && IsSpelledWithGlide(word.Spelling, letters[index + 1]))
        {
            Append(ipa, 'j');
        }
    }

    private static bool IsSpelledWithGlide(string spelling, VowelLetters.Group group)
    {
        if (group.Length == 0)
            return false;

        var letters = spelling.AsSpan(group.Start, group.Length);
        return letters[0] == 'u' || letters is "eu"
            || (letters is "e" && group.Start + 1 < spelling.Length && spelling[group.Start + 1] == 'w');
    }

    private static string AmericanVowel(Phone phone, bool rFollows, Pronunciation word) => phone.Symbol switch
    {
        Arpa.AA => "ɑː",
        Arpa.AE => "æ",
        Arpa.AH => phone.IsStressed ? "ʌ" : word.Article ? "ɐ" : "ə",
        Arpa.AO => "ɔː",
        Arpa.AW => "aʊ",
        Arpa.AY => "aɪ",
        Arpa.EH => "ɛ",
        Arpa.ER => phone.IsStressed ? "ɜː" : "ɚ",
        Arpa.EY => "eɪ",
        Arpa.IH => "ɪ",
        // Before r the long vowels of «near» and «poor» are the short ones.
        Arpa.IY => rFollows ? "ɪ" : phone.IsStressed ? "iː" : "i",
        Arpa.OW => "oʊ",
        Arpa.OY => "ɔɪ",
        Arpa.UH => "ʊ",
        Arpa.UW => rFollows ? "ʊ" : "uː",
        _ => NotA("vowel", phone)
    };

    private string BritishVowel(Phone phone, bool rFollows, bool vowelAfterR, Pronunciation word, VowelLetters.Group group)
    {
        var spelling = word.Spelling;
        var letters = group.Length == 0 ? ReadOnlySpan<char>.Empty : spelling.AsSpan(group.Start, group.Length);
        var first = letters.IsEmpty ? '\0' : letters[0];
        var after = group.Length == 0 ? ReadOnlySpan<char>.Empty : spelling.AsSpan(group.Start + group.Length);

        switch (phone.Symbol)
        {
            case Arpa.AA when rFollows:
                // «sorry» has the vowel of «lot», «starry» and «car» the one of «father».
                return vowelAfterR && first == 'o' ? "ɒ" : "ɑː";
            case Arpa.AA:
                return IsLotVowel(spelling, group) ? "ɒ" : "ɑː";
            case Arpa.AE:
                return _trap;
            case Arpa.AH:
                return phone.IsStressed ? "ʌ" : word.Article ? "ɐ" : "ə";
            case Arpa.AO when rFollows:
                return vowelAfterR && first == 'o' && after.StartsWith("rr") ? "ɒ" : "ɔː";
            case Arpa.AO:
                // American has the vowel of «thought» in «dog» and «long»; British has the vowel of «lot» there.
                return letters is "o" ? "ɒ" : "ɔː";
            case Arpa.AW:
                return rFollows && !vowelAfterR ? "aʊə" : "aʊ";
            case Arpa.AY:
                return rFollows && !vowelAfterR ? "aɪə" : "aɪ";
            case Arpa.EH when rFollows:
                // «there» and «parent» have the vowel of «square»; «very» and «carry» keep a short vowel before the r.
                if (!vowelAfterR)
                    return "eə";
                if (first == 'a')
                    return after.StartsWith("rr") ? _trap : "eə";
                return "ɛ";
            case Arpa.EH:
                return "ɛ";
            case Arpa.ER:
                return phone.IsStressed ? "ɜː" : "ə";
            case Arpa.EY:
                return "eɪ";
            case Arpa.IH when rFollows:
                // «here» and «serious» have the vowel of «near»; «spirit» keeps the short one.
                return !vowelAfterR || first == 'e' ? "ɪə" : "ɪ";
            case Arpa.IH:
                return "ɪ";
            case Arpa.IY:
                return rFollows ? "ɪə" : phone.IsStressed ? "iː" : "i";
            case Arpa.OW:
                return "əʊ";
            case Arpa.OY:
                return rFollows && !vowelAfterR ? "ɔɪə" : "ɔɪ";
            case Arpa.UH:
                return rFollows ? "ʊə" : "ʊ";
            case Arpa.UW:
                return rFollows ? "ʊə" : "uː";
            default:
                return NotA("vowel", phone);
        }
    }

    // The dictionary's AA is two British vowels. Written with o ("hot"), with au ("because"), or with a after w or qu
    // ("want", "what", "quality"), it is the vowel of «lot»; written with a otherwise ("father", "spa") it is the vowel
    // of «father». Where the letters cannot be told, a word with an o in it is taken for the first kind.
    private static bool IsLotVowel(string spelling, VowelLetters.Group group)
    {
        if (group.Length == 0)
            return spelling.Contains('o');

        var letters = spelling.AsSpan(group.Start, group.Length);
        if (letters[0] != 'a')
            return true;
        if (letters.StartsWith("au"))
            return true;

        var before = spelling.AsSpan(0, group.Start);
        return before.EndsWith("w") || before.EndsWith("wh") || before.EndsWith("qu");
    }

    private void Append(StringBuilder ipa, string symbols)
    {
        foreach (var symbol in symbols)
            Append(ipa, symbol);
    }

    private void Append(StringBuilder ipa, char symbol)
    {
        if (!_written.TryGetValue(symbol, out var written))
            throw new InvalidOperationException($"U+{(int)symbol:X4} is not in the {_accent} inventory.");
        ipa.Append(written);
    }
}

/// <summary>
/// Lines a word's vowel phonemes up with the vowel letters it is written with, so that an accent rule can ask how a
/// vowel is spelled.
/// </summary>
internal static class VowelLetters
{
    /// <summary>A run of vowel letters in the spelling; <c>Length</c> 0 where the letters could not be told.</summary>
    public readonly record struct Group(int Start, int Length);

    /// <summary>
    /// Returns, for each phoneme, the letters of its vowel (an empty group for a consonant), or <see langword="null"/>
    /// when the word has a different number of vowel sounds than of vowel-letter runs — after allowing for a silent
    /// final <c>e</c> — and nothing can be said about which is which.
    /// </summary>
    public static Group[]? Align(string spelling, Phone[] phones)
    {
        var groups = new List<Group>(8);
        for (var i = 0; i < spelling.Length; i++)
        {
            if (!IsVowelLetter(spelling, i))
                continue;

            var start = i;
            while (i + 1 < spelling.Length && IsVowelLetter(spelling, i + 1))
                i++;
            groups.Add(new Group(start, i - start + 1));
        }

        var vowels = phones.Count(p => p.IsVowel);
        if (groups.Count == vowels + 1 && IsSilentE(spelling, groups[^1]))
            groups.RemoveAt(groups.Count - 1);
        if (groups.Count != vowels)
            return null;

        var aligned = new Group[phones.Length];
        var next = 0;
        for (var i = 0; i < phones.Length; i++)
        {
            if (phones[i].IsVowel)
                aligned[i] = groups[next++];
        }

        return aligned;
    }

    // y is a vowel letter except at the start of the word and before another vowel letter ("yes", "beyond").
    private static bool IsVowelLetter(string spelling, int index) => spelling[index] switch
    {
        'a' or 'e' or 'i' or 'o' or 'u' => true,
        'y' => index > 0 && !(index + 1 < spelling.Length && spelling[index + 1] is 'a' or 'e' or 'i' or 'o' or 'u'),
        _ => false
    };

    // The last run is a lone e that ends the word, or comes before a final s or d ("tune", "tunes", "tuned").
    private static bool IsSilentE(string spelling, Group last) =>
        last.Length == 1 && spelling[last.Start] == 'e' && last.Start > 0
        && (last.Start == spelling.Length - 1 || (last.Start == spelling.Length - 2 && spelling[^1] is 's' or 'd'));
}
