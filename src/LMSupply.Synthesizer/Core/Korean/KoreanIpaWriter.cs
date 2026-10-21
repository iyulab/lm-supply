using System.Diagnostics.CodeAnalysis;
using System.Globalization;
using System.Text;

namespace LMSupply.Synthesizer.Core.Korean;

/// <summary>
/// Writes pronounced syllables as the IPA a Korean voice trained on eSpeak NG phonemes takes, in symbols the voice has an
/// id for.
/// </summary>
/// <remarks>
/// The symbols were chosen by what the voice is heard to say (Whisper transcripts of sentences written each way), not by
/// running eSpeak NG: ㅏ is <c>ɐ</c>; a tense consonant is its lax one with a length mark (<c>kː</c>); a lax stop or
/// affricate is voiced between voiced sounds within a word (<c>ɡ d b dʑ</c>) and voiceless at its start; ㄹ is a tap
/// between vowels and <c>l</c> at the end of a syllable and after another ㄹ; no stress is marked.
/// </remarks>
internal sealed class KoreanIpaWriter
{
    // Marks a voice can do without: they are left out where it has no id for them.
    private const string Optional = ",;:.?!";

    // Every symbol the writer uses, but the optional marks.
    private const string Inventory = " ɐɛʌeouɯijw kɡtdpbsɕʑhnmŋlɾʰː";

    private readonly HashSet<char> _missingOptional = [];

    /// <param name="hasSymbol">Whether the voice has an id for a symbol (a Unicode code point).</param>
    /// <param name="voiceName">The voice, for the message when it cannot be written for.</param>
    /// <exception cref="NotSupportedException">The voice has no id for a symbol of Korean as this writer writes it.</exception>
    public KoreanIpaWriter(Func<int, bool> hasSymbol, string voiceName)
    {
        var missing = new StringBuilder();
        foreach (var symbol in Inventory.Where(symbol => !hasSymbol(symbol)))
            missing.Append(missing.Length > 0 ? ", " : string.Empty).Append(CultureInfo.InvariantCulture, $"U+{(int)symbol:X4}");
        if (missing.Length > 0)
            ThrowUnwritable(voiceName, missing);

        foreach (var mark in Optional.Where(mark => !hasSymbol(mark)))
            _missingOptional.Add(mark);
    }

    [DoesNotReturn]
    private static void ThrowUnwritable(string voiceName, StringBuilder missing) =>
        throw new NotSupportedException(
            $"The voice '{voiceName}' declares Korean phonemes, but its phoneme_id_map has no id for {missing}. " +
            "The built-in Korean phonemizer cannot write for it. Set SynthesizerOptions.Phonemizer to an IPhonemizer " +
            "for this voice, or pass phonemes to SynthesizePhonemesAsync.");

    /// <summary>Appends a pause, if the voice has an id for the mark that makes it.</summary>
    public void WritePause(char mark, StringBuilder ipa)
    {
        if (!_missingOptional.Contains(mark))
            ipa.Append(mark);
    }

    /// <summary>Appends one pronounced word.</summary>
    public static void WriteWord(IReadOnlyList<Syllable> word, StringBuilder ipa)
    {
        for (var i = 0; i < word.Count; i++)
        {
            // What comes before this syllable's initial: nothing at the start of the word, else the previous final.
            var before = i == 0 ? (char?)null : word[i - 1].Coda;
            var voiced = before is Syllable.None or 'ㄴ' or 'ㄹ' or 'ㅁ' or 'ㅇ';

            ipa.Append(Onset(word[i].Onset, voiced, afterL: before == 'ㄹ'));
            ipa.Append(Vowel(word[i].Vowel));
            ipa.Append(Coda(word[i].Coda));
        }
    }

    // The methods that hold IPA literals do not throw themselves: operational text is scanned method by method.
    private static string NotA(string kind, char jamo) =>
        throw new InvalidOperationException($"{jamo} is not a {kind}.");

    private static string Onset(char onset, bool voiced, bool afterL) => onset switch
    {
        'ㄱ' => voiced ? "ɡ" : "k",
        'ㄲ' => "kː",
        'ㄴ' => "n",
        'ㄷ' => voiced ? "d" : "t",
        'ㄸ' => "tː",
        'ㄹ' => afterL ? "l" : "ɾ",
        'ㅁ' => "m",
        'ㅂ' => voiced ? "b" : "p",
        'ㅃ' => "pː",
        'ㅅ' => "s",
        'ㅆ' => "sː",
        'ㅇ' => "",
        'ㅈ' => voiced ? "dʑ" : "tɕ",
        'ㅉ' => "tɕː",
        'ㅊ' => "tɕʰ",
        'ㅋ' => "kʰ",
        'ㅌ' => "tʰ",
        'ㅍ' => "pʰ",
        'ㅎ' => "h",
        _ => NotA("initial consonant", onset),
    };

    private static string Vowel(char vowel) => vowel switch
    {
        'ㅏ' => "ɐ",
        'ㅐ' => "ɛ",
        'ㅑ' => "jɐ",
        'ㅒ' => "jɛ",
        'ㅓ' => "ʌ",
        'ㅔ' => "e",
        'ㅕ' => "jʌ",
        'ㅖ' => "je",
        'ㅗ' => "o",
        'ㅘ' => "wɐ",
        'ㅙ' => "wɛ",
        'ㅚ' => "we",
        'ㅛ' => "jo",
        'ㅜ' => "u",
        'ㅝ' => "wʌ",
        'ㅞ' => "we",
        'ㅟ' => "wi",
        'ㅠ' => "ju",
        'ㅡ' => "ɯ",
        'ㅢ' => "ɯi",
        'ㅣ' => "i",
        _ => NotA("vowel", vowel),
    };

    // A pronounced syllable ends in one of seven sounds (or none).
    private static string Coda(char coda) => coda switch
    {
        Syllable.None => "",
        'ㄱ' => "k",
        'ㄴ' => "n",
        'ㄷ' => "t",
        'ㄹ' => "l",
        'ㅁ' => "m",
        'ㅂ' => "p",
        'ㅇ' => "ŋ",
        _ => NotA("final sound", coda),
    };
}
