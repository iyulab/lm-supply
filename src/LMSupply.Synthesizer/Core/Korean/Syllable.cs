namespace LMSupply.Synthesizer.Core.Korean;

/// <summary>
/// One Hangul syllable as its three parts, each a compatibility jamo (U+3131–U+3163): the initial consonant
/// (<c>ㅇ</c> when the syllable starts with its vowel), the vowel, and the final consonant (<see cref="None"/> when it
/// has none). The pronunciation rules rewrite these parts; the writer reads them.
/// </summary>
internal record struct Syllable(char Onset, char Vowel, char Coda)
{
    /// <summary>The final consonant of a syllable that has none.</summary>
    public const char None = '\0';

    /// <summary>The initial consonant of a syllable that starts with its vowel.</summary>
    public const char Silent = 'ㅇ';

    private const char First = '가';
    private const char Last = '힣';

    private const string Onsets = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ";
    private const string Vowels = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ";
    private const string Codas = "\0ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ";

    public static bool IsSyllable(char c) => c is >= First and <= Last;

    /// <summary>The parts of a precomposed syllable (U+AC00–U+D7A3).</summary>
    public static Syllable Of(char c)
    {
        if (!IsSyllable(c))
            throw new ArgumentOutOfRangeException(nameof(c), $"U+{(int)c:X4} is not a precomposed Hangul syllable.");

        var index = c - First;
        return new Syllable(Onsets[index / 588], Vowels[index % 588 / 28], Codas[index % 28]);
    }

    /// <summary>The syllables of a string of precomposed syllables.</summary>
    public static IEnumerable<Syllable> Of(string syllables) => syllables.Select(Of);

    /// <summary>The precomposed syllable these parts make: what a rewritten syllable reads as.</summary>
    public override string ToString() =>
        ((char)(First + (Onsets.IndexOf(Onset) * 21 + Vowels.IndexOf(Vowel)) * 28 + Codas.IndexOf(Coda))).ToString();
}
