namespace LMSupply.Synthesizer.Core.Korean;

/// <summary>
/// Turns a written word into the syllables it is said as, by the rules of the standard pronunciation (표준 발음법) that
/// follow from the spelling alone: a final consonant carried over to a following vowel, the seven final sounds and the
/// clusters reduced to one, aspiration with ㅎ, palatalization of ㄷ and ㅌ before 이, tensing after a stop, and the
/// nasal and liquid assimilations.
/// </summary>
/// <remarks>
/// The rules apply inside a word, not across the space between words. What depends on the word's structure rather than
/// its spelling is not done: tensing after a verb stem in ㄴ or ㅁ (안고), after the future ending ㄹ (할 수), and in
/// Sino-Korean compounds (갈등); ㄴ inserted between the parts of a compound (솜이불); the exceptions to ㄴ + ㄹ (의견란).
/// </remarks>
internal static class KoreanPronunciation
{
    private const char N = Syllable.None;

    /// <summary>Rewrites <paramref name="word"/> in place into its pronounced syllables.</summary>
    public static void Apply(List<Syllable> word)
    {
        for (var i = 0; i + 1 < word.Count; i++)
            Join(word, i);

        for (var i = 0; i < word.Count; i++)
            word[i] = word[i] with { Coda = FinalSound(word[i]) };

        for (var i = 0; i + 1 < word.Count; i++)
            Assimilate(word, i);

        for (var i = 0; i < word.Count; i++)
            word[i] = word[i] with { Vowel = SaidVowel(word[i], initial: i == 0) };
    }

    // The final consonant of syllable i meets the initial of i + 1: ㅎ on either side, then a vowel that takes the final
    // consonant over.
    private static void Join(List<Syllable> word, int i)
    {
        var (here, next) = (word[i], word[i + 1]);

        if (here.Coda is 'ㅎ' or 'ㄶ' or 'ㅀ')
        {
            var rest = here.Coda switch { 'ㄶ' => 'ㄴ', 'ㅀ' => 'ㄹ', _ => N };
            (here, next) = next.Onset switch
            {
                'ㄱ' or 'ㄷ' or 'ㅈ' => (here with { Coda = rest }, next with { Onset = Aspirated(next.Onset) }),
                'ㅅ' => (here with { Coda = rest }, next with { Onset = 'ㅆ' }),
                'ㄴ' => (here with { Coda = rest == N ? 'ㄴ' : rest }, next),
                _ => (here with { Coda = rest }, next),
            };
        }
        else if (next.Onset == 'ㅎ' && Stop(here.Coda) is var (rest, stop) && stop != N)
        {
            here = here with { Coda = rest };
            next = next with { Onset = Aspirated(stop) };
            if (next is { Onset: 'ㅌ', Vowel: 'ㅣ' })
                next = next with { Onset = 'ㅊ' };
        }

        // A cluster that only a verb stem ends in tenses the lax consonant after it (앉고, 삶다, 넓게, 핥지); of ㄺ the
        // ㄹ stays before ㄱ and the ㄱ goes with it (맑게).
        if (here.Coda == 'ㄺ' && next.Onset == 'ㄱ')
            (here, next) = (here with { Coda = 'ㄹ' }, next with { Onset = 'ㄲ' });
        else if (here.Coda is 'ㄵ' or 'ㄻ' or 'ㄼ' or 'ㄾ')
            next = next with { Onset = Tense(next.Onset) };

        if (next.Onset == Syllable.Silent && here.Coda is not (N or 'ㅇ'))
        {
            var (kept, moved) = Split(here.Coda);
            if (next.Vowel == 'ㅣ')
                moved = moved switch { 'ㄷ' => 'ㅈ', 'ㅌ' => 'ㅊ', _ => moved };
            (here, next) = (here with { Coda = kept }, next with { Onset = moved });
        }

        (word[i], word[i + 1]) = (here, next);
    }

    // A final that ends in a stop, split into what stays and the stop ㅎ aspirates (ㄱ, ㄷ, ㅂ or ㅈ); N for the stop
    // when the final is not one.
    private static (char Kept, char Stop) Stop(char coda) => coda switch
    {
        'ㄱ' or 'ㄲ' or 'ㅋ' or 'ㄳ' => (N, 'ㄱ'),
        'ㄷ' or 'ㅅ' or 'ㅆ' or 'ㅈ' or 'ㅊ' or 'ㅌ' => (N, 'ㄷ'),
        'ㅂ' or 'ㅍ' or 'ㅄ' => (N, 'ㅂ'),
        'ㄺ' => ('ㄹ', 'ㄱ'),
        'ㄼ' => ('ㄹ', 'ㅂ'),
        'ㄵ' => ('ㄴ', 'ㅈ'),
        _ => (coda, N),
    };

    // A final carried over to a vowel: of a cluster the second consonant goes, and ㅅ after a stop is said tense.
    private static (char Kept, char Moved) Split(char coda) => coda switch
    {
        'ㄳ' => ('ㄱ', 'ㅆ'),
        'ㄵ' => ('ㄴ', 'ㅈ'),
        'ㄺ' => ('ㄹ', 'ㄱ'),
        'ㄻ' => ('ㄹ', 'ㅁ'),
        'ㄼ' => ('ㄹ', 'ㅂ'),
        'ㄽ' => ('ㄹ', 'ㅅ'),
        'ㄾ' => ('ㄹ', 'ㅌ'),
        'ㄿ' => ('ㄹ', 'ㅍ'),
        'ㅄ' => ('ㅂ', 'ㅆ'),
        _ => (N, coda),
    };

    private static char Tense(char lax) => lax switch
    {
        'ㄱ' => 'ㄲ',
        'ㄷ' => 'ㄸ',
        'ㅂ' => 'ㅃ',
        'ㅅ' => 'ㅆ',
        'ㅈ' => 'ㅉ',
        _ => lax,
    };

    private static char Aspirated(char lax) => lax switch
    {
        'ㄱ' => 'ㅋ',
        'ㄷ' => 'ㅌ',
        'ㅂ' => 'ㅍ',
        'ㅈ' => 'ㅊ',
        _ => lax,
    };

    // A syllable's final as one of the seven sounds a syllable can end in; of ㄼ the ㅂ is said in 밟.
    private static char FinalSound(Syllable syllable) => syllable.Coda switch
    {
        'ㄱ' or 'ㄲ' or 'ㅋ' or 'ㄳ' or 'ㄺ' => 'ㄱ',
        'ㄴ' or 'ㄵ' or 'ㄶ' => 'ㄴ',
        'ㄷ' or 'ㅅ' or 'ㅆ' or 'ㅈ' or 'ㅊ' or 'ㅌ' or 'ㅎ' => 'ㄷ',
        'ㄼ' => syllable is { Onset: 'ㅂ', Vowel: 'ㅏ' } ? 'ㅂ' : 'ㄹ',
        'ㄹ' or 'ㄽ' or 'ㄾ' or 'ㅀ' => 'ㄹ',
        'ㅁ' or 'ㄻ' => 'ㅁ',
        'ㅂ' or 'ㅍ' or 'ㅄ' or 'ㄿ' => 'ㅂ',
        var coda => coda,
    };

    // After the finals are reduced: a stop tenses the lax consonant after it, ㄹ after another consonant is said ㄴ,
    // a stop before a nasal becomes a nasal, and ㄴ next to ㄹ becomes ㄹ.
    private static void Assimilate(List<Syllable> word, int i)
    {
        var (here, next) = (word[i], word[i + 1]);
        if (here.Coda is 'ㄱ' or 'ㄷ' or 'ㅂ')
            next = next with { Onset = Tense(next.Onset) };

        if (next.Onset == 'ㄹ' && here.Coda is 'ㄱ' or 'ㄷ' or 'ㅂ' or 'ㅁ' or 'ㅇ')
            next = next with { Onset = 'ㄴ' };

        if (next.Onset is 'ㄴ' or 'ㅁ')
            here = here with { Coda = here.Coda switch { 'ㄱ' => 'ㅇ', 'ㄷ' => 'ㄴ', 'ㅂ' => 'ㅁ', var c => c } };

        if (here.Coda == 'ㄴ' && next.Onset == 'ㄹ')
            here = here with { Coda = 'ㄹ' };
        else if (here.Coda == 'ㄹ' && next.Onset == 'ㄴ')
            next = next with { Onset = 'ㄹ' };

        (word[i], word[i + 1]) = (here, next);
    }

    // The vowels said otherwise than written: ㅢ after a consonant or past a word's first syllable is ㅣ, ㅖ after a
    // consonant other than ㄹ is ㅔ, and after ㅈ, ㅉ and ㅊ the y-vowels lose their y.
    private static char SaidVowel(Syllable syllable, bool initial) => (syllable.Onset, syllable.Vowel) switch
    {
        ('ㅇ', 'ㅢ') => initial ? 'ㅢ' : 'ㅣ',
        (_, 'ㅢ') => 'ㅣ',
        ('ㅇ' or 'ㄹ', 'ㅖ') => 'ㅖ',
        (_, 'ㅖ') => 'ㅔ',
        ('ㅈ' or 'ㅉ' or 'ㅊ', var v) => v switch { 'ㅑ' => 'ㅏ', 'ㅒ' => 'ㅐ', 'ㅕ' => 'ㅓ', 'ㅛ' => 'ㅗ', 'ㅠ' => 'ㅜ', _ => v },
        (_, var v) => v,
    };
}
