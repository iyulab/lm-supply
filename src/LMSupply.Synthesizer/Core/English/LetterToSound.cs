namespace LMSupply.Synthesizer.Core.English;

/// <summary>
/// Pronounces a word the dictionary does not have, from its spelling: the letter-to-sound rules give the phonemes, and a
/// few regularities of English give the stress, which the rules do not.
/// </summary>
internal static partial class LetterToSound
{
    // A vowel the rules produced, before the stress is assigned. The rules' schwa is unstressed from the start.
    private const byte Unassigned = 3;

    // Rules of our own, tried before the report's for the same letter: the report reads a word-final i as in «it», which
    // words borrowed into English do not say (taxi, sushi, emoji — the dictionary has that i as «ee» in nearly every
    // word that ends in a consonant and i), and a final ai as in «rain» (bonsai, samurai). Before a final a it reduces
    // the vowel the stress of such a word falls on (corona, Kagawa). A gh it keeps silent after any vowel is a g after
    // a consonant (Afghan) and between a single vowel and the next (Callaghan, Allegheny, spaghetti). The ough it reads
    // as in «rough» is that only there and after t and n (tough, enough); after d it is «dough», after c «cough», and otherwise
    // the ough of «bough» (Clough, Brougham).
    private static readonly Rule[] OwnRules =
    [
        new("^", "I", " ", "IY"),
        new("", "AI", " ", "AY"),
        new("", "A", "WA ", "AA"),
        new("", "ON", "A ", "OW N"),
        new("^", "GH", "", "G"),
        new("^A", "GH", "#", "G"),
        new("^E", "GH", "#", "G"),
        new("^O", "GH", "#", "G"),
        new("", "OUGHT", "", "AO T"),
        new(" R", "OUGH", "", "AH F"),
        new("T", "OUGH", "", "AH F"),
        new("N", "OUGH", "", "AH F"),
        new("D", "OUGH", "", "OW"),
        new("C", "OUGH", "", "AO F"),
        new("", "OUGH", "", "AW"),
    ];

    // After the rules it is grouped into: a static field is initialized in the order written.
    private static readonly Rule[][] RulesByLetter = GroupByLetter();

    // Endings that fix where the stress falls, counted in vowels from the end of the word.
    private static readonly (string Ending, int VowelFromEnd)[] StressByEnding =
    [
        ("ical", 3), ("ica", 3), ("ity", 3), ("ify", 3), ("ional", 3), ("ian", 3), ("ial", 3), ("ious", 3), ("eous", 3),
        ("ium", 3), ("ia", 3), ("io", 3),
        ("tion", 2), ("sion", 2), ("cian", 2), ("ic", 2), ("ics", 2), ("ish", 2),
        ("ee", 1), ("eer", 1), ("ese", 1), ("ette", 1), ("oon", 1), ("ique", 1), ("esque", 1),
    ];

    // Prefixes a two-syllable word is not stressed on, the stress going to the root (become, unpaid, exceed, dismiss).
    // One that ends in a vowel is a prefix only before consonants that can begin a syllable: before others the vowel is
    // the word's own, closed and stressed (beckon, belfry, dentist, record — beck-, bel-, den-, rec-).
    private static readonly string[] UnstressedPrefixes =
        ["be", "de", "re", "un", "dis", "mis", "ex", "en", "im", "ab", "af", "ap", "at", "ob", "sup", "trans"];

    // The consonant clusters a syllable can begin with, after a prefix that ends in a vowel (bespeak, destroy, refrain).
    private static readonly string[] SyllableOnsets =
    [
        "scr", "spl", "spr", "str", "squ", "shr", "thr",
        "bl", "br", "ch", "cl", "cr", "dr", "fl", "fr", "gl", "gr", "pl", "pr", "qu", "sc", "sh", "sk", "sl", "sm", "sn",
        "sp", "st", "sw", "th", "tr", "tw", "wh", "wr",
    ];

    // Endings that leave the stress where the word without them has it (nation, nations; visit, visiting), longest first,
    // with the vowels each one says. An -es is one only after a hissing sound (watches, boxes), and says its vowel; an
    // -ed says one only after t or d (wanted, not walked).
    private static readonly (string Suffix, int Vowels)[] StressNeutralSuffixes =
    [
        ("ments", 1), ("ment", 1), ("nesses", 2), ("ness", 1), ("less", 1), ("fully", 2), ("ful", 1), ("ings", 1),
        ("ing", 1), ("ers", 1), ("er", 1), ("ly", 1), ("ed", 1), ("es", 1), ("s", 0),
    ];

    /// <summary>
    /// Pronounces <paramref name="word"/>: lower-case letters <c>a</c>–<c>z</c>, and apostrophes, which are silent.
    /// </summary>
    public static Phone[] Pronounce(ReadOnlySpan<char> word)
    {
        // The rules see the word in capitals, with a space on each side for its edges.
        Span<char> text = word.Length <= 62 ? stackalloc char[word.Length + 2] : new char[word.Length + 2];
        text[0] = ' ';
        word.ToUpperInvariant(text[1..]);
        text[^1] = ' ';

        var phones = new List<Phone>(word.Length);
        var index = 1;
        while (index < text.Length - 1)
        {
            var letter = text[index];
            var next = index + 1;
            if (letter is >= 'A' and <= 'Z')
            {
                foreach (var rule in RulesByLetter[letter - 'A'])
                {
                    var end = index + rule.Match.Length;
                    if (end > text.Length - 1
                        || !text[index..end].SequenceEqual(rule.Match)
                        || !MatchesLeft(rule.Left, text[..index])
                        || !MatchesRight(rule.Right, text[end..]))
                    {
                        continue;
                    }

                    phones.AddRange(rule.Phones);
                    next = end;
                    break;
                }
            }

            index = next;
        }

        // A doubled consonant letter is one sound (bassetti, bobbi): the report's rules read each letter of the pair.
        for (var i = phones.Count - 1; i > 0; i--)
        {
            if (!phones[i].IsVowel && phones[i] == phones[i - 1])
                phones.RemoveAt(i);
        }

        var result = phones.ToArray();
        AssignStress(result, word);
        return result;
    }

    private static bool IsVowelLetter(char c) => c is 'A' or 'E' or 'I' or 'O' or 'U';

    private static bool IsConsonantLetter(char c) => c is >= 'A' and <= 'Z' && !IsVowelLetter(c);

    private static bool IsVoicedConsonantLetter(char c) => c is 'B' or 'D' or 'V' or 'G' or 'J' or 'L' or 'M' or 'N' or 'R' or 'W' or 'Z';

    private static bool IsFrontVowelLetter(char c) => c is 'E' or 'I' or 'Y';

    // The left context is matched from its last symbol backwards, against the text that ends just before the match.
    private static bool MatchesLeft(string pattern, ReadOnlySpan<char> text)
    {
        var at = text.Length - 1;
        for (var p = pattern.Length - 1; p >= 0; p--)
        {
            var symbol = pattern[p];
            var current = at >= 0 ? text[at] : '\0';
            switch (symbol)
            {
                case '#':
                    if (!IsVowelLetter(current))
                        return false;
                    while (at >= 0 && IsVowelLetter(text[at]))
                        at--;
                    break;
                case ':':
                    while (at >= 0 && IsConsonantLetter(text[at]))
                        at--;
                    break;
                case '^':
                    if (!IsConsonantLetter(current))
                        return false;
                    at--;
                    break;
                case '.':
                    if (!IsVoicedConsonantLetter(current))
                        return false;
                    at--;
                    break;
                case '+':
                    if (!IsFrontVowelLetter(current))
                        return false;
                    at--;
                    break;
                default:
                    if (symbol != current)
                        return false;
                    at--;
                    break;
            }
        }

        return true;
    }

    // The right context is matched forwards, against the text that starts just after the match.
    private static bool MatchesRight(string pattern, ReadOnlySpan<char> text)
    {
        var at = 0;
        foreach (var symbol in pattern)
        {
            var current = at < text.Length ? text[at] : '\0';
            switch (symbol)
            {
                case '#':
                    if (!IsVowelLetter(current))
                        return false;
                    while (at < text.Length && IsVowelLetter(text[at]))
                        at++;
                    break;
                case ':':
                    while (at < text.Length && IsConsonantLetter(text[at]))
                        at++;
                    break;
                case '^':
                    if (!IsConsonantLetter(current))
                        return false;
                    at++;
                    break;
                case '.':
                    if (!IsVoicedConsonantLetter(current))
                        return false;
                    at++;
                    break;
                case '+':
                    if (!IsFrontVowelLetter(current))
                        return false;
                    at++;
                    break;
                case '%':
                    // A suffix: ER, E, ES, ED, ELY or ING.
                    if (current == 'E')
                    {
                        at++;
                        var after = at < text.Length ? text[at] : '\0';
                        if (after == 'L')
                        {
                            if (at + 1 < text.Length && text[at + 1] == 'Y')
                                at += 2;
                        }
                        else if (after is 'R' or 'S' or 'D')
                        {
                            at++;
                        }
                    }
                    else if (text[at..].StartsWith("ING"))
                    {
                        at += 3;
                    }
                    else
                    {
                        return false;
                    }

                    break;
                default:
                    if (symbol != current)
                        return false;
                    at++;
                    break;
            }
        }

        return true;
    }

    /// <summary>
    /// Gives the word one primary stress. The rules' schwa never takes it. An ending that fixes the stress decides
    /// (<c>-tion</c>, <c>-ic</c>, <c>-ity</c>, <c>-ee</c>, ...); otherwise it falls on the first syllable, on the second of
    /// a two-syllable word that starts with a prefix such as <c>be-</c>, <c>re-</c>, <c>un-</c>, <c>dis-</c> or <c>ex-</c>
    /// (<see cref="UnstressedPrefixes"/>), and on the third from the end of a word of four syllables or more. A word of
    /// three syllables or more that ends in a consonant and <c>a</c>, <c>i</c> or <c>o</c> is stressed on the vowel
    /// before the last. Endings that do not move the stress (<c>-s</c>, <c>-ed</c>,
    /// <c>-ing</c>, <c>-er</c>, <c>-ly</c>, <c>-ness</c>, <c>-ment</c>, ...) are set aside first: the rules apply to
    /// what is left, and the ending's vowels are never stressed.
    /// </summary>
    private static void AssignStress(Phone[] phones, ReadOnlySpan<char> word)
    {
        var vowels = new List<int>(8);
        for (var i = 0; i < phones.Length; i++)
        {
            if (phones[i].IsVowel)
                vowels.Add(i);
        }

        if (vowels.Count == 0)
            return;

        var (stem, suffixVowels) = Stem(word, vowels.Count);
        var stressed = PickStressedVowel(phones, vowels[..(vowels.Count - suffixVowels)], stem);
        foreach (var i in vowels)
            phones[i] = i == stressed ? phones[i] with { Stress = 1 } : Unstressed(phones, i, word[0]);
    }

    // An unstressed vowel as the dictionary marks it. A short a, e or o inside the word is a schwa (Attenborough,
    // Ellison, Madison: «ə», not the «æ», «ɛ», «ɑ» the rules give); one that begins the word is a schwa for a (abound),
    // the i of «it» for e (eclectic), and an i there keeps a secondary stress (iconic). An i before another vowel is
    // the i of «see» (accordion), and a diphthong is not reduced and keeps a secondary stress (Monday, Holloway).
    private static Phone Unstressed(Phone[] phones, int at, char firstLetter)
    {
        var vowel = phones[at];
        if (at == 0)
        {
            switch (firstLetter, vowel.Symbol)
            {
                case ('a', Arpa.AE):
                    return new Phone(Arpa.AH, 0);
                case ('e', Arpa.EH or Arpa.IY):
                    return new Phone(Arpa.IH, 0);
                case ('i', Arpa.IH):
                    return vowel with { Stress = 2 };
            }
        }

        return vowel.Symbol switch
        {
            Arpa.AE or Arpa.EH or Arpa.AA when at > 0 => new Phone(Arpa.AH, 0),
            Arpa.IH when at + 1 < phones.Length && phones[at + 1].IsVowel => new Phone(Arpa.IY, 0),
            Arpa.EY or Arpa.AY or Arpa.OY or Arpa.AW => vowel with { Stress = 2 },
            _ => vowel with { Stress = 0 },
        };
    }

    private static int PickStressedVowel(Phone[] phones, List<int> vowels, ReadOnlySpan<char> word)
    {
        var count = vowels.Count;
        var choice = 0;

        var byEnding = false;
        foreach (var (ending, vowelFromEnd) in StressByEnding)
        {
            if (word.EndsWith(ending) && count >= vowelFromEnd && word.Length > ending.Length + 1)
            {
                choice = count - vowelFromEnd;
                byEnding = true;
                break;
            }
        }

        if (!byEnding && count >= 3 && EndsInConsonantAndOpenVowel(word))
        {
            // Words borrowed with a final a, i or o keep the stress of their language, on the vowel before it (agenda,
            // spaghetti, avocado).
            choice = count - 2;
            byEnding = true;
        }

        if (!byEnding)
        {
            if (count == 2 && StartsWithUnstressedPrefix(word))
                choice = 1;
            else if (count >= 4)
                choice = count - 3;
        }

        // A schwa is not stressed: move to the nearest vowel that can be, later ones first.
        for (var distance = 0; distance < count; distance++)
        {
            foreach (var candidate in (ReadOnlySpan<int>)[choice + distance, choice - distance])
            {
                if (candidate >= 0 && candidate < count && phones[vowels[candidate]].Stress == Unassigned)
                    return vowels[candidate];
            }
        }

        // Every vowel is a schwa: the first one carries the word.
        return vowels[0];
    }

    // The word without the stress-neutral endings it ends in — up to three (care-less-ness), each leaving at least three
    // letters and a vowel — and how many vowels they say. A stem whose y became i before the ending is given its y back
    // (accompanied, accompany), so the endings that fix the stress see the word they belong to.
    private static (string Stem, int SuffixVowels) Stem(ReadOnlySpan<char> word, int vowels)
    {
        var length = word.Length;
        var suffixVowels = 0;
        var startsWithE = false;
        for (var stripped = 0; stripped < 3; stripped++)
        {
            var found = false;
            foreach (var (suffix, said) in StressNeutralSuffixes)
            {
                var rest = word[..length];
                if (rest.Length < suffix.Length + 3 || !rest.EndsWith(suffix) || !IsSuffixHere(rest, suffix))
                    continue;
                var saying = suffix == "ed" && rest[^3] is not ('t' or 'd') ? 0 : said;
                if (vowels - suffixVowels - saying < 1)
                    continue;
                length -= suffix.Length;
                suffixVowels += saying;
                startsWithE = suffix[0] == 'e';
                found = true;
                break;
            }

            if (!found)
                break;
        }

        var stem = word[..length].ToString();
        if (startsWithE && stem.Length > 3 && stem[^1] == 'i' && IsConsonantLetter(char.ToUpperInvariant(stem[^2])))
            stem = string.Concat(stem.AsSpan(0, stem.Length - 1), "y");
        return (stem, suffixVowels);
    }

    private static bool IsSuffixHere(ReadOnlySpan<char> word, string suffix)
    {
        var before = word[^(suffix.Length + 1)];
        return suffix switch
        {
            // A hissing sound before -es: the e is the ending's (boxes, watches); otherwise it belongs to the word.
            "es" => before is 's' or 'x' or 'z' || (before == 'h' && word[^4] is 'c' or 's'),
            // Not the s of -ss, -us, -is (glass, census, tennis).
            "s" => before is not ('s' or 'u' or 'i'),
            _ => true,
        };
    }

    private static bool EndsInConsonantAndOpenVowel(ReadOnlySpan<char> word) =>
        word.Length >= 3 && (word[^1] is 'a' or 'i' or 'o') && IsConsonantLetter(char.ToUpperInvariant(word[^2]));

    private static bool StartsWithUnstressedPrefix(ReadOnlySpan<char> word)
    {
        foreach (var prefix in UnstressedPrefixes)
        {
            if (word.Length > prefix.Length + 2 && word.StartsWith(prefix) && BeginsTheRoot(word[prefix.Length..], prefix))
                return true;
        }

        return false;
    }

    // Whether the letters after a prefix can begin the root: a consonant, and after a prefix that ends in a vowel a single
    // one or a cluster a syllable begins with. An x of ex- runs into the vowel after it (exact, exist).
    private static bool BeginsTheRoot(ReadOnlySpan<char> rest, string prefix)
    {
        if (!IsConsonantLetter(char.ToUpperInvariant(rest[0])))
            return prefix == "ex";
        if (!IsVowelLetter(char.ToUpperInvariant(prefix[^1])))
            return rest[0] != 'h' || prefix == "ex";
        if (IsVowelLetter(char.ToUpperInvariant(rest[1])) || rest[1] == 'y')
            return true;
        foreach (var onset in SyllableOnsets)
        {
            if (rest.StartsWith(onset) && rest.Length > onset.Length && IsVowelLetter(char.ToUpperInvariant(rest[onset.Length])))
                return true;
        }

        return false;
    }

    private static Rule[][] GroupByLetter()
    {
        var groups = new List<Rule>[26];
        for (var i = 0; i < groups.Length; i++)
            groups[i] = [];
        foreach (var rule in OwnRules.Concat(CreateRules()))
            groups[rule.Match[0] - 'A'].Add(rule);
        return [.. groups.Select(g => g.ToArray())];
    }

    private sealed class Rule
    {
        public Rule(string left, string match, string right, string phones)
        {
            Left = left;
            Match = match;
            Right = right;
            Phones = [.. phones.Split(' ', StringSplitOptions.RemoveEmptyEntries).Select(ParsePhone)];
        }

        public string Left { get; }
        public string Match { get; }
        public string Right { get; }
        public Phone[] Phones { get; }

        private static Phone ParsePhone(string token)
        {
            if (token == "AX")
                return new Phone(Arpa.AH, 0);
            if (!Arpabet.TryParse(token, out var phone))
                throw new FormatException($"'{token}' is not a phoneme of the letter-to-sound rules.");
            return phone.IsVowel ? phone with { Stress = Unassigned } : phone;
        }
    }
}
