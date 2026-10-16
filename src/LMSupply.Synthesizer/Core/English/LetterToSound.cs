namespace LMSupply.Synthesizer.Core.English;

/// <summary>
/// Pronounces a word the dictionary does not have, from its spelling: the letter-to-sound rules give the phonemes, and a
/// few regularities of English give the stress, which the rules do not.
/// </summary>
internal static partial class LetterToSound
{
    // A vowel the rules produced, before the stress is assigned. The rules' schwa is unstressed from the start.
    private const byte Unassigned = 3;

    private static readonly Rule[][] RulesByLetter = GroupByLetter();

    // Endings that fix where the stress falls, counted in vowels from the end of the word.
    private static readonly (string Ending, int VowelFromEnd)[] StressByEnding =
    [
        ("ical", 3), ("ity", 3), ("ify", 3), ("ional", 3), ("ian", 3), ("ial", 3), ("ious", 3), ("eous", 3),
        ("ium", 3), ("ia", 3), ("io", 3),
        ("tion", 2), ("sion", 2), ("cian", 2), ("ic", 2), ("ics", 2), ("ish", 2),
        ("ee", 1), ("eer", 1), ("ese", 1), ("ette", 1), ("oon", 1), ("ique", 1), ("esque", 1),
    ];

    private static readonly string[] UnstressedPrefixes = ["be", "de", "re"];

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
    /// a two-syllable word that starts with <c>be-</c>, <c>de-</c> or <c>re-</c>, and on the third from the end of a word
    /// of four syllables or more.
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

        var stressed = PickStressedVowel(phones, vowels, word);
        foreach (var i in vowels)
            phones[i] = phones[i] with { Stress = (byte)(i == stressed ? 1 : 0) };
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

    private static bool StartsWithUnstressedPrefix(ReadOnlySpan<char> word)
    {
        foreach (var prefix in UnstressedPrefixes)
        {
            if (word.Length > prefix.Length + 2 && word.StartsWith(prefix) && !IsVowelLetter(char.ToUpperInvariant(word[prefix.Length])))
                return true;
        }

        return false;
    }

    private static Rule[][] GroupByLetter()
    {
        var groups = new List<Rule>[26];
        for (var i = 0; i < groups.Length; i++)
            groups[i] = [];
        foreach (var rule in CreateRules())
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
