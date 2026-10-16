namespace LMSupply.Synthesizer.Core.English;

/// <summary>
/// One word as it is to be said: its phonemes, the letters it was written with (the accent rules read them), and how it
/// sits in the sentence.
/// </summary>
/// <param name="Phones">The phonemes, with the dictionary's stress.</param>
/// <param name="Spelling">The word in lower-case letters.</param>
/// <param name="Weak">A function word: said without a stress mark.</param>
/// <param name="Article">The article «a» or «an», whose vowel is written as the reduced <c>ɐ</c>.</param>
internal sealed record Pronunciation(Phone[] Phones, string Spelling, bool Weak = false, bool Article = false);

/// <summary>
/// Finds the pronunciation of a word: the dictionary's first-listed one, a small table of the function words and frequent
/// homographs where that is not the form wanted, regular endings on a word the dictionary has, a compound of two words
/// it has, the names of the letters for an initialism, and the letter-to-sound rules for everything else.
/// </summary>
internal sealed class EnglishLexicon
{
    // Function words are said without stress. Where a pronunciation is given it replaces the dictionary's first-listed
    // one (which, for some of these, is the emphatic or the less usual form); "|" separates the American form from the
    // British one where they differ in more than the accent rules derive.
    private static readonly Dictionary<string, string?> FunctionWords = new(StringComparer.Ordinal)
    {
        ["a"] = "AH0", ["an"] = "AH0 N", ["the"] = "DH AH0", ["to"] = "T AH0",
        ["of"] = "AH1 V|AA1 V", ["from"] = "F R AH1 M|F R AA1 M", ["was"] = "W AH1 Z|W AA1 Z", ["been"] = "B IH1 N|B IY1 N",
        ["and"] = "AE1 N D", ["for"] = "F AO1 R", ["as"] = "AE1 Z", ["that"] = "DH AE1 T", ["is"] = "IH1 Z",
        ["are"] = "AA1 R", ["with"] = "W IH1 DH", ["your"] = "Y AO1 R", ["our"] = "AW1 ER0", ["i'm"] = "AY1 M",
        ["in"] = null, ["on"] = null, ["at"] = null, ["by"] = null, ["or"] = null, ["but"] = null, ["if"] = null,
        ["than"] = null, ["am"] = null, ["were"] = null, ["be"] = null, ["has"] = null, ["have"] = null, ["had"] = null,
        ["do"] = null, ["does"] = null, ["did"] = null, ["will"] = null, ["would"] = null, ["shall"] = null,
        ["should"] = null, ["can"] = null, ["could"] = null, ["may"] = null, ["might"] = null, ["must"] = null,
        ["i"] = null, ["you"] = null, ["he"] = null, ["she"] = null, ["it"] = null, ["we"] = null, ["they"] = null,
        ["me"] = null, ["him"] = null, ["her"] = null, ["us"] = null, ["them"] = null, ["my"] = null, ["his"] = null,
        ["its"] = null, ["their"] = null, ["this"] = null, ["into"] = null,
    };

    // Words whose first-listed pronunciation in the dictionary is the less frequent reading, or one the British accent
    // rules cannot reach from the American form.
    private static readonly Dictionary<string, string> Preferred = new(StringComparer.Ordinal)
    {
        ["read"] = "R IY1 D", ["live"] = "L IH1 V", ["lead"] = "L IY1 D", ["wind"] = "W IH1 N D",
        ["what"] = "W AH1 T|W AA1 T", ["because"] = "B IH0 K AO1 Z|B IH0 K AA1 Z",
    };

    private static readonly string[] LetterNames =
    [
        "EY1", "B IY1", "S IY1", "D IY1", "IY1", "EH1 F", "JH IY1", "EY1 CH", "AY1", "JH EY1", "K EY1", "EH1 L", "EH1 M",
        "EH1 N", "OW1", "P IY1", "K Y UW1", "AA1 R", "EH1 S", "T IY1", "Y UW1", "V IY1", "D AH1 B AH0 L Y UW0", "EH1 K S",
        "W AY1", "Z IY1|Z EH1 D"
    ];

    // The shortest part of a compound the dictionary is asked for, and the longest initialism named letter by letter.
    private const int MinCompoundPart = 3;
    private const int MaxInitialismLength = 5;

    private static readonly System.Buffers.SearchValues<char> VowelLetters = System.Buffers.SearchValues.Create("aeiouy");

    private readonly PronunciationDictionary _dictionary;
    private readonly bool _british;

    public EnglishLexicon(PronunciationDictionary dictionary, bool british)
    {
        _dictionary = dictionary;
        _british = british;
    }

    /// <summary>Appends what <paramref name="spoken"/> is said as: one word, or one per letter of an initialism.</summary>
    public void Pronounce(Spoken spoken, List<Pronunciation> words)
    {
        var text = spoken.Text;
        if (spoken.Kind == SpokenKind.Letters && !IsReadAsWord(text))
        {
            SpellOut(text, words);
            return;
        }

        if (TryKnown(text, out var known))
        {
            words.Add(known);
            return;
        }

        // No vowel letter: nothing to read, so the letters are named ("km", "txt").
        if (text.AsSpan().IndexOfAny(VowelLetters) < 0)
        {
            SpellOut(text, words);
            return;
        }

        words.Add(new Pronunciation(LetterToSound.Pronounce(text), text));
    }

    // A token written in capitals is read as a word when the dictionary has it and it is longer than two letters
    // ("NASA"; the dictionary lists "FBI" as its letters), or when it is too long to be an initialism.
    private bool IsReadAsWord(string text) =>
        text.Length > MaxInitialismLength || (text.Length > 2 && _dictionary.TryGetValue(text, out _));

    private void SpellOut(string text, List<Pronunciation> words)
    {
        var letters = text.Where(char.IsAsciiLetterLower).ToArray();
        for (var i = 0; i < letters.Length; i++)
        {
            // The last letter carries the stress of the whole; the others have a secondary one.
            var phones = Parse(LetterNames[letters[i] - 'a']);
            if (i < letters.Length - 1)
            {
                for (var p = 0; p < phones.Length; p++)
                {
                    if (phones[p].Stress == 1)
                        phones[p] = phones[p] with { Stress = 2 };
                }
            }

            words.Add(new Pronunciation(phones, letters[i].ToString()));
        }
    }

    private bool TryKnown(string word, out Pronunciation pronunciation)
    {
        if (FunctionWords.TryGetValue(word, out var weakForm))
        {
            var phones = weakForm is not null ? Parse(weakForm) : _dictionary.TryGetValue(word, out var listed) ? listed : null;
            if (phones is not null)
            {
                pronunciation = new Pronunciation(phones, word, Weak: true, Article: word is "a" or "an");
                return true;
            }
        }

        if (TryListed(word, out var found) || TryInflected(word, out found) || TryCompound(word, out found))
        {
            pronunciation = new Pronunciation(found, word);
            return true;
        }

        pronunciation = null!;
        return false;
    }

    // The preferred reading, a letter's name, or the dictionary's entry.
    private bool TryListed(string word, out Phone[] phones)
    {
        if (Preferred.TryGetValue(word, out var preferred))
        {
            phones = Parse(preferred);
            return true;
        }

        if (word.Length == 1 && char.IsAsciiLetterLower(word[0]) && word != "i")
        {
            phones = Parse(LetterNames[word[0] - 'a']);
            return true;
        }

        return _dictionary.TryGetValue(word, out phones!);
    }

    /// <summary>
    /// A regular ending on a word the dictionary has: the possessive and the plural, <c>-ed</c>, <c>-ing</c> and
    /// <c>-ly</c>. The possessive is also added to a word read by the letter-to-sound rules.
    /// </summary>
    private bool TryInflected(string word, out Phone[] phones)
    {
        phones = [];
        if (word.EndsWith("'s", StringComparison.Ordinal) && word.Length > 2)
        {
            var owner = word[..^2];
            var stem = TryListed(owner, out var listed) ? listed : LetterToSound.Pronounce(owner);
            phones = WithSibilantEnding(stem);
            return true;
        }

        if (word.Length < 4)
            return false;

        if (word.EndsWith('s') && !word.EndsWith("ss", StringComparison.Ordinal))
        {
            if ((TryStem(word[..^1], out var stem) && !IsSibilant(stem[^1].Symbol))
                || (word.EndsWith("es", StringComparison.Ordinal) && TryStem(word[..^2], out stem) && IsSibilant(stem[^1].Symbol))
                || (word.EndsWith("ies", StringComparison.Ordinal) && TryStem(word[..^3] + "y", out stem)))
            {
                phones = WithSibilantEnding(stem);
                return true;
            }
        }

        if (word.EndsWith("ed", StringComparison.Ordinal))
        {
            if (TryStem(word[..^1], out var stem) || TryStem(word[..^2], out stem) || TryUndoubled(word[..^2], out stem)
                || (word.EndsWith("ied", StringComparison.Ordinal) && TryStem(word[..^3] + "y", out stem)))
            {
                var last = stem[^1].Symbol;
                phones = last is Arpa.T or Arpa.D ? [.. stem, new Phone(Arpa.IH), new Phone(Arpa.D)]
                    : IsVoiceless(last) ? [.. stem, new Phone(Arpa.T)]
                    : [.. stem, new Phone(Arpa.D)];
                return true;
            }
        }

        if (word.EndsWith("ing", StringComparison.Ordinal) && word.Length > 4)
        {
            if (TryStem(word[..^3] + "e", out var stem) || TryStem(word[..^3], out stem) || TryUndoubled(word[..^3], out stem))
            {
                phones = [.. stem, new Phone(Arpa.IH), new Phone(Arpa.NG)];
                return true;
            }
        }

        if (word.EndsWith("ly", StringComparison.Ordinal) && TryStem(word[..^2], out var adjective))
        {
            phones = [.. adjective, new Phone(Arpa.L), new Phone(Arpa.IY)];
            return true;
        }

        return false;
    }

    private bool TryStem(string stem, out Phone[] phones)
    {
        phones = [];
        return stem.Length >= 2 && _dictionary.TryGetValue(stem, out phones!);
    }

    // "stopped" is "stop" with its last consonant doubled.
    private bool TryUndoubled(string stem, out Phone[] phones)
    {
        phones = [];
        return stem.Length >= 3 && stem[^1] == stem[^2] && TryStem(stem[..^1], out phones);
    }

    // The plural and the possessive: a syllable after a sibilant, [s] after a voiceless consonant, [z] otherwise.
    private static Phone[] WithSibilantEnding(Phone[] stem)
    {
        var last = stem[^1].Symbol;
        return IsSibilant(last) ? [.. stem, new Phone(Arpa.IH), new Phone(Arpa.Z)]
            : IsVoiceless(last) ? [.. stem, new Phone(Arpa.S)]
            : [.. stem, new Phone(Arpa.Z)];
    }

    private static bool IsSibilant(Arpa symbol) => symbol is Arpa.S or Arpa.Z or Arpa.SH or Arpa.ZH or Arpa.CH or Arpa.JH;

    private static bool IsVoiceless(Arpa symbol) => symbol is Arpa.P or Arpa.T or Arpa.K or Arpa.F or Arpa.TH;

    /// <summary>
    /// Two dictionary words written as one ("smartphone"): the split whose shorter part is longest. The first part keeps
    /// its stress and the second part's becomes secondary.
    /// </summary>
    private bool TryCompound(string word, out Phone[] phones)
    {
        phones = [];
        var best = 0;
        for (var split = MinCompoundPart; split <= word.Length - MinCompoundPart; split++)
        {
            var shorter = Math.Min(split, word.Length - split);
            if (shorter <= best || !_dictionary.TryGetValue(word.AsSpan(0, split), out var head)
                || !_dictionary.TryGetValue(word.AsSpan(split), out var tail))
            {
                continue;
            }

            for (var i = 0; i < tail.Length; i++)
            {
                if (tail[i].Stress == 1)
                    tail[i] = tail[i] with { Stress = 2 };
            }

            phones = [.. head, .. tail];
            best = shorter;
        }

        return best > 0;
    }

    private Phone[] Parse(string forms)
    {
        var bar = forms.IndexOf('|');
        return Arpabet.Parse(bar < 0 ? forms : _british ? forms[(bar + 1)..] : forms[..bar]);
    }
}
