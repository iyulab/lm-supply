using System.Text;

namespace LMSupply.Synthesizer.Core.English;

/// <summary>
/// The package's own phonemizer for English voices trained on eSpeak NG phonemes: text is read into words
/// (<see cref="EnglishText"/>), the words are pronounced from the CMU Pronouncing Dictionary and, where it has no entry,
/// by letter-to-sound rules (<see cref="EnglishLexicon"/>), and the pronunciations are written in the voice's accent and
/// symbols (<see cref="EspeakEnglishWriter"/>). It runs without eSpeak NG, without a network and without a model.
/// </summary>
/// <remarks>
/// It is a dictionary and rules, not a model of the language: a word with two readings gets one of them whatever the
/// sentence («read» is always the present tense), and stress is the word's own, with the function words unstressed.
/// </remarks>
internal sealed class EnglishPhonemizer : IPhonemizer
{
    private static readonly Phone[] TheBeforeVowel = Arpabet.Parse("DH IY0");
    private static readonly Phone[] ToBeforeVowel = Arpabet.Parse("T UH0");
    private static readonly Phone[] ToBeforePause = Arpabet.Parse("T UW1");

    private readonly EspeakEnglishWriter _writer;
    private readonly Lazy<EnglishLexicon> _lexicon;

    /// <exception cref="NotSupportedException">The voice has no id for a phoneme of the accent.</exception>
    public EnglishPhonemizer(EnglishAccent accent, Func<int, bool> hasSymbol, string voiceName, PronunciationDictionary? dictionary = null)
    {
        _writer = new EspeakEnglishWriter(accent, hasSymbol, voiceName);

        // The shipped dictionary is read when the first text arrives, not when the voice loads.
        _lexicon = new Lazy<EnglishLexicon>(() =>
            new EnglishLexicon(dictionary ?? PronunciationDictionary.Default, british: accent != EnglishAccent.American));
    }

    /// <summary>
    /// The accent a voice's phonemes are in, from the phonemizer voice its config names, or from its language where the
    /// config names none. <see langword="false"/> for a voice that is not English, or is in an accent this phonemizer
    /// does not write.
    /// </summary>
    public static bool TryGetAccent(PhonemeScheme scheme, out EnglishAccent accent)
    {
        accent = default;
        if (!string.Equals(scheme.Type, VoiceConfig.EspeakType, StringComparison.OrdinalIgnoreCase))
            return false;

        EnglishAccent? found = (scheme.Voice ?? scheme.Language)?.ToLowerInvariant() switch
        {
            "en-us" => EnglishAccent.American,
            "en" or "en-gb" => EnglishAccent.British,
            "en-gb-x-rp" => EnglishAccent.ReceivedPronunciation,
            _ => null
        };

        accent = found.GetValueOrDefault();
        return found is not null;
    }

    public ValueTask<IReadOnlyList<string>> PhonemizeAsync(string text, PhonemeScheme scheme, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(text);
        cancellationToken.ThrowIfCancellationRequested();
        return new(Phonemize(text));
    }

    /// <summary>One string of phonemes per sentence of <paramref name="text"/>.</summary>
    public IReadOnlyList<string> Phonemize(string text)
    {
        var lexicon = _lexicon.Value;
        var sentences = new List<string>();
        var words = new List<Pronunciation>();
        var ipa = new StringBuilder();

        foreach (var sentence in EnglishText.Read(text))
        {
            ipa.Clear();
            for (var i = 0; i < sentence.Count; i++)
            {
                if (sentence[i].Kind == SpokenKind.Pause)
                {
                    _writer.WritePause(sentence[i].Text[0], ipa);
                    continue;
                }

                // A run of words up to the next pause is said in one breath: a word can lean on the one after it.
                words.Clear();
                for (; i < sentence.Count && sentence[i].Kind != SpokenKind.Pause; i++)
                    lexicon.Pronounce(sentence[i], words);
                i--;

                for (var w = 0; w < words.Count; w++)
                {
                    var beforeVowel = w + 1 < words.Count && words[w + 1].Phones is [{ IsVowel: true }, ..];
                    if (ipa.Length > 0)
                        _writer.WriteSpace(ipa);
                    _writer.WriteWord(InContext(words[w], beforeVowel, last: w == words.Count - 1), beforeVowel, ipa);
                }
            }

            if (ipa.Length > 0)
                sentences.Add(ipa.ToString());
        }

        return sentences;
    }

    // «the» and «to» change with what follows them.
    private static Pronunciation InContext(Pronunciation word, bool beforeVowel, bool last)
    {
        if (!word.Weak)
            return word;

        return word.Spelling switch
        {
            "the" when beforeVowel => word with { Phones = TheBeforeVowel },
            "to" when beforeVowel => word with { Phones = ToBeforeVowel },
            "to" when last => word with { Phones = ToBeforePause },
            _ => word
        };
    }
}
