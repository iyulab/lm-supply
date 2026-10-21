using System.Security.Cryptography;
using AwesomeAssertions;
using LMSupply.Synthesizer.Core;
using LMSupply.Synthesizer.Core.English;
using LMSupply.Synthesizer.Models;

namespace LMSupply.Synthesizer.Tests;

/// <summary>
/// The built-in English phonemizer: text in, the phonemes of the voice's own accent and symbols out. The expected strings
/// are written out in full, so a changed mapping, a lost rule or a symbol the voice does not have fails on the exact text.
/// The voices are the catalog's own configs (Fixtures), resolved the way a loaded model resolves them.
/// </summary>
public class EnglishPhonemizerTests
{
    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    private static IReadOnlyList<string> Say(SynthesizerModelInfo voice, string text) =>
        Phonemizer(voice).Phonemize(text);

    private static EnglishPhonemizer Phonemizer(SynthesizerModelInfo voice) =>
        (EnglishPhonemizer)OnnxSynthesizerModel.ResolvePhonemizer(callers: null, VoiceFixture.Catalog(voice), voice.AliasName);

    // ── en-us: the dictionary's American, written directly ──────────────────────────────────────────────

    [Theory]
    [InlineData("The weather is beautiful today.", "ðə wˈɛðɚ ɪz bjˈuːɾəfəl tədˈeɪ.")]
    [InlineData("The car is far, and the water is hotter than ever.", "ðə kˈɑːɹ ɪz fˈɑːɹ, ænd ðə wˈɔːɾɚ ɪz hˈɑːɾɚ ðæn ˈɛvɚ.")]
    [InlineData("Four sorry tourists are here.", "fˈɔːɹ sˈɑːɹi tˈʊɹəsts ɑːɹ hˈɪɹ.")]
    // Not in the dictionary: two words it has, and letter-to-sound rules with a stress of their own.
    [InlineData("smartwatch", "smˈɑːɹtwˌɑːtʃ")]
    [InlineData("Zorblat", "zˈɔːɹblət")]
    // Figures: a year in pairs, dollars and cents, a time of day, an ordinal, separators, a decimal and a percentage.
    [InlineData("In 1999 she paid $3.50 for 21 apples at 4:30.",
        "ɪn nˈaɪntˈiːn nˈaɪnti nˈaɪn ʃiː pˈeɪd θɹˈiː dˈɑːlɚz ænd fˈɪfti sˈɛnts fɔːɹ twˈɛnti wˈʌn ˈæpəlz æt fˈɔːɹ θˈɜːdˌiː.")]
    [InlineData("The 3rd test took 2,500 people and 12.5% of 7:00 and 9:05.",
        "ðə θˈɜːd tˈɛst tˈʊk tˈuː θˈaʊzənd fˈaɪv hˈʌndɹəd pˈiːpəl ænd twˈɛlv pˈɔɪnt fˈaɪv pɚsˈɛnt ʌv sˈɛvən əklˈɑːk ænd nˈaɪn ˈoʊ fˈaɪv.")]
    [InlineData("-4 & 1990s", "mˈaɪnəs fˈɔːɹ ænd nˈaɪntˈiːn nˈaɪntiz")]
    // Abbreviations: titles that do not end the sentence, e.g., dotted letters, etc.
    [InlineData("Dr. Smith met Mr. Jones, e.g. at the U.S. office, etc.",
        "dˈɑːktɚ smˈɪθ mˈɛt mˈɪstɚ dʒˈoʊnz, fɔːɹ ɪɡzˈæmpəl æt ðə jˌuː ˈɛs ˈɔːfɪs, ˈɛt sˈɛɾɚɹə.")]
    // Punctuation: a sentence ends at . ? ! (a run of them is one), a clause pauses at , ; :
    [InlineData("Wait... really? Yes! No; maybe: fine.", "wˈeɪt. | ɹˈɪli? | jˈɛs! | nˈoʊ; mˈeɪbi: fˈaɪn.")]
    // Words in context: «the» and «to» before a vowel, «to» before a pause.
    [InlineData("The apple. To eat it, go to.", "ði ˈæpəl. | tʊ ˈiːt ɪt, ɡˈoʊ tuː.")]
    public void American(string text, string expected)
    {
        string.Join(" | ", Say(SynthesizerModels.EnUsAmy, text)).Should().Be(expected);
        string.Join(" | ", Say(SynthesizerModels.EnUsRyan, text)).Should().Be(expected, "both en-us voices have every symbol");
    }

    // ── en: the default voice, British-style ──────────────────────────────────────────────────────────

    [Theory]
    [InlineData("The weather is beautiful today.", "ðə wˈɛðəɹ ɪz bjˈuːtəfəl tədˈeɪ.")]
    // Non-rhotic: r is said only before a vowel, in the word or at the start of the next one.
    [InlineData("The car is far, and the water is hotter than ever.", "ðə kˈɑːɹ ɪz fˈɑː, and ðə wˈɔːtəɹ ɪz hˈɒtə ðan ˈɛvə.")]
    // Centring diphthongs before a dropped r, and the vowel of «lot» where American has the one of «father».
    [InlineData("Four sorry tourists are here.", "fˈɔː sˈɒɹi tˈʊəɹəsts ɑː hˈɪə.")]
    // The glide after t, d and n where the spelling has u or ew.
    [InlineData("Our new tune knew the duke.", "aʊə njˈuː tjˈuːn njˈuː ðə djˈuːk.")]
    [InlineData("Zorblat", "zˈɔːblət")]
    [InlineData("Dr. Smith met Mr. Jones, e.g. at the U.S. office, etc.",
        "dˈɒktə smˈɪθ mˈɛt mˈɪstə dʒˈəʊnz, fɔːɹ ɪɡzˈampəl at ðə jˌuː ˈɛs ˈɒfɪs, ˈɛt sˈɛtəɹə.")]
    public void British(string text, string expected)
    {
        string.Join(" | ", Say(SynthesizerModels.EnUsLjSpeech, text)).Should().Be(expected);
    }

    // ── en-gb-x-rp: the same conversion, with the vowel of «trap» written æ ──────────────────────────────

    [Theory]
    [InlineData("The car is far, and the water is hotter than ever.", "ðə kˈɑːɹ ɪz fˈɑː, ænd ðə wˈɔːtəɹ ɪz hˈɒtə ðæn ˈɛvə.")]
    [InlineData("Four sorry tourists are here.", "fˈɔː sˈɒɹi tˈʊəɹəsts ɑː hˈɪə.")]
    [InlineData("Zorblat", "zˈɔːblət")]
    public void ReceivedPronunciation(string text, string expected)
    {
        string.Join(" | ", Say(SynthesizerModels.EnGbSemaine, text)).Should().Be(expected);
    }

    // ── The voice's own symbols ───────────────────────────────────────────────────────────────────────

    // Every vowel and consonant of the dictionary, the weak forms, figures and every pause mark.
    private const string Coverage =
        "Father, the hot cat ate a pie; now boys sat by the boat. Put two foods: bird, bed, bit, beat? Yes! " +
        "Church judge, thing, measure, shoe, think, this, very zoo, young hope with law. " +
        "Our tour here, there and far: hurry, sorry, very carry! The 21st of 1,250 at 9:05, $3.50 or 12.5%.";

    public static TheoryData<string> EnglishVoices { get; } = [.. VoiceFixture.EnglishVoices.Select(v => v.AliasName)];

    [Theory]
    [MemberData(nameof(EnglishVoices))]
    public void EveryEnglishCatalogVoice_GetsOnlySymbolsItsOwnMapHas(string alias)
    {
        var voice = SynthesizerModels.All.Single(m => m.AliasName == alias);
        var config = VoiceFixture.Catalog(voice);

        var sentences = Phonemizer(voice).Phonemize(Coverage);

        sentences.Should().HaveCountGreaterThan(3);
        foreach (var sentence in sentences)
        {
            var encoded = config.Encoder.Encode(sentence);
            encoded.Unknown.Should().BeEmpty($"the voice '{alias}' has an id for every symbol of «{sentence}»");
            encoded.PhonemeCount.Should().Be(sentence.EnumerateRunes().Count());
        }
    }

    [Fact]
    public void TheAccent_IsTheOneTheVoicesConfigNames()
    {
        static EnglishAccent? Accent(string? voice, string? language)
        {
            var scheme = new PhonemeScheme { Type = "espeak", Voice = voice, Language = language };
            return EnglishPhonemizer.TryGetAccent(scheme, out var accent) ? accent : null;
        }

        Accent("en-us", "en-US").Should().Be(EnglishAccent.American);
        Accent("en", "en-US").Should().Be(EnglishAccent.British, "the default voice was transcribed by the British-style 'en'");
        Accent("en-gb-x-rp", "en-GB").Should().Be(EnglishAccent.ReceivedPronunciation);
        Accent(null, "en-US").Should().Be(EnglishAccent.American, "with no phonemizer voice named, the language decides");
        Accent("ko", "ko-KR").Should().BeNull();
        Accent("cmn", "zh-CN").Should().BeNull();
        EnglishPhonemizer.TryGetAccent(new PhonemeScheme { Type = "text", Language = "en-US" }, out _).Should().BeFalse();
    }

    // ── A symbol the voice has no id for ─────────────────────────────────────────────────────────────

    private static string SayWithout(string symbols, string text, EnglishAccent accent = EnglishAccent.American)
    {
        var voice = VoiceFixture.LjSpeech();
        var phonemizer = new EnglishPhonemizer(accent, c => voice.Encoder.Has(c) && !symbols.Contains((char)c), "test");
        return string.Join(" | ", phonemizer.Phonemize(text));
    }

    [Fact]
    public void AStandIn_IsWrittenForASymbolWithOne()
    {
        // The IPA g is written as the ASCII letter, the flap as t, the r-coloured schwa and the article's vowel as schwa.
        SayWithout("ɡ", "A big dog.").Should().Be("ɐ bˈɪg dˈɔːg.");
        SayWithout("ɾ", "The water.").Should().Be("ðə wˈɔːtɚ.");
        SayWithout("ɚ", "The water.").Should().Be("ðə wˈɔːɾə.");
        SayWithout("ɐ", "A dog.").Should().Be("ə dˈɔːɡ.");
    }

    [Fact]
    public void AMarkTheVoiceLacks_IsLeftOut()
    {
        SayWithout("ˈˌ", "Hello, my friend.").Should().Be("həloʊ, maɪ fɹɛnd.");
        SayWithout(",", "Hello, my friend.").Should().Be("həlˈoʊ maɪ fɹˈɛnd.");
        SayWithout("ː", "Far.", EnglishAccent.British).Should().Be("fˈɑ.");
    }

    [Fact]
    public void AVoiceWithoutAPhonemeOfTheAccent_IsRefused_NamingWhatItLacks()
    {
        var act = () => SayWithout("θ", "Thanks.");

        var message = act.Should().Throw<NotSupportedException>().Which.Message;
        message.Should().Contain("'test'").And.Contain("U+03B8");
        message.Should().Contain("SynthesizerOptions.Phonemizer").And.Contain("SynthesizePhonemesAsync");
        message.Should().MatchRegex(@"^[\x20-\x7E]+$", "operational text is ASCII");
    }

    [Fact]
    public void AnEnglishConfigWithAFewSymbols_IsRefusedWhenItsTextArrives()
    {
        var voice = VoiceFixture.Minimal("\"espeak\": {\"voice\": \"en-us\"}");

        var act = () => OnnxSynthesizerModel.ResolvePhonemizer(callers: null, voice, "tiny");

        act.Should().Throw<NotSupportedException>().WithMessage("*'tiny'*U+00F0*");
    }

    // ── Determinism, cancellation ───────────────────────────────────────────────────────────────────

    [Fact]
    public void TheSameText_GivesTheSamePhonemes_EveryTime()
    {
        var first = Say(SynthesizerModels.EnUsLjSpeech, Coverage);
        var again = Phonemizer(SynthesizerModels.EnUsLjSpeech);

        again.Phonemize(Coverage).Should().Equal(first);
        again.Phonemize(Coverage).Should().Equal(first);
    }

    [Fact]
    public async Task ACancelledRequest_IsNotPhonemized()
    {
        using var cancelled = new CancellationTokenSource();
        await cancelled.CancelAsync();
        var voice = VoiceFixture.LjSpeech();

        var act = async () => await Phonemizer(SynthesizerModels.EnUsLjSpeech).PhonemizeAsync("Hello.", voice.Scheme, cancelled.Token);

        await act.Should().ThrowAsync<OperationCanceledException>();
    }

    [Fact]
    public async Task TextWithNothingToSay_IsNoSentence()
    {
        var voice = VoiceFixture.LjSpeech();

        var sentences = await Phonemizer(SynthesizerModels.EnUsLjSpeech).PhonemizeAsync(" \"«»\" ... ", voice.Scheme, Ct);

        sentences.Should().BeEmpty();
    }
}

/// <summary>The pieces under the phonemizer: the shipped dictionary, numbers, the reader and the letter-to-sound rules.</summary>
public class EnglishPhonemizerPartsTests
{
    [Fact]
    public void TheShippedDictionary_IsTheCmuFileUnmodified()
    {
        using var text = PronunciationDictionary.OpenShipped();

        // cmudict.dict at cmusphinx/cmudict commit 74790861f652b15e4ac49015a90074ad62a27690 (Data/cmudict/README.md).
        Convert.ToHexStringLower(SHA256.HashData(text))
            .Should().Be("81917843c7f44ce2b094ac63873c2c7a4cf802040792c455ba3ca406891c3d22");
    }

    [Fact]
    public void TheShippedDictionary_KeepsEachWordsFirstPronunciation()
    {
        var dictionary = PronunciationDictionary.Default;

        dictionary.Count.Should().Be(126_052);
        dictionary.TryGetValue("weather", out var weather).Should().BeTrue();
        weather.Should().Equal(Arpabet.Parse("W EH1 DH ER0"));
        dictionary.TryGetValue("read", out var read).Should().BeTrue();
        read.Should().Equal(Arpabet.Parse("R EH1 D"), "the file lists «read» as R EH1 D first");
        dictionary.TryGetValue("zorblat", out _).Should().BeFalse();
        dictionary.TryGetValue("café", out _).Should().BeFalse("a word outside ASCII is not in the file");
    }

    [Fact]
    public void ADictionaryText_IsParsedSortedWithItsFirstPronunciationAndWithoutComments()
    {
        var dictionary = PronunciationDictionary.Parse("b B IY1\na EY1 # a comment\na(2) AH0\nc S IY1\n"u8);

        dictionary.Count.Should().Be(3);
        dictionary.TryGetValue("a", out var a).Should().BeTrue();
        a.Should().Equal(Arpabet.Parse("EY1"));
        dictionary.TryGetValue("c", out _).Should().BeTrue();
    }

    [Theory]
    [InlineData(0, "zero")]
    [InlineData(21, "twenty one")]
    [InlineData(100, "one hundred")]
    [InlineData(2_500, "two thousand five hundred")]
    [InlineData(1_000_005, "one million five")]
    [InlineData(999_999_999_999_999, "nine hundred ninety nine trillion nine hundred ninety nine billion nine hundred ninety nine million nine hundred ninety nine thousand nine hundred ninety nine")]
    public void Cardinals(long value, string expected)
    {
        var words = new List<string>();
        NumberWords.AppendCardinal(value, words);
        string.Join(" ", words).Should().Be(expected);
    }

    [Theory]
    [InlineData(1999, "nineteen ninety nine")]
    [InlineData(1905, "nineteen oh five")]
    [InlineData(1900, "nineteen hundred")]
    [InlineData(2005, "two thousand five")]
    [InlineData(2023, "twenty twenty three")]
    public void Years(int value, string expected)
    {
        var words = new List<string>();
        NumberWords.AppendInPairs(value, words);
        string.Join(" ", words).Should().Be(expected);
    }

    [Theory]
    [InlineData(1, "first")]
    [InlineData(12, "twelfth")]
    [InlineData(20, "twentieth")]
    [InlineData(23, "twenty third")]
    [InlineData(100, "one hundredth")]
    public void Ordinals(int value, string expected)
    {
        var words = new List<string>();
        NumberWords.AppendCardinal(value, words);
        NumberWords.MakeOrdinal(words);
        string.Join(" ", words).Should().Be(expected);
    }

    [Theory]
    // A title's full stop does not end the sentence; the last full stop of a dotted abbreviation ends it at the end.
    [InlineData("Dr. Smith arrived. He left.", "doctor smith arrived . | he left .")]
    [InlineData("Mr Smith left at 10:00!", "mister smith left at ten o'clock !")]
    [InlineData("It costs $1 or $5 million.", "it costs one dollar or five million dollars .")]
    [InlineData("Call No. 5 on St. James Street.", "call number five on saint james street .")]
    [InlineData("Version 3.5.1 is at example.com", "version three point five point one is at example dot com")]
    [InlineData("The 1990s, the 21st, the FBI.", "the nineteen nineties , the twenty first , the fbi .")]
    [InlineData("Café — naïve.", "cafe , naive .")]
    [InlineData("Line one\nline two\n\nNew paragraph", "line one line two | new paragraph")]
    public void TheReader_SaysWhatIsWritten(string text, string expected)
    {
        var sentences = EnglishText.Read(text).Select(s => string.Join(" ", s.Select(spoken => spoken.Text)));
        string.Join(" | ", sentences).Should().Be(expected);
    }

    [Theory]
    [InlineData("zorblat", "Z AO1 R B L AH0 T")]
    [InlineData("pellington", "P EH1 L IH0 NG T AH0 N")]
    public void AWordNotInTheDictionary_IsReadByTheLetterToSoundRules(string word, string expected)
    {
        LetterToSound.Pronounce(word).Should().Equal(Arpabet.Parse(expected));
    }

    [Theory]
    [InlineData("quindex")]
    [InlineData("brindlemoor")]
    [InlineData("emoji")]
    [InlineData("glorptastic")]
    public void ALetterToSoundWord_HasOnePrimaryStress(string word)
    {
        var phones = LetterToSound.Pronounce(word);

        phones.Should().NotBeEmpty();
        phones.Count(p => p.Stress == 1).Should().Be(1);
        phones.Where(p => !p.IsVowel).Should().OnlyContain(p => p.Stress == 0);
    }
}
