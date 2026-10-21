using AwesomeAssertions;
using LMSupply.Synthesizer.Core.English;

namespace LMSupply.Synthesizer.Tests;

/// <summary>
/// The letter-to-sound rules read against the shipped dictionary: for the words it has, the dictionary is the answer the
/// rules should come close to, so a change to the rules is measured on a hundred thousand words, not on a few chosen ones.
/// </summary>
public class LetterToSoundAgreementTests
{
    private static readonly Lazy<Dictionary<string, List<Phone[]>>> Dictionary = new(Load);

    [Fact]
    public void WordsEndingInAConsonantAndI_EndInEe_AndAreStressedLikeTheDictionary()
    {
        var words = EndingInAConsonantAnd('i');

        var finalVowel = Share(words, (rules, readings) => readings.Any(r => LastVowel(r) == LastVowel(rules)));

        words.Should().HaveCountGreaterThan(2500);
        finalVowel.Should().BeGreaterThan(0.9);
        StressAgreement(words).Should().BeGreaterThan(0.9);
    }

    [Theory]
    [InlineData('a')]
    [InlineData('o')]
    public void LongWordsEndingInAConsonantAndAOrO_AreStressedLikeTheDictionary(char last)
    {
        var words = EndingInAConsonantAnd(last).Where(entry => VowelCount(LetterToSound.Pronounce(entry.Key)) >= 3).ToList();

        words.Should().HaveCountGreaterThan(2000);
        StressAgreement(words).Should().BeGreaterThan(0.9);
    }

    [Fact]
    public void ADoubledConsonantLetter_IsOneSound()
    {
        var phones = LetterToSound.Pronounce("bassetti");

        phones.Zip(phones.Skip(1)).Should().NotContain(pair => !pair.First.IsVowel && pair.First == pair.Second);
        Dictionary.Value["bassetti"].Should().Contain(r => r.Count(p => p.Symbol == Arpa.S) == phones.Count(p => p.Symbol == Arpa.S));
    }

    [Theory]
    [InlineData("zorblation", "zorblations")]
    [InlineData("pellington", "pellingtons")]
    [InlineData("glorbify", "glorbified")]
    [InlineData("brindlemoor", "brindlemoorings")]
    public void AStressNeutralEnding_LeavesTheStressWhereTheWordWithoutItHasIt(string word, string withEnding)
    {
        StressedVowel(LetterToSound.Pronounce(withEnding)).Should().Be(StressedVowel(LetterToSound.Pronounce(word)));
    }

    [Fact]
    public void AFinalAi_IsSaidAsInEye()
    {
        LetterToSound.Pronounce("zorbai")[^1].Symbol.Should().Be(Arpa.AY);
    }

    [Fact]
    public void AcrossTheDictionary_TheRulesSayAQuarterOfTheWordsExactly_AndStressMostOfThemLikeIt()
    {
        var exact = Share([.. Dictionary.Value], (rules, readings) => readings.Any(r => r.SequenceEqual(rules)));

        exact.Should().BeGreaterThan(0.25);
        StressAgreement([.. Dictionary.Value]).Should().BeGreaterThan(0.85);
    }

    private static List<KeyValuePair<string, List<Phone[]>>> EndingInAConsonantAnd(char last) =>
        [.. Dictionary.Value.Where(entry => entry.Key.Length >= 3 && entry.Key[^1] == last && !IsVowel(entry.Key[^2]))];

    // Of the words the rules give as many vowels as one of the dictionary's readings, the share stressed on the same one.
    private static double StressAgreement(List<KeyValuePair<string, List<Phone[]>>> words) =>
        Share(
            [.. words.Where(entry => entry.Value.Any(r => VowelCount(r) == VowelCount(LetterToSound.Pronounce(entry.Key))))],
            (rules, readings) => readings.Any(r => VowelCount(r) == VowelCount(rules) && StressedVowel(r) == StressedVowel(rules)));

    private static double Share(
        List<KeyValuePair<string, List<Phone[]>>> words, Func<Phone[], List<Phone[]>, bool> agrees) =>
        (double)words.Count(entry => agrees(LetterToSound.Pronounce(entry.Key), entry.Value)) / words.Count;

    private static bool IsVowel(char c) => c is 'a' or 'e' or 'i' or 'o' or 'u';

    private static int VowelCount(Phone[] phones) => phones.Count(p => p.IsVowel);

    private static Arpa? LastVowel(Phone[] phones) => phones.LastOrDefault(p => p.IsVowel) is { IsVowel: true } last ? last.Symbol : null;

    private static int StressedVowel(Phone[] phones) => phones.Where(p => p.IsVowel).ToList().FindIndex(p => p.Stress == 1);

    // Every reading of every word of plain letters, alternatives («word(2)») under the word itself.
    private static Dictionary<string, List<Phone[]>> Load()
    {
        var words = new Dictionary<string, List<Phone[]>>();
        using var reader = new StreamReader(PronunciationDictionary.OpenShipped());
        while (reader.ReadLine() is { } line)
        {
            var space = line.IndexOf(' ');
            if (space < 0)
                continue;
            var word = line[..space];
            if (word.IndexOf('(') is var variant and > 0)
                word = word[..variant];
            if (!word.All(c => c is >= 'a' and <= 'z'))
                continue;

            if (!words.TryGetValue(word, out var readings))
                words[word] = readings = [];
            readings.Add(Arpabet.Parse(line[(space + 1)..].Split('#')[0].Trim()));
        }

        return words;
    }
}
