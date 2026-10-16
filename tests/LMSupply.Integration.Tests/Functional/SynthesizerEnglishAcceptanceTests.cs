using System.Globalization;
using LMSupply.Exceptions;
using LMSupply.Synthesizer;
using LMSupply.Transcriber;

namespace LMSupply.Integration.Tests.Functional;

/// <summary>
/// The built-in English phonemizer, judged by ear: plain text goes in, each English voice of the catalog speaks it with no
/// phonemizer set, Whisper transcribes the audio, and the word error rate against the text is the measure. A voice or the
/// transcriber that is not in the local model cache is skipped, not downloaded.
/// </summary>
[Trait("Category", "Integration")]
[Trait("Category", "LocalOnly")]
[Trait("Domain", "Synthesizer")]
public sealed class SynthesizerEnglishAcceptanceTests(ITestOutputHelper output)
{
    // Measured (Whisper "default", CPU), word error rate over all 120 words / over the 97 words of prose and figures,
    // lowest and highest of the runs made:
    //   default (espeak voice "en", British-style)  0.008-0.083 / 0.000-0.031 (4 runs)
    //   lessac ("en-us")                            0.033       / 0.000-0.010 (2 runs)
    //   fast ("en-us")                              0.017-0.025 / 0.000-0.010 (2 runs)
    //   quality ("en-us")                           0.025-0.067 / 0.000-0.021 (2 runs)
    //   british ("en-gb-x-rp")                      0.025-0.075 / 0.010       (2 runs)
    // Runs differ because the voice samples noise. Most errors are the out-of-dictionary names spelled another way
    // ("Pelington", "smart watch"). For comparison, the default voice given the American transcription instead of the
    // British-style one it was trained on measured 0.125 / 0.082. The bounds leave room for about twice the worst run.
    private const double MaxWordErrorRate = 0.15;
    private const double MaxInDictionaryWordErrorRate = 0.08;

    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    /// <summary>The sentences, by set: ordinary prose, figures, and words the pronunciation dictionary does not have.</summary>
    private static readonly (string Set, string Text)[] Sentences =
    [
        ("prose", "The weather is beautiful today."),
        ("prose", "The quick brown fox jumps over the lazy dog."),
        ("prose", "She sells fresh bread at the market every morning."),
        ("prose", "Please close the window before you leave the house."),
        ("prose", "We walked along the river and talked about our plans for the summer."),
        ("prose", "Reading a good book is one of the simplest pleasures in life."),
        ("numbers", "The train leaves at 7 and arrives 45 minutes later."),
        ("numbers", "There are 12 months and 365 days in a year."),
        ("numbers", "He paid $20 for 3 tickets."),
        ("numbers", "Prices rose by 15% in 2023."),
        ("out-of-dictionary", "My smartwatch can talk to the chatbot."),
        ("out-of-dictionary", "The livestream showed an emoji on every slide."),
        ("out-of-dictionary", "We drove from Pellington to Oakhurst and Fernvale."),
    ];

    public static TheoryData<string> Voices { get; } = ["default", "lessac", "fast", "quality", "british"];

    [Theory]
    [MemberData(nameof(Voices))]
    public async Task TextSpokenByTheVoice_WhisperHears(string alias)
    {
        await using var synthesizer = await LoadCachedAsync(
            () => LocalSynthesizer.LoadAsync(alias, new SynthesizerOptions { DisableAutoDownload = true }, cancellationToken: Ct));
        await using var transcriber = await LoadCachedAsync(
            () => LocalTranscriber.LoadAsync("default", new TranscriberOptions { DisableAutoDownload = true }, cancellationToken: Ct));

        var totals = new Dictionary<string, (int Errors, int Words)>();
        foreach (var (set, text) in Sentences)
        {
            var audio = await synthesizer.SynthesizeAsync(text, cancellationToken: Ct);
            audio.UnknownPhonemes.Should().BeEmpty("the built-in phonemizer writes only symbols the voice has");

            var heard = await transcriber.TranscribeAsync(audio.ToWavBytes(), new TranscribeOptions { Language = "en" }, Ct);
            var said = Words(text);
            var errors = EditDistance(said, Words(heard.Text));
            output.WriteLine($"{alias} [{set}] {errors}/{said.Length} said «{text}» heard «{heard.Text.Trim()}» phonemes «{string.Join(" | ", audio.Phonemes)}»");

            var (e, w) = totals.GetValueOrDefault(set);
            totals[set] = (e + errors, w + said.Length);
        }

        foreach (var (set, (errors, words)) in totals)
            output.WriteLine($"{alias} [{set}] word error rate {(double)errors / words:F3} ({errors}/{words})");

        var wer = Rate(totals.Values);
        var inDictionary = Rate(totals.Where(t => t.Key != "out-of-dictionary").Select(t => t.Value));
        output.WriteLine($"{alias} [all] word error rate {wer:F3}, prose and figures {inDictionary:F3}");
        Assert.True(wer <= MaxWordErrorRate, $"{alias}: word error rate {wer:F3}, bound {MaxWordErrorRate}");
        Assert.True(inDictionary <= MaxInDictionaryWordErrorRate,
            $"{alias}: word error rate on prose and figures {inDictionary:F3}, bound {MaxInDictionaryWordErrorRate}");

        static double Rate(IEnumerable<(int Errors, int Words)> sets) =>
            (double)sets.Sum(s => s.Errors) / sets.Sum(s => s.Words);
    }

    private static async Task<T> LoadCachedAsync<T>(Func<Task<T>> load)
    {
        try
        {
            return await load();
        }
        catch (ModelNotFoundException ex)
        {
            Assert.Skip($"Not in the local model cache: {ex.Message}");
            throw;
        }
    }

    // The words of a text, lower-cased, with what Whisper may write as figures or symbols read out by a converter of the
    // test's own (not the phonemizer's), so that «7» and «seven» are the same word on both sides.
    private static string[] Words(string text)
    {
        var words = new List<string>();
        foreach (var token in text.ToLowerInvariant().Replace("-", " ", StringComparison.Ordinal)
                     .Split(' ', StringSplitOptions.RemoveEmptyEntries))
        {
            var core = token.Trim('.', ',', '!', '?', ';', ':', '"', '«', '»');
            if (core.Length == 0)
                continue;

            var dollars = core.StartsWith('$');
            var percent = core.EndsWith('%');
            var digits = core.Trim('$', '%').Replace(",", "", StringComparison.Ordinal);
            if (digits.Length > 0 && digits.All(char.IsAsciiDigit))
            {
                var value = int.Parse(digits, CultureInfo.InvariantCulture);
                words.AddRange(digits.Length == 4 && value is >= 1100 and <= 2099 and not (>= 2000 and <= 2009)
                    ? [.. Number(value / 100), .. Number(value % 100)]
                    : Number(value));
                if (dollars) words.Add("dollars");
                if (percent) words.Add("percent");
                continue;
            }

            words.AddRange(new string([.. core.Select(c => char.IsLetterOrDigit(c) || c == '\'' ? c : ' ')])
                .Split(' ', StringSplitOptions.RemoveEmptyEntries));
        }

        return [.. words];
    }

    private static readonly string[] Ones =
    [
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
        "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"
    ];

    private static readonly string[] Tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"];

    private static IEnumerable<string> Number(int value)
    {
        if (value >= 1000)
        {
            foreach (var word in Number(value / 1000)) yield return word;
            yield return "thousand";
            value %= 1000;
            if (value == 0) yield break;
        }

        if (value >= 100)
        {
            yield return Ones[value / 100];
            yield return "hundred";
            value %= 100;
            if (value == 0) yield break;
        }

        if (value >= 20)
        {
            yield return Tens[value / 10];
            if (value % 10 != 0) yield return Ones[value % 10];
            yield break;
        }

        yield return Ones[value];
    }

    // Word-level edit distance: substitutions, insertions and deletions.
    private static int EditDistance(string[] said, string[] heard)
    {
        var distance = new int[said.Length + 1, heard.Length + 1];
        for (var i = 0; i <= said.Length; i++) distance[i, 0] = i;
        for (var j = 0; j <= heard.Length; j++) distance[0, j] = j;

        for (var i = 1; i <= said.Length; i++)
        {
            for (var j = 1; j <= heard.Length; j++)
            {
                var substitution = distance[i - 1, j - 1] + (said[i - 1] == heard[j - 1] ? 0 : 1);
                distance[i, j] = Math.Min(substitution, Math.Min(distance[i - 1, j] + 1, distance[i, j - 1] + 1));
            }
        }

        return distance[said.Length, heard.Length];
    }
}
