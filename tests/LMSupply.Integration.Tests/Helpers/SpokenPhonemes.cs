using LMSupply.Synthesizer;

namespace LMSupply.Integration.Tests.Helpers;

/// <summary>
/// Hand-written phonemes for the English voices, in the form eSpeak NG writes them (IPA with stress and length marks,
/// one string per sentence). A test that must not depend on the built-in English phonemizer passes these, directly or
/// through <see cref="Phonemizer"/>.
/// </summary>
internal static class SpokenPhonemes
{
    public const string WeatherText = "The weather is beautiful today.";
    public const string FoxText = "The quick brown fox jumps over the lazy dog.";
    public const string HelloText = "Hello, this is a test.";

    public const string Weather = "ðə wˈɛðəɹ ɪz bjˈuːtifəl tədˈeɪ.";
    public const string Fox = "ðə kwˈɪk bɹˈaʊn fˈɒks dʒˈʌmps ˌəʊvə ðə lˈeɪzi dˈɒɡ.";
    public const string Hello = "həlˈəʊ, ðɪs ɪz ɐ tˈɛst.";

    /// <summary>A phonemizer that knows the sentences above and nothing else, and records what it was asked.</summary>
    public sealed class Phonemizer : IPhonemizer
    {
        private static readonly Dictionary<string, string> Known = new(StringComparer.Ordinal)
        {
            [WeatherText] = Weather,
            [FoxText] = Fox,
            [HelloText] = Hello,
        };

        public List<PhonemeScheme> Schemes { get; } = [];

        public ValueTask<IReadOnlyList<string>> PhonemizeAsync(string text, PhonemeScheme scheme, CancellationToken cancellationToken = default)
        {
            Schemes.Add(scheme);

            // One sentence of text is one sentence of phonemes: the text is split after each full stop.
            var sentences = text.Split(". ", StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
                .Select(s => s.EndsWith('.') ? s : s + ".")
                .Select(s => Known.TryGetValue(s, out var phonemes)
                    ? phonemes
                    : throw new NotSupportedException($"The test phonemizer has no phonemes for '{s}'."))
                .ToList();
            return new(sentences);
        }
    }
}
