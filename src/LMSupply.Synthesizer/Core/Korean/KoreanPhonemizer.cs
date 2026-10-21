using System.Text;

namespace LMSupply.Synthesizer.Core.Korean;

/// <summary>
/// The package's own phonemizer for Korean voices trained on eSpeak NG phonemes: text is read into words
/// (<see cref="KoreanText"/>), each word is turned into the syllables it is said as (<see cref="KoreanPronunciation"/>),
/// and those are written in the voice's symbols (<see cref="KoreanIpaWriter"/>). It runs without eSpeak NG, without a
/// network and without a model.
/// </summary>
/// <remarks>
/// It is rules over the spelling, not a model of the language: what the pronunciation of a word owes to its structure
/// (tensing after a verb stem or a future ending, ㄴ inserted in a compound) and which numeral a counter takes are not
/// known to it.
/// </remarks>
internal sealed class KoreanPhonemizer : IPhonemizer
{
    private readonly KoreanIpaWriter _writer;

    /// <exception cref="NotSupportedException">The voice has no id for a symbol of Korean as the writer writes it.</exception>
    public KoreanPhonemizer(Func<int, bool> hasSymbol, string voiceName) =>
        _writer = new KoreanIpaWriter(hasSymbol, voiceName);

    /// <summary>
    /// Whether a voice's phonemes are Korean: eSpeak NG phonemes for the phonemizer voice <c>ko</c>, or, where the config
    /// names none, for a voice whose language is Korean.
    /// </summary>
    public static bool IsKorean(PhonemeScheme scheme)
    {
        if (!string.Equals(scheme.Type, VoiceConfig.EspeakType, StringComparison.OrdinalIgnoreCase))
            return false;

        var tag = scheme.Voice ?? scheme.Language;
        return tag is not null
            && (tag.Equals("ko", StringComparison.OrdinalIgnoreCase) || tag.StartsWith("ko-", StringComparison.OrdinalIgnoreCase));
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
        var sentences = new List<string>();
        var ipa = new StringBuilder();

        foreach (var sentence in KoreanText.Read(text))
        {
            ipa.Clear();
            foreach (var token in sentence)
            {
                if (token.IsPause)
                {
                    _writer.WritePause(token.Pause, ipa);
                    continue;
                }

                if (ipa.Length > 0)
                    ipa.Append(' ');
                KoreanPronunciation.Apply(token.Word!);
                KoreanIpaWriter.WriteWord(token.Word!, ipa);
            }

            if (ipa.Length > 0)
                sentences.Add(ipa.ToString());
        }

        return sentences;
    }
}
