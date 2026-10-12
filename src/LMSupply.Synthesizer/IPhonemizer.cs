namespace LMSupply.Synthesizer;

/// <summary>
/// Turns text into the phonemes a voice was trained on. A voice declares which phonemes those are
/// (<see cref="ISynthesizerModel.PhonemeScheme"/>); the package produces them itself only for the <c>text</c> type, so a
/// voice of any other type speaks text through the implementation set in <see cref="SynthesizerOptions.Phonemizer"/>.
/// </summary>
public interface IPhonemizer
{
    /// <summary>
    /// Converts <paramref name="text"/> to phonemes for a voice with the given <paramref name="scheme"/>.
    /// </summary>
    /// <param name="text">The text to speak.</param>
    /// <param name="scheme">What the voice declares about its phonemes: their type, and the voice and language they are for.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>
    /// One string per sentence, in order. In a string every Unicode code point is one phoneme, a space included, and the
    /// punctuation that ends a clause or the sentence (<c>, ; : . ? !</c>) is a phoneme of that sentence. The voice
    /// synthesizes each string on its own, so a sentence boundary is where intonation resets and where streaming yields a
    /// chunk. Phonemes the voice has no id for are skipped (see <see cref="SynthesisResult.UnknownPhonemes"/>).
    /// </returns>
    /// <exception cref="NotSupportedException">The implementation cannot produce phonemes of this scheme.</exception>
    ValueTask<IReadOnlyList<string>> PhonemizeAsync(
        string text,
        PhonemeScheme scheme,
        CancellationToken cancellationToken = default);
}

/// <summary>
/// The phonemes a voice was trained on, as the voice's own config declares them. Phonemes passed to the voice, by an
/// <see cref="IPhonemizer"/> or by the caller, have to be of this scheme.
/// </summary>
public sealed record PhonemeScheme
{
    /// <summary>
    /// The voice's <c>phoneme_type</c>: <c>"espeak"</c> (IPA as eSpeak NG writes it, with its stress and length marks) or
    /// <c>"text"</c> (the characters of the text itself). A config that does not name a type means <c>"espeak"</c>.
    /// </summary>
    public required string Type { get; init; }

    /// <summary>
    /// The phonemizer voice the config names (its <c>espeak.voice</c>, for example <c>"en-us"</c> or <c>"en-gb-x-rp"</c>),
    /// passed through as declared, or <see langword="null"/> when the config names none.
    /// </summary>
    public string? Voice { get; init; }

    /// <summary>
    /// The voice's language as a BCP 47 tag (for example <c>"en-US"</c>): the config's <c>language.code</c>, or the
    /// catalog entry's language when the config has none.
    /// </summary>
    public string? Language { get; init; }
}
