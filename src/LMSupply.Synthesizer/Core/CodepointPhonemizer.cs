using System.Text;

namespace LMSupply.Synthesizer.Core;

/// <summary>
/// Piper's phonemizer for voices of the <c>text</c> type, whose phonemes are the characters of the text: lower-cased and
/// decomposed, the whole text as one sentence (Piper detects no sentence boundaries for this type).
/// </summary>
internal sealed class CodepointPhonemizer : IPhonemizer
{
    public static CodepointPhonemizer Instance { get; } = new();

    public ValueTask<IReadOnlyList<string>> PhonemizeAsync(string text, PhonemeScheme scheme, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(text);

        // Piper applies full Unicode case folding. .NET has none, and invariant lower-casing agrees with it except for a
        // few characters whose folding is not their lower case (for example U+00DF folds to "ss").
        return new([text.ToLowerInvariant().Normalize(NormalizationForm.FormD)]);
    }
}
