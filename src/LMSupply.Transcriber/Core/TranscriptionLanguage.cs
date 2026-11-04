using LMSupply.Transcriber.Decoding;
using LMSupply.Transcriber.Models;

namespace LMSupply.Transcriber.Core;

/// <summary>
/// Settles <see cref="TranscribeOptions.Language"/> once, at the entry of a transcription, so the decoder and the result
/// only ever see a code the model decodes (or none).
/// </summary>
internal static class TranscriptionLanguage
{
    /// <summary>
    /// <paramref name="options"/> with the language hint replaced by its code, or the same instance when it already is one.
    /// No hint (null, empty, white space) becomes null — identification from the audio, as when none was set.
    /// </summary>
    /// <exception cref="ArgumentException">
    /// The hint is not a language Whisper decodes, or one the model — an English-only export, a fine-tune for one
    /// language, a model family with a shorter list — does not.
    /// </exception>
    internal static TranscribeOptions? Resolve(TranscribeOptions? options, TranscriberModelInfo? model)
    {
        if (options is null)
            return null;

        var code = WhisperTokenizer.ResolveLanguage(options.Language);

        if (code is not null && model?.SupportedLanguages is { Count: > 0 } supported
            && !supported.Contains(code, StringComparer.OrdinalIgnoreCase))
        {
            throw new ArgumentException(
                $"Model '{model.Id}' does not transcribe '{code}' (TranscribeOptions.Language '{options.Language}'). " +
                $"It transcribes: {string.Join(", ", supported)}. Leave Language null to identify the language from the audio.",
                nameof(options));
        }

        return string.Equals(code, options.Language, StringComparison.Ordinal) ? options : options.WithLanguage(code);
    }
}
