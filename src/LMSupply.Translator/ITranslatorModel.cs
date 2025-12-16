using LMSupply.Translator.Models;

namespace LMSupply.Translator;

/// <summary>
/// Interface for machine translation models.
/// </summary>
public interface ITranslatorModel : IAsyncDisposable
{
    /// <summary>
    /// Gets the model identifier.
    /// </summary>
    string ModelId { get; }

    /// <summary>
    /// Gets the source language code.
    /// </summary>
    string SourceLanguage { get; }

    /// <summary>
    /// Gets the target language code.
    /// </summary>
    string TargetLanguage { get; }

    /// <summary>
    /// Warms up the model by running a dummy inference.
    /// </summary>
    /// <param name="cancellationToken">Cancellation token.</param>
    Task WarmupAsync(CancellationToken cancellationToken = default);

    /// <summary>
    /// Gets model information if available.
    /// </summary>
    /// <returns>Model information or null.</returns>
    TranslatorModelInfo? GetModelInfo();

    /// <summary>
    /// Translates text from source to target language.
    /// </summary>
    /// <param name="text">The text to translate.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Translation result containing translated text.</returns>
    Task<TranslationResult> TranslateAsync(
        string text,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Translates multiple texts from source to target language.
    /// </summary>
    /// <param name="texts">The texts to translate.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>List of translation results.</returns>
    Task<IReadOnlyList<TranslationResult>> TranslateBatchAsync(
        IEnumerable<string> texts,
        CancellationToken cancellationToken = default);
}
