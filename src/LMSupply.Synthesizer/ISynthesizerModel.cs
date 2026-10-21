using LMSupply.Synthesizer.Models;

namespace LMSupply.Synthesizer;

/// <summary>
/// Interface for text-to-speech synthesis models.
/// </summary>
/// <remarks>
/// Text is turned into the voice's phonemes by <see cref="SynthesizerOptions.Phonemizer"/> when one is set, and otherwise
/// by the package: the built-in English and Korean phonemizers for the voices in those languages, the text itself for a
/// voice whose phonemes are its text. A voice in another language needs a phonemizer, or phonemes (<see cref="SynthesizePhonemesAsync"/>).
/// </remarks>
public interface ISynthesizerModel : IAsyncDisposable
{
    /// <summary>
    /// Gets the model identifier.
    /// </summary>
    string ModelId { get; }

    /// <summary>
    /// Gets whether GPU acceleration is being used for inference.
    /// </summary>
    bool IsGpuActive { get; }

    /// <summary>
    /// Gets the list of active execution providers.
    /// </summary>
    IReadOnlyList<string> ActiveProviders { get; }

    /// <summary>
    /// Gets the execution provider that was requested.
    /// </summary>
    ExecutionProvider RequestedProvider { get; }

    /// <summary>
    /// Gets the estimated memory usage of this model in bytes.
    /// Based on ONNX model file size with overhead factor.
    /// </summary>
    long? EstimatedMemoryBytes { get; }

    /// <summary>
    /// Gets the current voice being used.
    /// </summary>
    string? Voice { get; }

    /// <summary>
    /// Gets the sample rate of the generated audio, as the voice's config declares it.
    /// </summary>
    int SampleRate { get; }

    /// <summary>
    /// Gets the phonemes the voice takes, as its config declares them: what an <see cref="IPhonemizer"/> is asked for and
    /// what <see cref="SynthesizePhonemesAsync"/> expects.
    /// </summary>
    PhonemeScheme PhonemeScheme { get; }

    /// <summary>
    /// Warms up the model by running a minimal inference.
    /// </summary>
    /// <param name="cancellationToken">Cancellation token.</param>
    Task WarmupAsync(CancellationToken cancellationToken = default);

    /// <summary>
    /// Gets information about the loaded model.
    /// </summary>
    /// <returns>Model information, or null if not available.</returns>
    SynthesizerModelInfo? GetModelInfo();

    /// <summary>
    /// Synthesizes speech from text. The text is turned into phonemes by <see cref="SynthesizerOptions.Phonemizer"/>, or
    /// by the package itself for an English or Korean voice and for a voice whose phonemes are its text.
    /// </summary>
    /// <param name="text">The text to synthesize.</param>
    /// <param name="options">Optional synthesis options.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>The synthesis result containing audio data.</returns>
    /// <exception cref="NotSupportedException">
    /// The voice needs a phonemizer and none was supplied (a voice in a language other than English and Korean), or an
    /// English or Korean voice lacks a phoneme the built-in phonemizer writes. No audio is produced. This holds for every method that takes text.
    /// </exception>
    Task<SynthesisResult> SynthesizeAsync(
        string text,
        SynthesizeOptions? options = null,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Synthesizes speech from phonemes of the voice's <see cref="PhonemeScheme"/>, in the form an
    /// <see cref="IPhonemizer"/> returns them.
    /// </summary>
    /// <param name="sentences">
    /// One string per sentence; every Unicode code point is one phoneme, spaces and clause punctuation included.
    /// </param>
    /// <param name="options">Optional synthesis options.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>The synthesis result: the sentences' audio one after another.</returns>
    Task<SynthesisResult> SynthesizePhonemesAsync(
        IReadOnlyList<string> sentences,
        SynthesizeOptions? options = null,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Synthesizes speech from text and writes directly to a stream.
    /// </summary>
    /// <param name="text">The text to synthesize.</param>
    /// <param name="outputStream">The stream to write audio data to.</param>
    /// <param name="options">Optional synthesis options.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    Task SynthesizeToStreamAsync(
        string text,
        Stream outputStream,
        SynthesizeOptions? options = null,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Synthesizes speech from text and saves to a WAV file.
    /// </summary>
    /// <param name="text">The text to synthesize.</param>
    /// <param name="outputPath">The path to save the WAV file.</param>
    /// <param name="options">Optional synthesis options.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    Task SynthesizeToFileAsync(
        string text,
        string outputPath,
        SynthesizeOptions? options = null,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Synthesizes speech from text with streaming output: one chunk per sentence the phonemizer returns.
    /// </summary>
    /// <param name="text">The text to synthesize.</param>
    /// <param name="options">Optional synthesis options.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Async enumerable of audio chunks.</returns>
    IAsyncEnumerable<AudioChunk> SynthesizeStreamingAsync(
        string text,
        SynthesizeOptions? options = null,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Synthesizes speech from phonemes with streaming output: one chunk per sentence.
    /// </summary>
    /// <param name="sentences">The phonemes, as <see cref="SynthesizePhonemesAsync"/> takes them.</param>
    /// <param name="options">Optional synthesis options.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Async enumerable of audio chunks.</returns>
    IAsyncEnumerable<AudioChunk> SynthesizePhonemesStreamingAsync(
        IReadOnlyList<string> sentences,
        SynthesizeOptions? options = null,
        CancellationToken cancellationToken = default);
}
