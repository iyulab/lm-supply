using LMSupply.Transcriber.Models;

namespace LMSupply.Transcriber;

/// <summary>
/// Interface for speech-to-text transcription models.
/// </summary>
public interface ITranscriberModel : IAsyncDisposable
{
    /// <summary>
    /// Gets the model identifier.
    /// </summary>
    string ModelId { get; }

    /// <summary>
    /// Gets the detected or specified language code.
    /// </summary>
    string? Language { get; }

    /// <summary>
    /// Warms up the model by running a dummy inference.
    /// </summary>
    /// <param name="cancellationToken">Cancellation token.</param>
    Task WarmupAsync(CancellationToken cancellationToken = default);

    /// <summary>
    /// Gets model information if available.
    /// </summary>
    /// <returns>Model information or null.</returns>
    TranscriberModelInfo? GetModelInfo();

    /// <summary>
    /// Transcribes audio from a file path.
    /// </summary>
    /// <param name="audioPath">Path to the audio file.</param>
    /// <param name="options">Optional transcription options.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Transcription result containing text and segments.</returns>
    Task<TranscriptionResult> TranscribeAsync(
        string audioPath,
        TranscribeOptions? options = null,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Transcribes audio from a stream.
    /// </summary>
    /// <param name="audioStream">Stream containing audio data.</param>
    /// <param name="options">Optional transcription options.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Transcription result containing text and segments.</returns>
    Task<TranscriptionResult> TranscribeAsync(
        Stream audioStream,
        TranscribeOptions? options = null,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Transcribes audio from a byte array.
    /// </summary>
    /// <param name="audioData">Byte array containing audio data.</param>
    /// <param name="options">Optional transcription options.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Transcription result containing text and segments.</returns>
    Task<TranscriptionResult> TranscribeAsync(
        byte[] audioData,
        TranscribeOptions? options = null,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Transcribes audio with streaming output (segment by segment).
    /// </summary>
    /// <param name="audioPath">Path to the audio file.</param>
    /// <param name="options">Optional transcription options.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Async enumerable of transcription segments.</returns>
    IAsyncEnumerable<TranscriptionSegment> TranscribeStreamingAsync(
        string audioPath,
        TranscribeOptions? options = null,
        CancellationToken cancellationToken = default);
}
