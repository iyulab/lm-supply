using System.Diagnostics.CodeAnalysis;
using LMSupply.Synthesizer.Models;

namespace LMSupply.Synthesizer;

/// <summary>
/// Interface for text-to-speech synthesis models.
/// </summary>
/// <remarks>Experimental: no text-to-phoneme step yet — see <see cref="SynthesizerDiagnostics.ExperimentalId"/>.</remarks>
[Experimental(SynthesizerDiagnostics.ExperimentalId, UrlFormat = SynthesizerDiagnostics.ExperimentalUrl)]
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
    /// Gets the sample rate of the generated audio.
    /// </summary>
    int SampleRate { get; }

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
    /// Synthesizes speech from text.
    /// </summary>
    /// <param name="text">The text to synthesize.</param>
    /// <param name="options">Optional synthesis options.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>The synthesis result containing audio data.</returns>
    Task<SynthesisResult> SynthesizeAsync(
        string text,
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
    /// Synthesizes speech from text with streaming output.
    /// </summary>
    /// <param name="text">The text to synthesize.</param>
    /// <param name="options">Optional synthesis options.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Async enumerable of audio chunks.</returns>
    IAsyncEnumerable<AudioChunk> SynthesizeStreamingAsync(
        string text,
        SynthesizeOptions? options = null,
        CancellationToken cancellationToken = default);
}
