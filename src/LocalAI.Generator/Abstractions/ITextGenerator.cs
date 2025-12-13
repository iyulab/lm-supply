using LocalAI.Generator.Models;

namespace LocalAI.Generator.Abstractions;

/// <summary>
/// Interface for text generation with streaming support.
/// </summary>
public interface ITextGenerator : IAsyncDisposable
{
    /// <summary>
    /// Gets the model identifier.
    /// </summary>
    string ModelId { get; }

    /// <summary>
    /// Generates text with streaming output.
    /// </summary>
    /// <param name="prompt">The input prompt.</param>
    /// <param name="options">Generation options. If null, default options are used.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>An async enumerable of generated text tokens.</returns>
    IAsyncEnumerable<string> GenerateAsync(
        string prompt,
        GeneratorOptions? options = null,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Generates text from chat messages with streaming output.
    /// </summary>
    /// <param name="messages">The chat messages.</param>
    /// <param name="options">Generation options. If null, default options are used.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>An async enumerable of generated text tokens.</returns>
    IAsyncEnumerable<string> GenerateChatAsync(
        IEnumerable<ChatMessage> messages,
        GeneratorOptions? options = null,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Generates complete text response (non-streaming).
    /// </summary>
    /// <param name="prompt">The input prompt.</param>
    /// <param name="options">Generation options. If null, default options are used.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>The complete generated text.</returns>
    Task<string> GenerateCompleteAsync(
        string prompt,
        GeneratorOptions? options = null,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Generates complete chat response (non-streaming).
    /// </summary>
    /// <param name="messages">The chat messages.</param>
    /// <param name="options">Generation options. If null, default options are used.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>The complete generated text.</returns>
    Task<string> GenerateChatCompleteAsync(
        IEnumerable<ChatMessage> messages,
        GeneratorOptions? options = null,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Pre-loads the model to avoid cold start latency on first inference.
    /// </summary>
    /// <param name="cancellationToken">Cancellation token.</param>
    Task WarmupAsync(CancellationToken cancellationToken = default);
}
