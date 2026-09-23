using LMSupply.Generator.Models;

namespace LMSupply.Generator.Abstractions;

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
        GenerationOptions? options = null,
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
        GenerationOptions? options = null,
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
        GenerationOptions? options = null,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Generates a complete text response (non-streaming) together with the reason it ended — the same text as
    /// <see cref="GenerateCompleteAsync"/>, plus <see cref="GenerationResult.FinishReason"/> so a caller can tell
    /// an answer cut off at <see cref="GenerationOptions.MaxTokens"/> (<c>"length"</c>) from a finished one.
    /// Reaching the limit is not an error here: the caller set it, and decides what a cut-off answer means.
    /// </summary>
    /// <param name="prompt">The input prompt.</param>
    /// <param name="options">Generation options. If null, default options are used.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>The generated text, estimated token usage and the finish reason.</returns>
    Task<GenerationResult> GenerateCompleteResultAsync(
        string prompt,
        GenerationOptions? options = null,
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
        GenerationOptions? options = null,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Generates a complete chat response (non-streaming) together with the reason it ended — the same text as
    /// <see cref="GenerateChatCompleteAsync"/>, plus <see cref="GenerationResult.FinishReason"/> so a caller can tell
    /// an answer cut off at <see cref="GenerationOptions.MaxTokens"/> (<c>"length"</c>) from a finished one.
    /// Reaching the limit is not an error here: the caller set it, and decides what a cut-off answer means.
    /// </summary>
    /// <param name="messages">The chat messages.</param>
    /// <param name="options">Generation options. If null, default options are used.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>The generated text, estimated token usage and the finish reason.</returns>
    Task<GenerationResult> GenerateChatCompleteResultAsync(
        IEnumerable<ChatMessage> messages,
        GenerationOptions? options = null,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Generates a chat completion with tool calling support.
    /// Returns structured result that may contain tool calls.
    /// </summary>
    /// <param name="messages">The chat messages.</param>
    /// <param name="options">Generation options. If null, default options are used.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>A chat completion result that may contain text content and/or tool calls.</returns>
    Task<ChatCompletionResult> GenerateChatWithToolsAsync(
        IEnumerable<ChatMessage> messages,
        GenerationOptions? options = null,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Generates a streaming chat completion with structured chunks.
    /// Returns text deltas, tool call deltas, and finish reason in a unified stream.
    /// </summary>
    /// <param name="messages">The chat messages.</param>
    /// <param name="options">Generation options. If null, default options are used.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>An async enumerable of structured stream chunks.</returns>
    IAsyncEnumerable<ChatStreamChunk> GenerateChatStreamAsync(
        IEnumerable<ChatMessage> messages,
        GenerationOptions? options = null,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Pre-loads the model to avoid cold start latency on first inference.
    /// </summary>
    /// <param name="cancellationToken">Cancellation token.</param>
    Task WarmupAsync(CancellationToken cancellationToken = default);
}
