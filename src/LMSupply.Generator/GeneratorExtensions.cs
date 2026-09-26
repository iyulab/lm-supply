using LMSupply.Generator.Abstractions;
using LMSupply.Generator.Models;

namespace LMSupply.Generator;

/// <summary>
/// Extension methods for ITextGenerator and IGeneratorModel.
/// </summary>
public static class GeneratorExtensions
{
    /// <summary>
    /// Generates a complete chat response with token usage statistics.
    /// </summary>
    /// <param name="generator">The generator model.</param>
    /// <param name="messages">The chat messages.</param>
    /// <param name="options">Generation options.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>
    /// Generation result including content, usage statistics, the finish reason and, on llama-server, the server's
    /// timings. Usage is the backend's own count where it reports one, and an estimate marked
    /// <see cref="TokenUsage.IsEstimated"/> otherwise.
    /// </returns>
    public static Task<GenerationResult> GenerateChatWithUsageAsync(
        this ITextGenerator generator,
        IEnumerable<ChatMessage> messages,
        GenerationOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(generator);
        return generator.GenerateChatCompleteResultAsync(messages, options, cancellationToken);
    }

    /// <summary>
    /// Generates a complete text response with token usage statistics.
    /// </summary>
    /// <param name="generator">The generator model.</param>
    /// <param name="prompt">The input prompt.</param>
    /// <param name="options">Generation options.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Generation result including content, usage statistics and the finish reason.</returns>
    public static Task<GenerationResult> GenerateWithUsageAsync(
        this ITextGenerator generator,
        string prompt,
        GenerationOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(generator);
        return generator.GenerateCompleteResultAsync(prompt, options, cancellationToken);
    }
}
