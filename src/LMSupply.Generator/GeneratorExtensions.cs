using System.Globalization;
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
    /// <returns>Generation result including content and usage statistics.</returns>
    public static async Task<GenerationResult> GenerateChatWithUsageAsync(
        this ITextGenerator generator,
        IEnumerable<ChatMessage> messages,
        GenerationOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        // Format messages to estimate prompt tokens
        var formattedPrompt = FormatMessagesForEstimation(messages);
        var promptTokens = TokenUsage.EstimateTokens(formattedPrompt);

        // Generate the completion
        var sb = new StringBuilder();
        await foreach (var token in generator.GenerateChatAsync(messages, options, cancellationToken))
        {
            sb.Append(token);
        }

        var content = sb.ToString();
        var completionTokens = TokenUsage.EstimateTokens(content);

        return new GenerationResult(
            content,
            new TokenUsage(promptTokens, completionTokens));
    }

    /// <summary>
    /// Generates a complete text response with token usage statistics.
    /// </summary>
    /// <param name="generator">The generator model.</param>
    /// <param name="prompt">The input prompt.</param>
    /// <param name="options">Generation options.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Generation result including content and usage statistics.</returns>
    public static async Task<GenerationResult> GenerateWithUsageAsync(
        this ITextGenerator generator,
        string prompt,
        GenerationOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        var promptTokens = TokenUsage.EstimateTokens(prompt);

        // Generate the completion
        var sb = new StringBuilder();
        await foreach (var token in generator.GenerateAsync(prompt, options, cancellationToken))
        {
            sb.Append(token);
        }

        var content = sb.ToString();
        var completionTokens = TokenUsage.EstimateTokens(content);

        return new GenerationResult(
            content,
            new TokenUsage(promptTokens, completionTokens));
    }

    private static string FormatMessagesForEstimation(IEnumerable<ChatMessage> messages)
    {
        var sb = new StringBuilder();
        foreach (var message in messages)
        {
            sb.AppendLine(CultureInfo.InvariantCulture, $"{message.Role}: {message.Content}");
        }
        return sb.ToString();
    }
}
