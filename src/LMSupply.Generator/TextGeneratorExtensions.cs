using System.Text;
using LMSupply.Generator.Abstractions;
using LMSupply.Generator.Internal;
using LMSupply.Generator.Models;

namespace LMSupply.Generator;

/// <summary>
/// Extension methods for ITextGenerator.
/// </summary>
public static class TextGeneratorExtensions
{
    /// <summary>A <see cref="ReasoningResult"/> holds text only; log probabilities asked for here would be dropped.</summary>
    private static void ThrowIfTopLogprobsRequested(GenerationOptions options)
    {
        if (options.TopLogprobs is not null)
            throw new NotSupportedException(
                "The reasoning helpers return text only and cannot report per-token log probabilities. Use " +
                "GenerateChatCompleteResultAsync or GenerateChatStreamAsync with TopLogprobs, or leave it null here.");
    }

    /// <summary>
    /// Generates chat response with reasoning content extracted separately.
    /// Useful for DeepSeek R1 and other reasoning models.
    /// </summary>
    /// <param name="generator">The text generator.</param>
    /// <param name="messages">The chat messages.</param>
    /// <param name="options">Generation options (ExtractReasoningTokens will be enabled).</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Result containing both the response and extracted reasoning.</returns>
    public static async Task<ReasoningResult> GenerateChatWithReasoningAsync(
        this ITextGenerator generator,
        IEnumerable<ChatMessage> messages,
        GenerationOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        options ??= GenerationOptions.Default;
        ThrowIfTopLogprobsRequested(options);

        // Create options with reasoning extraction enabled
        var extractOptions = options.Clone();
        extractOptions.FilterReasoningTokens = true;
        extractOptions.ExtractReasoningTokens = true;

        // Use our own filter to capture reasoning
        var filter = new ReasoningTokenFilter(extractReasoning: true);
        var responseBuilder = new StringBuilder();

        await foreach (var token in generator.GenerateChatAsync(messages, extractOptions, cancellationToken))
        {
            // Tokens are already filtered by the generator if it supports it
            responseBuilder.Append(token);
        }

        // For generators that don't support filtering internally,
        // we can process the full response
        var response = responseBuilder.ToString();

        // If the generator didn't filter, do it here
        if (response.Contains("<think>") || response.Contains("<｜begin▁of▁thinking｜>"))
        {
            var refilter = new ReasoningTokenFilter(extractReasoning: true);
            var filteredResponse = new StringBuilder();

            foreach (var chunk in ChunkText(response, 50))
            {
                var filtered = refilter.Process(chunk);
                filteredResponse.Append(filtered);
            }

            filteredResponse.Append(refilter.Flush());
            return new ReasoningResult(filteredResponse.ToString(), refilter.ReasoningContent);
        }

        return new ReasoningResult(response, string.Empty);
    }

    /// <summary>
    /// Generates text with reasoning content extracted separately.
    /// Useful for DeepSeek R1 and other reasoning models.
    /// </summary>
    /// <param name="generator">The text generator.</param>
    /// <param name="prompt">The input prompt.</param>
    /// <param name="options">Generation options (ExtractReasoningTokens will be enabled).</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Result containing both the response and extracted reasoning.</returns>
    public static async Task<ReasoningResult> GenerateWithReasoningAsync(
        this ITextGenerator generator,
        string prompt,
        GenerationOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        options ??= GenerationOptions.Default;
        ThrowIfTopLogprobsRequested(options);

        var extractOptions = options.Clone();
        extractOptions.FilterReasoningTokens = true;
        extractOptions.ExtractReasoningTokens = true;

        var filter = new ReasoningTokenFilter(extractReasoning: true);
        var responseBuilder = new StringBuilder();

        await foreach (var token in generator.GenerateAsync(prompt, extractOptions, cancellationToken))
        {
            responseBuilder.Append(token);
        }

        var response = responseBuilder.ToString();

        if (response.Contains("<think>") || response.Contains("<｜begin▁of▁thinking｜>"))
        {
            var refilter = new ReasoningTokenFilter(extractReasoning: true);
            var filteredResponse = new StringBuilder();

            foreach (var chunk in ChunkText(response, 50))
            {
                var filtered = refilter.Process(chunk);
                filteredResponse.Append(filtered);
            }

            filteredResponse.Append(refilter.Flush());
            return new ReasoningResult(filteredResponse.ToString(), refilter.ReasoningContent);
        }

        return new ReasoningResult(response, string.Empty);
    }

    private static IEnumerable<string> ChunkText(string text, int chunkSize)
    {
        for (int i = 0; i < text.Length; i += chunkSize)
        {
            yield return text.Substring(i, Math.Min(chunkSize, text.Length - i));
        }
    }
}
