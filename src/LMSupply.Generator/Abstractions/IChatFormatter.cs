using LMSupply.Generator.Models;

namespace LMSupply.Generator.Abstractions;

/// <summary>
/// Interface for formatting chat messages into model-specific prompt formats.
/// </summary>
public interface IChatFormatter
{
    /// <summary>
    /// Gets the name of the chat format (e.g., "phi3", "llama3", "chatml").
    /// </summary>
    string FormatName { get; }

    /// <summary>
    /// Formats a sequence of chat messages into a model-specific prompt string.
    /// </summary>
    /// <param name="messages">The chat messages to format.</param>
    /// <returns>The formatted prompt string ready for model input.</returns>
    string FormatPrompt(IEnumerable<ChatMessage> messages);

    /// <summary>
    /// Gets the primary stop token for this format.
    /// </summary>
    string GetStopToken();

    /// <summary>
    /// Gets all stop sequences that should terminate generation.
    /// </summary>
    IReadOnlyList<string> GetStopSequences();
}
