using LMSupply.Generator.Abstractions;
using LMSupply.Generator.Models;

namespace LMSupply.Generator.ChatFormatters;

/// <summary>
/// Chat formatter for Phi-3 and Phi-3.5 models.
/// Format: &lt;|system|&gt;\n{content}&lt;|end|&gt;\n&lt;|user|&gt;\n{content}&lt;|end|&gt;\n&lt;|assistant|&gt;\n
/// </summary>
public sealed class Phi3ChatFormatter : IChatFormatter
{
    /// <inheritdoc />
    public string FormatName => "phi3";

    /// <inheritdoc />
    public string FormatPrompt(IEnumerable<ChatMessage> messages)
    {
        var sb = new StringBuilder();

        foreach (var message in messages)
        {
            var role = message.Role switch
            {
                ChatRole.System => "system",
                ChatRole.User => "user",
                ChatRole.Assistant => "assistant",
                _ => throw new ArgumentOutOfRangeException(nameof(message.Role))
            };

            sb.Append($"<|{role}|>\n");
            sb.Append(message.Content);
            sb.Append("<|end|>\n");
        }

        // Add assistant prompt to start generation
        sb.Append("<|assistant|>\n");

        return sb.ToString();
    }

    /// <inheritdoc />
    public string GetStopToken() => "<|end|>";

    /// <inheritdoc />
    public IReadOnlyList<string> GetStopSequences() =>
        ["<|end|>", "<|user|>", "<|system|>", "<|endoftext|>"];
}
