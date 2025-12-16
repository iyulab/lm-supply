using LMSupply.Generator.Abstractions;
using LMSupply.Generator.Models;

namespace LMSupply.Generator.ChatFormatters;

/// <summary>
/// Chat formatter for ChatML format (used by Qwen, some Mistral variants).
/// Format: &lt;|im_start|&gt;system\n{content}&lt;|im_end|&gt;\n&lt;|im_start|&gt;user\n{content}&lt;|im_end|&gt;\n...
/// </summary>
public sealed class ChatMLFormatter : IChatFormatter
{
    private const string ImStart = "<|im_start|>";
    private const string ImEnd = "<|im_end|>";

    /// <inheritdoc />
    public string FormatName => "chatml";

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

            sb.Append(ImStart);
            sb.Append(role);
            sb.Append('\n');
            sb.Append(message.Content);
            sb.Append(ImEnd);
            sb.Append('\n');
        }

        // Add assistant prompt to start generation
        sb.Append(ImStart);
        sb.Append("assistant\n");

        return sb.ToString();
    }

    /// <inheritdoc />
    public string GetStopToken() => ImEnd;

    /// <inheritdoc />
    public IReadOnlyList<string> GetStopSequences() =>
        [ImEnd, ImStart];
}
