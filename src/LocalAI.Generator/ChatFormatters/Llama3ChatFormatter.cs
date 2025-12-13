using LocalAI.Generator.Abstractions;
using LocalAI.Generator.Models;

namespace LocalAI.Generator.ChatFormatters;

/// <summary>
/// Chat formatter for Llama 3 and Llama 3.2 models.
/// Format: &lt;|begin_of_text|&gt;&lt;|start_header_id|&gt;system&lt;|end_header_id|&gt;\n\n{content}&lt;|eot_id|&gt;...
/// </summary>
public sealed class Llama3ChatFormatter : IChatFormatter
{
    private const string BeginOfText = "<|begin_of_text|>";
    private const string StartHeaderId = "<|start_header_id|>";
    private const string EndHeaderId = "<|end_header_id|>";
    private const string EotId = "<|eot_id|>";

    /// <inheritdoc />
    public string FormatName => "llama3";

    /// <inheritdoc />
    public string FormatPrompt(IEnumerable<ChatMessage> messages)
    {
        var sb = new StringBuilder();
        var isFirst = true;

        foreach (var message in messages)
        {
            var role = message.Role switch
            {
                ChatRole.System => "system",
                ChatRole.User => "user",
                ChatRole.Assistant => "assistant",
                _ => throw new ArgumentOutOfRangeException(nameof(message.Role))
            };

            // Add begin_of_text only at the start
            if (isFirst)
            {
                sb.Append(BeginOfText);
                isFirst = false;
            }

            sb.Append(StartHeaderId);
            sb.Append(role);
            sb.Append(EndHeaderId);
            sb.Append("\n\n");
            sb.Append(message.Content);
            sb.Append(EotId);
        }

        // Add assistant prompt to start generation
        sb.Append(StartHeaderId);
        sb.Append("assistant");
        sb.Append(EndHeaderId);
        sb.Append("\n\n");

        return sb.ToString();
    }

    /// <inheritdoc />
    public string GetStopToken() => EotId;

    /// <inheritdoc />
    public IReadOnlyList<string> GetStopSequences() =>
        [EotId, StartHeaderId];
}
