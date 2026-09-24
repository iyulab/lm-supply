using System.Text.Json;
using LMSupply.Generator.Abstractions;
using LMSupply.Generator.Models;

namespace LMSupply.Generator.ChatFormatters;

/// <summary>
/// Chat formatter for Mistral/Mixtral models.
/// Format: [INST] {user_message} [/INST] {assistant_message}
/// Tool calls and results use Mistral v3 instruct tokens
/// ([TOOL_CALLS], [TOOL_RESULTS], [/TOOL_RESULTS]).
/// </summary>
public sealed class MistralChatFormatter : IChatFormatter
{
    private const string InstStart = "[INST]";
    private const string InstEnd = "[/INST]";
    private const string ToolCallsTag = "[TOOL_CALLS]";
    private const string ToolResultsStart = "[TOOL_RESULTS]";
    private const string ToolResultsEnd = "[/TOOL_RESULTS]";
    private const string BosToken = "<s>";
    private const string EosToken = "</s>";

    /// <inheritdoc />
    public string FormatName => "mistral";

    /// <inheritdoc />
    public string FormatPrompt(IEnumerable<ChatMessage> messages)
    {
        var sb = new StringBuilder();
        // The format has no system turn: system text rides in the next [INST] block.
        // Every system message is carried (several in a row are joined), none is dropped.
        var pendingSystem = new List<string>();

        sb.Append(BosToken);

        foreach (var message in messages)
        {
            if (message.Role == ChatRole.System)
            {
                pendingSystem.Add(message.Content);
            }
            else if (message.Role == ChatRole.User)
            {
                sb.Append(InstStart);
                sb.Append(' ');

                if (pendingSystem.Count > 0)
                {
                    sb.Append(string.Join("\n\n", pendingSystem));
                    sb.Append("\n\n");
                    pendingSystem.Clear();
                }

                sb.Append(message.Content);
                sb.Append(' ');
                sb.Append(InstEnd);
            }
            else if (message.Role == ChatRole.Assistant)
            {
                if (message.ToolCalls is { Count: > 0 } && string.IsNullOrEmpty(message.Content))
                {
                    sb.Append(ToolCallsTag);
                    sb.Append(' ');
                    sb.Append(SerializeToolCallsForMistral(message.ToolCalls));
                    sb.Append(EosToken);
                }
                else
                {
                    sb.Append(' ');
                    sb.Append(message.Content);
                    sb.Append(EosToken);
                }
            }
            else if (message.Role == ChatRole.Tool)
            {
                sb.Append(ToolResultsStart);
                sb.Append(' ');
                sb.Append(SerializeToolResult(message.ToolCallId, message.Content));
                sb.Append(' ');
                sb.Append(ToolResultsEnd);
            }
        }

        return sb.ToString();
    }

    /// <inheritdoc />
    public string GetStopToken() => EosToken;

    /// <inheritdoc />
    public IReadOnlyList<string> GetStopSequences() =>
        [EosToken, InstStart];

    private static string SerializeToolCallsForMistral(IReadOnlyList<ChatToolCall> toolCalls)
    {
        // Mistral v3 emits [{ "name": ..., "arguments": ..., "id": ... }] inside [TOOL_CALLS]
        using var stream = new MemoryStream();
        using (var writer = new Utf8JsonWriter(stream))
        {
            writer.WriteStartArray();
            foreach (var call in toolCalls)
            {
                writer.WriteStartObject();
                writer.WriteString("name", call.FunctionName);
                writer.WriteString("arguments", call.Arguments);
                writer.WriteString("id", call.Id);
                writer.WriteEndObject();
            }
            writer.WriteEndArray();
        }
        return Encoding.UTF8.GetString(stream.ToArray());
    }

    private static string SerializeToolResult(string? callId, string content)
    {
        using var stream = new MemoryStream();
        using (var writer = new Utf8JsonWriter(stream))
        {
            writer.WriteStartObject();
            writer.WriteString("call_id", callId ?? string.Empty);
            writer.WriteString("content", content);
            writer.WriteEndObject();
        }
        return Encoding.UTF8.GetString(stream.ToArray());
    }
}
