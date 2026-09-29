using System.Text.Json;
using LMSupply.Generator.Abstractions;
using LMSupply.Generator.Models;

namespace LMSupply.Generator.ChatFormatters;

/// <summary>
/// Chat formatter for Gemma 4 models with native system role support.
/// Format: &lt;|turn&gt;system\n{content}&lt;turn|&gt;\n&lt;|turn&gt;user\n...
/// Unlike Gemma 2 which mapped system to user, Gemma 4 supports system natively.
/// Tool results are folded into a user turn with a [tool_result: id] marker
/// because Gemma's chat template does not define a dedicated tool turn.
/// </summary>
public sealed class Gemma4ChatFormatter : IChatFormatter
{
    private const string StartOfTurn = "<|turn>";
    private const string EndOfTurn = "<turn|>";

    /// <inheritdoc />
    public string FormatName => "gemma4";

    /// <inheritdoc />
    public string FormatPrompt(IEnumerable<ChatMessage> messages)
    {
        var sb = new StringBuilder();

        foreach (var message in messages)
        {
            if (message.Role == ChatRole.Tool)
            {
                sb.Append(StartOfTurn);
                sb.Append("user\n[tool_result");
                if (!string.IsNullOrEmpty(message.ToolCallId))
                {
                    sb.Append(": ");
                    sb.Append(message.ToolCallId);
                }
                sb.Append("] ");
                sb.Append(message.Content);
                sb.Append(EndOfTurn);
                sb.Append('\n');
                continue;
            }

            var role = message.Role switch
            {
                ChatRole.System => "system",
                ChatRole.User => "user",
                ChatRole.Assistant => "model",
                _ => throw new ArgumentOutOfRangeException(nameof(messages), message.Role, "Unsupported chat role")
            };

            sb.Append(StartOfTurn);
            sb.Append(role);
            sb.Append('\n');
            sb.Append(message.Role == ChatRole.Assistant
                ? ChatMessageRendering.GetAssistantText(message)
                : message.Content);
            sb.Append(EndOfTurn);
            sb.Append('\n');
        }

        // Add model prompt to start generation
        sb.Append(StartOfTurn);
        sb.Append("model");
        sb.Append('\n');

        return sb.ToString();
    }

    /// <inheritdoc />
    public string GetStopToken() => EndOfTurn;

    /// <inheritdoc />
    public IReadOnlyList<string> GetStopSequences() =>
        [EndOfTurn, StartOfTurn];

    /// <inheritdoc />
    /// <remarks>
    /// Gemma 4 emits tool calls inside its native
    /// <c>&lt;|tool_call&gt;call:NAME{ARGS_JSON}&lt;tool_call|&gt;</c> wrapper.
    /// llama-server forwards those wrapper tokens verbatim in the text channel
    /// (often together with spurious half-formed name-only tool-call deltas
    /// extracted from elsewhere in the stream), so chunks never reach the
    /// invoke layer upstream. <see cref="Gemma4ToolCallStreamParser"/> closes
    /// the gap by consuming the wrappers and emitting fully formed
    /// <c>ChatToolCallDelta</c> events. The generator suppresses server-side
    /// tool-call deltas when this parser is active.
    /// </remarks>
    public IToolCallStreamParser CreateToolCallStreamParser()
    {
        return new Gemma4ToolCallStreamParser();
    }

    /// <inheritdoc />
    /// <remarks>
    /// Gemma 4 activates built-in thinking by including <c>&lt;|think|&gt;</c> at the
    /// start of the system prompt. Google recommends this for E2B/E4B when complex
    /// function calling is required.
    /// </remarks>
    public string? GetThinkingToken() => "<|think|>";

    /// <inheritdoc />
    /// <remarks>
    /// Gemma 4 E4B at gguf:gemma4-default emits empty tool args under llama-server's
    /// native template because the model misinterprets the raw JSON schema.
    /// This override exposes <c>Required parameters (MUST be provided): name (type)</c>
    /// marker lines that LlamaServerGeneratorModel injects as a system message.
    /// </remarks>
    public string? RenderToolPromptFragment(IReadOnlyList<ChatToolDefinition>? tools)
    {
        if (tools is null || tools.Count == 0)
        {
            return null;
        }

        var sb = new StringBuilder();
        sb.AppendLine("You have access to the following tools. When you call a tool, you MUST provide every required parameter listed below.");
        sb.AppendLine();

        for (var i = 0; i < tools.Count; i++)
        {
            var tool = tools[i];
            AppendToolFragment(sb, tool);
            if (i < tools.Count - 1)
            {
                sb.AppendLine();
            }
        }

        return sb.ToString();
    }

    /// <inheritdoc />
    /// <remarks>
    /// When thinking is active, the Jinja2 template in llama-server already injects the
    /// full structured tool schema (<c>&lt;|tool&gt;declaration:...&lt;tool|&gt;</c>).
    /// Repeating the full text fragment creates system-prompt pressure that can
    /// degrade tool-call accuracy on small models. This override reduces the fragment
    /// to a minimal per-tool required-params line to preserve the "required args hint"
    /// benefit while eliminating redundant tool descriptions.
    /// </remarks>
    public string? RenderToolPromptFragmentWhenThinking(IReadOnlyList<ChatToolDefinition>? tools)
    {
        if (tools is null || tools.Count == 0)
            return null;

        var sb = new StringBuilder();
        foreach (var tool in tools)
        {
            var (required, _) = ExtractParameters(tool.Parameters);
            if (required.Count == 0)
                continue;
            sb.Append(tool.Name).Append(": ").AppendLine(string.Join(", ", required));
        }
        return sb.Length > 0 ? sb.ToString().TrimEnd() : null;
    }

    private static void AppendToolFragment(StringBuilder sb, ChatToolDefinition tool)
    {
        if (string.IsNullOrEmpty(tool.Description))
        {
            sb.Append("Tool: ").AppendLine(tool.Name);
        }
        else
        {
            sb.Append("Tool: ").Append(tool.Name).Append(" - ").AppendLine(tool.Description);
        }

        var (required, optional) = ExtractParameters(tool.Parameters);

        if (required.Count > 0)
        {
            sb.Append("Required parameters (MUST be provided): ");
            sb.AppendLine(string.Join(", ", required));
        }
        else
        {
            sb.AppendLine("Required parameters (MUST be provided): (none)");
        }

        if (optional.Count > 0)
        {
            sb.Append("Optional parameters: ");
            sb.AppendLine(string.Join(", ", optional));
        }
    }

    private static (List<string> Required, List<string> Optional) ExtractParameters(JsonElement? parameters)
    {
        var required = new List<string>();
        var optional = new List<string>();

        if (parameters is not { } schema || schema.ValueKind != JsonValueKind.Object)
        {
            return (required, optional);
        }

        var requiredNames = new HashSet<string>(StringComparer.Ordinal);
        if (schema.TryGetProperty("required", out var requiredArray) && requiredArray.ValueKind == JsonValueKind.Array)
        {
            foreach (var name in requiredArray.EnumerateArray())
            {
                if (name.ValueKind == JsonValueKind.String && name.GetString() is { Length: > 0 } str)
                {
                    requiredNames.Add(str);
                }
            }
        }

        if (!schema.TryGetProperty("properties", out var properties) || properties.ValueKind != JsonValueKind.Object)
        {
            return (required, optional);
        }

        foreach (var property in properties.EnumerateObject())
        {
            var name = property.Name;
            var typeText = property.Value.TryGetProperty("type", out var typeElem) && typeElem.ValueKind == JsonValueKind.String
                ? typeElem.GetString()
                : null;
            var rendered = string.IsNullOrEmpty(typeText) ? name : $"{name} ({typeText})";

            if (requiredNames.Contains(name))
            {
                required.Add(rendered);
            }
            else
            {
                optional.Add(rendered);
            }
        }

        return (required, optional);
    }
}
