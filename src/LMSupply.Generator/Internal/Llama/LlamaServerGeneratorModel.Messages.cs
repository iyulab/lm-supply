using LMSupply.Exceptions;
using LMSupply.Generator.Abstractions;
using LMSupply.Generator.Models;
using LMSupply.Llama.Server;

namespace LMSupply.Generator.Internal.Llama;

internal sealed partial class LlamaServerGeneratorModel
{
    private static IEnumerable<ChatCompletionMessage> ConvertMessages(IEnumerable<ChatMessage> messages)
    {
        return messages.Select(m => new ChatCompletionMessage
        {
            Role = m.Role switch
            {
                ChatRole.System => "system",
                ChatRole.User => "user",
                ChatRole.Assistant => "assistant",
                ChatRole.Tool => "tool",
                _ => "user"
            },
            Content = m.Content,
            ToolCallId = m.ToolCallId,
            ToolCalls = m.ToolCalls?.Select(tc => new ToolCallMessage
            {
                Id = tc.Id,
                Type = "function",
                Function = new FunctionCallMessage
                {
                    Name = tc.FunctionName,
                    Arguments = tc.Arguments
                }
            }).ToList()
        });
    }

    private ChatCompletionOptions CreateChatOptions(GenerationOptions options)
    {
        return new ChatCompletionOptions
        {
            MaxTokens = options.ResolveMaxOutputTokens(),
            Temperature = options.Temperature,
            TopP = options.TopP,
            TopK = options.TopK,
            MinP = options.MinP,
            RepeatPenalty = options.RepetitionPenalty,
            FrequencyPenalty = options.FrequencyPenalty,
            PresencePenalty = options.PresencePenalty,
            DryMultiplier = options.DryMultiplier,
            DryBase = options.DryBase,
            DryAllowedLength = options.DryAllowedLength,
            DryPenaltyLastN = options.DryPenaltyLastN,
            RepeatLastN = options.RepeatLastN,
            Seed = options.Seed,
            StopSequences = MergeStopSequences(options.StopSequences),
            Grammar = options.Grammar,
            JsonSchema = options.JsonSchema,
            Tools = options.Tools?.Select(t => new LMSupply.Llama.Server.LlamaToolDefinition
            {
                Type = "function",
                Function = new LMSupply.Llama.Server.FunctionDefinition
                {
                    Name = t.Name,
                    Description = t.Description,
                    Parameters = t.Parameters
                }
            }).ToList(),
            ToolChoice = ToToolChoice(options.ToolChoice),
            // Forward thinking control to the chat template (enable_thinking). Auto -> null (omit ->
            // model default preserved); On/Off -> true/false. See ThinkingModeToEnableFlag.
            EnableThinking = ThinkingModeToEnableFlag(options.Thinking),
            TopLogprobs = options.TopLogprobs
        };
    }

    /// <summary>
    /// Maps the public <see cref="GenerationOptions.ToolChoice"/> to the server-level
    /// <see cref="LMSupply.Llama.Server.LlamaToolChoice"/>. Null (unset) -> null (server default "auto").
    /// </summary>
    internal static LMSupply.Llama.Server.LlamaToolChoice? ToToolChoice(ToolChoice? toolChoice) => toolChoice?.Mode switch
    {
        null or ToolChoiceMode.Auto => null,
        ToolChoiceMode.None => LMSupply.Llama.Server.LlamaToolChoice.None,
        ToolChoiceMode.Required => LMSupply.Llama.Server.LlamaToolChoice.Required,
        ToolChoiceMode.Function => LMSupply.Llama.Server.LlamaToolChoice.Function(toolChoice.FunctionName!),
        _ => null
    };

    /// <summary>
    /// Maps the public <see cref="ThinkingMode"/> to the server-level <c>enable_thinking</c> flag:
    /// Auto -> null (omit chat_template_kwargs so the model's template default applies), On -> true,
    /// Off -> false.
    /// </summary>
    internal static bool? ThinkingModeToEnableFlag(ThinkingMode mode) => mode switch
    {
        ThinkingMode.On => true,
        ThinkingMode.Off => false,
        _ => null
    };

    private List<string>? MergeStopSequences(IReadOnlyList<string>? userStops)
    {
        var merged = new List<string>();

        // 1. Stop sequences from chat formatter
        merged.AddRange(_chatFormatter.GetStopSequences());

        // 2. User-provided stop sequences
        if (userStops != null)
        {
            foreach (var stop in userStops)
            {
                if (!merged.Contains(stop, StringComparer.Ordinal))
                {
                    merged.Add(stop);
                }
            }
        }

        return merged.Count > 0 ? merged : null;
    }

    /// <summary>
    /// Conditionally prepends a textual reinforcement of the active tool schemas as a
    /// system message when the formatter opts in via
    /// <see cref="IChatFormatter.RenderToolPromptFragment"/>. Small/quantized models
    /// (Gemma 4 E4B at gguf:gemma4-default) misinterpret llama-server's raw JSON-schema
    /// rendering and emit empty tool args; the textual fragment raises first-attempt
    /// success.
    /// </summary>
    /// <remarks>
    /// Returns the original sequence unchanged when the formatter returns <c>null</c>
    /// (default for all formatters except Gemma 4) or when no tools are passed —
    /// avoids polluting other model families' prompts with redundant text.
    /// </remarks>
    /// <summary>
    /// Prepends the formatter's thinking token to the first system message when
    /// <paramref name="enableThinking"/> is <c>true</c> and the formatter opts in via
    /// <see cref="IChatFormatter.GetThinkingToken"/>. If no system message exists, a
    /// bare system message containing only the token is prepended.
    /// </summary>
    /// <remarks>
    /// For Gemma 4 E2B/E4B, Google recommends activating thinking mode (via
    /// <c>&lt;|think|&gt;</c>) when complex function calling is required; the model reasons
    /// internally before deciding to invoke a tool.
    /// Call after <see cref="MaybeInjectToolPromptFragment"/> so the thinking token
    /// lands at the top of the combined system message.
    /// </remarks>
    internal static IEnumerable<ChatMessage> MaybeInjectThinkingToken(
        IEnumerable<ChatMessage> messages,
        bool enableThinking,
        IChatFormatter formatter)
    {
        var token = enableThinking ? formatter.GetThinkingToken() : null;
        if (token is null)
            return messages;

        var list = messages.ToList();
        var idx = list.FindIndex(m => m.Role == ChatRole.System);
        if (idx >= 0)
            list[idx] = ChatMessage.System(token + "\n" + list[idx].Content);
        else
            list.Insert(0, ChatMessage.System(token));
        return list;
    }

    /// <summary>
    /// Builds the message list sent to llama-server: tool prompt fragment, thinking token,
    /// then <see cref="FoldLeadingSystemMessages"/>. The token-budget trim runs the same
    /// pipeline so it counts what is actually sent.
    /// </summary>
    internal static IReadOnlyList<ChatMessage> PrepareServerMessages(
        IEnumerable<ChatMessage> messages,
        GenerationOptions options,
        IChatFormatter formatter)
    {
        var thinking = options.Thinking == ThinkingMode.On;
        var augmented = MaybeInjectToolPromptFragment(messages, options.Tools, formatter, thinking);
        augmented = MaybeInjectThinkingToken(augmented, thinking, formatter);
        return FoldLeadingSystemMessages(augmented);
    }

    /// <summary>
    /// Merges a leading run of system messages into one, joined by a blank line.
    /// </summary>
    /// <remarks>
    /// Several system messages in a row are valid in the OpenAI chat format (a system prompt
    /// followed by a conversation summary, or the tool prompt fragment this model prepends),
    /// but llama-server renders them through the model's own Jinja chat template, and some
    /// templates (Qwen 3.x) raise "System message must be at the beginning" for any system
    /// message after index 0 — the request then fails with HTTP 500. One merged system
    /// message is accepted by every template. System messages later in the conversation
    /// are left in place.
    /// </remarks>
    internal static IReadOnlyList<ChatMessage> FoldLeadingSystemMessages(IEnumerable<ChatMessage> messages)
    {
        var list = messages as IReadOnlyList<ChatMessage> ?? messages.ToList();
        var run = 0;
        while (run < list.Count && list[run].Role == ChatRole.System)
            run++;
        if (run < 2)
            return list;

        var merged = ChatMessage.System(string.Join("\n\n", list.Take(run).Select(m => m.Content)));
        var result = new List<ChatMessage>(list.Count - run + 1) { merged };
        for (var i = run; i < list.Count; i++)
            result.Add(list[i]);
        return result;
    }

    internal static IEnumerable<ChatMessage> MaybeInjectToolPromptFragment(
        IEnumerable<ChatMessage> messages,
        IReadOnlyList<ChatToolDefinition>? tools,
        IChatFormatter formatter,
        bool enableThinking = false)
    {
        var fragment = enableThinking
            ? formatter.RenderToolPromptFragmentWhenThinking(tools)
            : formatter.RenderToolPromptFragment(tools);

        if (string.IsNullOrEmpty(fragment))
        {
            foreach (var msg in messages)
            {
                yield return msg;
            }
            yield break;
        }

        yield return ChatMessage.System(fragment);
        foreach (var msg in messages)
        {
            yield return msg;
        }
    }

    /// <summary>
    /// Removes the oldest non-system conversation turn in-place.
    /// A turn is: User + optional (Assistant + optional Tool*).
    /// System messages are never removed. The last User message is protected.
    /// Tool call / result pairs are treated atomically: removing an Assistant with ToolCalls
    /// also removes the following Tool result messages.
    /// </summary>
    /// <returns><c>true</c> if a turn was removed; <c>false</c> when trimming is impossible.</returns>
    internal static bool TryTrimOldestTurn(List<ChatMessage> messages)
    {
        var firstNonSystem = messages.FindIndex(m => m.Role != ChatRole.System);
        if (firstNonSystem < 0)
            return false;

        var turnEnd = firstNonSystem;
        if (turnEnd < messages.Count && messages[turnEnd].Role == ChatRole.User)
            turnEnd++;
        if (turnEnd < messages.Count && messages[turnEnd].Role == ChatRole.Assistant)
            turnEnd++;
        while (turnEnd < messages.Count && messages[turnEnd].Role == ChatRole.Tool)
            turnEnd++;

        // Ensure at least one User message remains after removing this turn
        var hasRemainingUser = false;
        for (var i = turnEnd; i < messages.Count; i++)
        {
            if (messages[i].Role == ChatRole.User)
            {
                hasRemainingUser = true;
                break;
            }
        }
        if (!hasRemainingUser)
            return false;

        messages.RemoveRange(firstNonSystem, turnEnd - firstNonSystem);
        return true;
    }

    /// <summary>
    /// Returns a (possibly trimmed) copy of <paramref name="messages"/> whose augmented prompt
    /// fits within the input token budget: MaxContextLength minus reserved output tokens.
    /// Skips trimming when MaxContextLength is unknown (0).
    /// Throws <see cref="ContextLengthExceededException"/> if the current turn alone exceeds the budget.
    /// </summary>
    private async Task<List<ChatMessage>> TrimToFitContextAsync(
        IEnumerable<ChatMessage> messages,
        GenerationOptions options,
        CancellationToken cancellationToken)
    {
        if (MaxContextLength <= 0)
            return messages.ToList();

        var outputReserved = options.ResolveMaxOutputTokens();
        var inputBudget = MaxContextLength - outputReserved;
        if (inputBudget <= 0)
            return messages.ToList();

        var list = messages.ToList();

        while (true)
        {
            // What the server will render, tools included — the formatter's prompt alone leaves them out.
            var tokenCount = await CountRenderedPromptTokensAsync(
                PrepareServerMessages(list, options, _chatFormatter), options, cancellationToken);

            if (tokenCount <= inputBudget)
                return list;

            if (!TryTrimOldestTurn(list))
                throw new ContextLengthExceededException(tokenCount, MaxContextLength);
        }
    }
}
