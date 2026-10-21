using LMSupply.Generator.Abstractions;
using LMSupply.Generator.ChatFormatters;
using LMSupply.Generator.Models;
using LMSupply.Llama.Server;

namespace LMSupply.Generator.Internal.Llama;

internal sealed partial class LlamaServerGeneratorModel
{
    /// <inheritdoc />
    public async IAsyncEnumerable<ChatStreamChunk> GenerateChatStreamAsync(
        IEnumerable<ChatMessage> messages,
        GenerationOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();
        options ??= GenerationOptions.Default;

        await _concurrencyLimiter.WaitAsync(cancellationToken);
        try
        {
            var trimmedMessages = await TrimToFitContextAsync(messages, options, cancellationToken);
            var serverMessages = ConvertMessages(PrepareServerMessages(trimmedMessages, options, _chatFormatter));
            var chatOptions = CreateChatOptions(options);
            var source = (await ClientAsync(cancellationToken)).GenerateChatStreamAsync(serverMessages, chatOptions, cancellationToken);

            await foreach (var chunk in ShapeChatStream(
                source, options, _chatFormatter.CreateToolCallStreamParser(), cancellationToken))
            {
                yield return chunk;
            }
        }
        finally
        {
            _concurrencyLimiter.Release();
        }
    }

    /// <summary>
    /// Shapes llama-server stream data into <see cref="ChatStreamChunk"/>s: the output-token safety limit, reasoning
    /// routing/filtering, formatter tool-call parsing, and one final chunk carrying the finish reason and usage.
    /// Separated from the server call so the shaping is testable over a synthetic stream.
    /// </summary>
    internal static async IAsyncEnumerable<ChatStreamChunk> ShapeChatStream(
        IAsyncEnumerable<ChatStreamData> source,
        GenerationOptions options,
        IToolCallStreamParser? toolStreamParser,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        // Client-side token limit as safety net
        var maxTokens = options.ResolveMaxOutputTokens();
        var tokenCount = 0;

        // Initialize reasoning token filter if needed
        var useReasoningFilter = options.FilterReasoningTokens || options.ExtractReasoningTokens;
        var reasoningFilter = useReasoningFilter
            ? new ReasoningTokenFilter(options.ExtractReasoningTokens)
            : null;

        // toolStreamParser is the formatter-supplied tool-call wrapper parser, if any: server deltas win per chunk and
        // the parser only fills in the chunks where the server gave nothing — see ToolCallStreamCoexistence.

        // The finish reason is held back to one final chunk that also carries the server's usage: with
        // include_usage the usage arrives on a separate chunk after finish_reason, and the parser/filter
        // flushes below can still release text — a consumer that stops reading at FinishReason must
        // not lose either.
        string? finishReason = null;
        ChatTokenUsage? usage = null;
        GenerationTimings? timings = null;

        await foreach (var data in source.WithCancellation(cancellationToken))
        {
            // Before the first token: nothing to shape, and the consumer is waiting for any sign of life.
            if (data.PromptProgress is { } progress)
            {
                yield return new ChatStreamChunk { PromptProgress = ToPromptProgress(progress) };
                continue;
            }

            if (data.FinishReason is not null)
                finishReason = data.FinishReason;
            if (data.Usage is { } reported)
                usage = new ChatTokenUsage
                {
                    PromptTokens = reported.PromptTokens,
                    CompletionTokens = reported.CompletionTokens,
                    TotalTokens = reported.TotalTokens,
                };
            if (data.Timings is { } reportedTimings)
                timings = ToGenerationTimings(reportedTimings);

            // Safety net: stop if token limit exceeded
            if (data.TextDelta is not null && maxTokens > 0 && ++tokenCount > maxTokens)
            {
                finishReason = "length";
                usage = null;
                timings = null;
                break;
            }

            // Convert tool call deltas from server types to Generator types.
            IReadOnlyList<ChatToolCallDelta>? serverToolCallDeltas = null;
            if (data.ToolCallDeltas is { Count: > 0 })
            {
                serverToolCallDeltas = data.ToolCallDeltas.Select(tc => new ChatToolCallDelta
                {
                    Index = tc.Index,
                    Id = tc.Id,
                    Name = tc.Function?.Name,
                    Arguments = tc.Function?.Arguments
                }).ToList();
            }

            IReadOnlyList<ChatToolCallDelta>? toolCallDeltas =
                toolStreamParser is null ? serverToolCallDeltas : null;

            // Log probabilities follow the generated tokens, so they leave on this chunk even when the reasoning
            // filter or the tool-call parser holds its text back.
            var logprobs = ToTokenLogprobs(data.Logprobs);

            // b8994+: reasoning_content arrives as ReasoningDelta (separate from content).
            // Route it to ChatStreamChunk.ReasoningDelta when extraction is requested;
            // silently discard when only filtering is requested (server already separates it).
            string? reasoningDelta = null;
            if (data.ReasoningDelta is not null && options.ExtractReasoningTokens)
                reasoningDelta = data.ReasoningDelta;

            // Apply reasoning filter to text delta (old-server path: <think> tags in content).
            var text = data.TextDelta;
            if (text is not null && reasoningFilter is not null)
            {
                text = reasoningFilter.Process(text);
                if (string.IsNullOrEmpty(text))
                    text = null;
            }

            // Route remaining text through the formatter-supplied wrapper parser.
            if (toolStreamParser is not null)
                (text, toolCallDeltas) = ToolCallStreamCoexistence.Resolve(text, serverToolCallDeltas, toolStreamParser);

            // Yield structured chunk
            if (text is not null || reasoningDelta is not null || toolCallDeltas is not null || logprobs is not null)
            {
                yield return new ChatStreamChunk
                {
                    Text = text,
                    ReasoningDelta = reasoningDelta,
                    ToolCalls = toolCallDeltas,
                    Logprobs = logprobs,
                };
            }
        }

        // Flush remaining reasoning content
        if (reasoningFilter is not null)
        {
            var remaining = reasoningFilter.Flush();
            if (!string.IsNullOrEmpty(remaining))
            {
                var residual = remaining;
                IReadOnlyList<ChatToolCallDelta>? residualCalls = null;
                if (toolStreamParser is not null)
                {
                    var parsed = toolStreamParser.Feed(residual);
                    residual = parsed.Text;
                    residualCalls = parsed.ToolCalls;
                }
                if (residual is not null || residualCalls is not null)
                {
                    yield return new ChatStreamChunk { Text = residual, ToolCalls = residualCalls };
                }
            }
        }

        // Flush formatter-supplied parser (releases trailing text outside any wrapper;
        // incomplete wrapper bodies are discarded — see Gemma4ToolCallStreamParser).
        if (toolStreamParser is not null)
        {
            var flushed = toolStreamParser.Flush();
            if (flushed.Text is not null || flushed.ToolCalls is { Count: > 0 })
            {
                yield return new ChatStreamChunk
                {
                    Text = flushed.Text,
                    ToolCalls = flushed.ToolCalls
                };
            }
        }

        if (finishReason is not null || usage is not null || timings is not null)
            yield return new ChatStreamChunk { FinishReason = finishReason, Usage = usage, Timings = timings };
    }

    /// <summary>
    /// Generates a chat completion with tool calling support.
    /// Returns structured result that may contain tool calls.
    /// </summary>
    public async Task<ChatCompletionResult> GenerateChatWithToolsAsync(
        IEnumerable<ChatMessage> messages,
        GenerationOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();
        options ??= GenerationOptions.Default;

        await _concurrencyLimiter.WaitAsync(cancellationToken);
        try
        {
            var trimmedMessages = await TrimToFitContextAsync(messages, options, cancellationToken);
            var serverMessages = ConvertMessages(PrepareServerMessages(trimmedMessages, options, _chatFormatter));
            var chatOptions = CreateChatOptions(options);

            var response = await (await ClientAsync(cancellationToken)).GenerateChatWithToolsAsync(
                serverMessages, chatOptions, cancellationToken);

            var choice = response.Choices?.FirstOrDefault();
            var message = choice?.Message;

            return new ChatCompletionResult
            {
                Content = message?.Content,
                FinishReason = choice?.FinishReason,
                ToolCalls = message?.ToolCalls?.Select(tc => new ChatToolCall(
                    tc.Id ?? string.Empty,
                    tc.Function?.Name ?? string.Empty,
                    tc.Function?.Arguments ?? string.Empty
                )).ToList(),
                // reasoning_content is carried as-is: a bridge that wants to show or drop it decides,
                // and an empty Content with finish=length is explainable only when this is visible.
                Reasoning = string.IsNullOrEmpty(message?.ReasoningContent) ? null : message.ReasoningContent,
                Usage = response.Usage is { } usage
                    ? new ChatTokenUsage
                    {
                        PromptTokens = usage.PromptTokens,
                        CompletionTokens = usage.CompletionTokens,
                        TotalTokens = usage.TotalTokens,
                    }
                    : null,
                Timings = response.Timings is { } timings ? ToGenerationTimings(timings) : null,
                Logprobs = ToTokenLogprobs(choice?.Logprobs?.Content)
            };
        }
        finally
        {
            _concurrencyLimiter.Release();
        }
    }

    internal static GenerationTimings ToGenerationTimings(LlamaServerTimings t) => new()
    {
        CachedPromptTokens = t.CacheN,
        PromptTokensEvaluated = t.PromptN,
        PromptDuration = t.PromptMs is { } promptMs ? TimeSpan.FromMilliseconds(promptMs) : null,
        PromptTokensPerSecond = t.PromptPerSecond,
        CompletionDuration = t.PredictedMs is { } predictedMs ? TimeSpan.FromMilliseconds(predictedMs) : null,
        CompletionTokensPerSecond = t.PredictedPerSecond,
    };

    internal static PromptProgress ToPromptProgress(LlamaPromptProgress p) => new()
    {
        TotalTokens = p.Total,
        CachedTokens = p.Cache,
        ProcessedTokens = p.Processed,
        Elapsed = TimeSpan.FromMilliseconds(p.TimeMs),
    };

    /// <summary>
    /// The server's per-token log probabilities as the public <see cref="TokenLogprob"/>; null when there are none.
    /// </summary>
    internal static IReadOnlyList<TokenLogprob>? ToTokenLogprobs(IReadOnlyList<LlamaTokenLogprob>? logprobs)
    {
        if (logprobs is not { Count: > 0 })
            return null;

        return logprobs.Select(t => new TokenLogprob
        {
            TokenId = t.Id,
            Token = t.Token ?? string.Empty,
            Logprob = t.Logprob,
            Bytes = ToBytes(t.Bytes),
            TopLogprobs = t.TopLogprobs?.Select(c => new TopLogprob
            {
                TokenId = c.Id,
                Token = c.Token ?? string.Empty,
                Logprob = c.Logprob,
                Bytes = ToBytes(c.Bytes),
            }).ToList() ?? [],
        }).ToList();

        static byte[]? ToBytes(List<int>? bytes) => bytes?.Select(b => (byte)b).ToArray();
    }
}
