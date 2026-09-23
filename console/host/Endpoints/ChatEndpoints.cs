using System.Runtime.CompilerServices;
using LMSupply.Generator;
using LMSupply.Generator.Abstractions;
using LMSupply.Generator.Models;
using LMSupply.Console.Host.Infrastructure;
using LMSupply.Console.Host.Infrastructure.ToolCalling;
using LMSupply.Console.Host.Models.OpenAI;
using ToolCall = LMSupply.Console.Host.Models.OpenAI.ToolCall;
using LMSupply.Console.Host.Services;

namespace LMSupply.Console.Host.Endpoints;

public static class ChatEndpoints
{
    public static void MapChatEndpoints(this WebApplication app)
    {
        var group = app.MapGroup("/v1/chat")
            .WithTags("Chat")
            ;

        // POST /v1/chat/completions - OpenAI compatible
        group.MapPost("/completions", async (ChatCompletionRequest request, HttpContext context, ModelManagerService manager) =>
        {
            var ct = context.RequestAborted;

            // Validate messages before model loading
            if (request.Messages == null || request.Messages.Count == 0)
            {
                context.Response.StatusCode = 400;
                await context.Response.WriteAsJsonAsync(new ErrorResponse
                {
                    Error = new ErrorDetail { Message = "'messages' field is required and must not be empty", Type = "invalid_request_error" }
                }, ct);
                return;
            }

            // Resolve max tokens
            var maxTokens = request.MaxCompletionTokens ?? request.MaxTokens ?? 2048;

            // Detect feature flags
            var toolsEnabled = ToolCallFormatter.ShouldEnableTools(request.ToolChoice, request.Tools);
            var hasImages = MultimodalHelper.HasImageContent(request.Messages);
            var isStrictJson = request.ResponseFormat?.Type == "json_schema" && request.ResponseFormat.JsonSchema?.Strict == true;

            // Handle streaming separately to avoid IResult after response started
            if (request.Stream)
            {
                try
                {
                    await using var scope = await manager.GetGeneratorAsync(request.Model, ct);
                    var messages = await BuildMessagesAsync(request, manager, toolsEnabled, hasImages, ct);
                    var options = BuildGenerationOptions(request, maxTokens);

                    if (request.DraftModel is { } draftModelId)
                    {
                        // Speculative decoding: draft model pre-generates candidate tokens verified by target
                        await using var draftScope = await manager.GetGeneratorAsync(draftModelId, ct);
                        var prompt = scope.Model.ChatFormatter.FormatPrompt(messages);
                        using var decoder = new SpeculativeDecoder(
                            new BorrowedGeneratorModel(draftScope.Model),
                            new BorrowedGeneratorModel(scope.Model));
                        var tokens = ToStringTokens(decoder.GenerateAsync(prompt, options, ct));
                        await SseHelper.StreamChatCompletionAsync(context, scope.Model.ModelId, tokens, ct);
                    }
                    else if (isStrictJson)
                    {
                        // Strict JSON: buffer full response, validate, retry up to 3 times, then stream result
                        var tokens = StreamValidatedJsonAsync(scope.Model, messages, options, request.ResponseFormat!, ct);
                        await SseHelper.StreamChatCompletionAsync(context, scope.Model.ModelId, tokens, ct);
                    }
                    else
                    {
                        var tokens = scope.Model.GenerateChatAsync(messages, options, ct);
                        await SseHelper.StreamChatCompletionAsync(context, scope.Model.ModelId, tokens, ct);
                    }
                }
                catch (Exception ex)
                {
                    // If response hasn't started, we can still write error
                    if (!context.Response.HasStarted)
                    {
                        context.Response.StatusCode = 500;
                        await context.Response.WriteAsJsonAsync(new ErrorResponse
                        {
                            Error = new ErrorDetail { Message = FormatError(ex), Type = "internal_error" }
                        }, ct);
                    }
                }
                return;
            }

            // Non-streaming response
            try
            {
                await using var scope = await manager.GetGeneratorAsync(request.Model, ct);
                var messages = await BuildMessagesAsync(request, manager, toolsEnabled, hasImages, ct);
                var options = BuildGenerationOptions(request, maxTokens);

                var id = ApiHelper.GenerateId("chatcmpl");
                var n = Math.Max(1, request.N ?? 1);
                var choices = new List<ChatCompletionChoice>();
                var aggregatedUsage = new Usage();

                if (request.DraftModel is { } draftModelId)
                {
                    // Speculative decoding path (non-streaming)
                    await using var draftScope = await manager.GetGeneratorAsync(draftModelId, ct);
                    var prompt = scope.Model.ChatFormatter.FormatPrompt(messages);
                    using var decoder = new SpeculativeDecoder(
                        new BorrowedGeneratorModel(draftScope.Model),
                        new BorrowedGeneratorModel(scope.Model));

                    for (var i = 0; i < n; i++)
                    {
                        var specResult = await decoder.GenerateCompleteAsync(prompt, options, ct);
                        var result = new GenerationResult(
                            specResult.Text,
                            new TokenUsage(TokenUsage.EstimateTokens(prompt), TokenUsage.EstimateTokens(specResult.Text)));

                        choices.Add(new ChatCompletionChoice
                        {
                            Index = i,
                            Message = new ChatCompletionResponseMessage { Role = "assistant", Content = result.Content },
                            FinishReason = "stop"
                        });

                        aggregatedUsage = i == 0
                            ? new Usage { PromptTokens = result.Usage.PromptTokens, CompletionTokens = result.Usage.CompletionTokens, TotalTokens = result.Usage.TotalTokens }
                            : new Usage { PromptTokens = aggregatedUsage.PromptTokens, CompletionTokens = aggregatedUsage.CompletionTokens + result.Usage.CompletionTokens, TotalTokens = aggregatedUsage.PromptTokens + aggregatedUsage.CompletionTokens + result.Usage.CompletionTokens };
                    }
                }
                else
                {
                    for (var i = 0; i < n; i++)
                    {
                        // For strict JSON schema, retry up to 3 times on invalid JSON
                        var maxAttempts = isStrictJson ? 3 : 1;
                        GenerationResult result = default;
                        for (var attempt = 0; attempt < maxAttempts; attempt++)
                        {
                            result = await scope.Model.GenerateChatWithUsageAsync(messages, options, ct);
                            if (ResponseFormatHelper.ValidateJson(result.Content, request.ResponseFormat))
                                break;
                        }

                        // Parse tool calls if tools are enabled
                        IReadOnlyList<ToolCall>? toolCalls = null;
                        string? content = result.Content;
                        string finishReason = "stop";

                        if (toolsEnabled)
                        {
                            toolCalls = ToolCallParser.TryParse(result.Content);
                            if (toolCalls != null)
                            {
                                content = null; // per OpenAI spec: content is null when tool_calls present
                                finishReason = "tool_calls";
                            }
                        }

                        choices.Add(new ChatCompletionChoice
                        {
                            Index = i,
                            Message = new ChatCompletionResponseMessage
                            {
                                Role = "assistant",
                                Content = content,
                                ToolCalls = toolCalls
                            },
                            FinishReason = finishReason
                        });

                        // Aggregate usage (prompt tokens only counted once for first choice)
                        if (i == 0)
                        {
                            aggregatedUsage = new Usage
                            {
                                PromptTokens = result.Usage.PromptTokens,
                                CompletionTokens = result.Usage.CompletionTokens,
                                TotalTokens = result.Usage.TotalTokens
                            };
                        }
                        else
                        {
                            aggregatedUsage = new Usage
                            {
                                PromptTokens = aggregatedUsage.PromptTokens,
                                CompletionTokens = aggregatedUsage.CompletionTokens + result.Usage.CompletionTokens,
                                TotalTokens = aggregatedUsage.PromptTokens + aggregatedUsage.CompletionTokens + result.Usage.CompletionTokens
                            };
                        }
                    }
                }

                await context.Response.WriteAsJsonAsync(new ChatCompletionResponse
                {
                    Id = id,
                    Model = scope.Model.ModelId,
                    SystemFingerprint = $"fp_{scope.Model.ModelId.GetHashCode():x8}",
                    Choices = choices,
                    Usage = aggregatedUsage
                }, ct);
            }
            catch (Exception ex)
            {
                context.Response.StatusCode = 500;
                await context.Response.WriteAsJsonAsync(new ErrorResponse
                {
                    Error = new ErrorDetail { Message = FormatError(ex), Type = "internal_error" }
                }, ct);
            }
        })
        .WithName("CreateChatCompletion")
        .WithSummary("Create a chat completion (OpenAI compatible)")
        .WithDescription("Creates a model response for the given chat conversation. Compatible with OpenAI's chat completions API.")
        .Produces<ChatCompletionResponse>()
        .Produces<ErrorResponse>(400)
        .Produces<ErrorResponse>(500);
    }

    private static GenerationOptions BuildGenerationOptions(ChatCompletionRequest request, int maxTokens) =>
        new()
        {
            MaxTokens = maxTokens,
            Temperature = request.Temperature ?? 0.7f,
            TopP = request.TopP ?? 0.9f,
            FrequencyPenalty = request.FrequencyPenalty ?? 0f,
            PresencePenalty = request.PresencePenalty ?? 0f,
            Seed = request.Seed ?? -1,
            StopSequences = request.Stop?.ToList(),
            JsonSchema = request.ResponseFormat?.Type == "json_schema" &&
                         request.ResponseFormat.JsonSchema?.Schema.HasValue == true
                ? System.Text.Json.JsonSerializer.Serialize(request.ResponseFormat.JsonSchema.Schema.Value)
                : null
        };

    /// <summary>
    /// Buffers complete generation, validates JSON, retries up to 3 times, then yields result as a single token.
    /// Used for streaming strict json_schema responses.
    /// </summary>
    private static async IAsyncEnumerable<string> StreamValidatedJsonAsync(
        IGeneratorModel model,
        IEnumerable<ChatMessage> messages,
        GenerationOptions options,
        ResponseFormat format,
        [EnumeratorCancellation] CancellationToken ct = default)
    {
        GenerationResult result = default;
        for (var attempt = 0; attempt < 3; attempt++)
        {
            result = await model.GenerateChatWithUsageAsync(messages, options, ct);
            if (ResponseFormatHelper.ValidateJson(result.Content, format)) break;
        }
        yield return result.Content;
    }

    /// <summary>
    /// Converts IAsyncEnumerable&lt;SpeculativeToken&gt; to IAsyncEnumerable&lt;string&gt;.
    /// </summary>
    private static async IAsyncEnumerable<string> ToStringTokens(
        IAsyncEnumerable<SpeculativeToken> source,
        [EnumeratorCancellation] CancellationToken ct = default)
    {
        await foreach (var t in source.WithCancellation(ct))
            yield return t.Token;
    }

    /// <summary>
    /// Builds the ChatMessage list from the request, injecting tool schemas and JSON instructions into the system prompt.
    /// </summary>
    private static async Task<IEnumerable<ChatMessage>> BuildMessagesAsync(
        ChatCompletionRequest request,
        ModelManagerService manager,
        bool toolsEnabled,
        bool hasImages,
        CancellationToken ct)
    {
        // Build JSON instruction if response_format is set
        var jsonInstruction = ResponseFormatHelper.BuildJsonInstruction(request.ResponseFormat);

        // Find the existing system message text (first system message)
        string? existingSystem = null;
        foreach (var m in request.Messages)
        {
            if (string.Equals(m.Role, "system", StringComparison.OrdinalIgnoreCase))
            {
                existingSystem = m.GetTextContent();
                break;
            }
        }

        // Build the effective system prompt: tool schema + JSON instruction injected
        string? effectiveSystem = existingSystem;
        if (toolsEnabled && request.Tools != null)
            effectiveSystem = ToolCallInjector.BuildSystemPrompt(effectiveSystem, request.Tools);
        if (jsonInstruction != null)
            effectiveSystem = string.IsNullOrWhiteSpace(effectiveSystem) ? jsonInstruction : effectiveSystem + "\n\n" + jsonInstruction;

        var result = new List<ChatMessage>();
        var systemHandled = false;

        foreach (var m in request.Messages)
        {
            var role = m.Role.ToLowerInvariant();

            if (role == "system")
            {
                if (!systemHandled)
                {
                    // Use the effective system prompt (with injections)
                    if (!string.IsNullOrWhiteSpace(effectiveSystem))
                        result.Add(new ChatMessage(ChatRole.System, effectiveSystem));
                    systemHandled = true;
                }
                // Skip additional system messages (collapsed into first)
                continue;
            }

            if (role == "tool")
            {
                // Tool result messages — represent as user messages with tool result context
                var toolContent = m.GetTextContent();
                var toolId = m.ToolCallId ?? "unknown";
                result.Add(new ChatMessage(ChatRole.User, $"[Tool result for {toolId}]: {toolContent}"));
                continue;
            }

            // Extract text content (with optional image captioning for multimodal)
            string text;
            if (hasImages && m.Content.HasValue && m.Content.Value.ValueKind == System.Text.Json.JsonValueKind.Array)
                text = await MultimodalHelper.ExtractTextAsync(m.Content.Value, manager, ct);
            else
                text = m.GetTextContent();

            var chatRole = role switch
            {
                "user" => ChatRole.User,
                "assistant" => ChatRole.Assistant,
                _ => ChatRole.User // fallback for unknown roles
            };

            result.Add(new ChatMessage(chatRole, text));
        }

        // If no system message was found but we have injections to add
        if (!systemHandled && !string.IsNullOrWhiteSpace(effectiveSystem))
            result.Insert(0, new ChatMessage(ChatRole.System, effectiveSystem));

        return result;
    }

    private static string FormatError(Exception ex)
    {
        var msg = ex.Message;

        if (msg.Contains("Could not allocate") && msg.Contains("key-value cache"))
            return "GPU memory insufficient for this model's context length. Try a smaller model or reduce max_tokens.";

        if (msg.Contains("out of memory", StringComparison.OrdinalIgnoreCase))
            return $"Out of memory during inference. Try a smaller model. ({msg})";

        return msg;
    }

    /// <summary>
    /// Non-owning wrapper for IGeneratorModel that prevents the underlying model from being disposed.
    /// Used to safely pass borrowed models to SpeculativeDecoder which would otherwise take ownership.
    /// </summary>
    private sealed class BorrowedGeneratorModel(IGeneratorModel inner) : IGeneratorModel
    {
        public string ModelId => inner.ModelId;
        public int MaxContextLength => inner.MaxContextLength;
        public IChatFormatter ChatFormatter => inner.ChatFormatter;
        public bool IsGpuActive => inner.IsGpuActive;
        public IReadOnlyList<string> ActiveProviders => inner.ActiveProviders;
        public ExecutionProvider RequestedProvider => inner.RequestedProvider;
        public long? EstimatedMemoryBytes => inner.EstimatedMemoryBytes;
        public GeneratorModelInfo GetModelInfo() => inner.GetModelInfo();

        public IAsyncEnumerable<string> GenerateAsync(string prompt, GenerationOptions? options = null, CancellationToken ct = default)
            => inner.GenerateAsync(prompt, options, ct);

        public IAsyncEnumerable<string> GenerateChatAsync(IEnumerable<ChatMessage> messages, GenerationOptions? options = null, CancellationToken ct = default)
            => inner.GenerateChatAsync(messages, options, ct);

        public Task<string> GenerateCompleteAsync(string prompt, GenerationOptions? options = null, CancellationToken ct = default)
            => inner.GenerateCompleteAsync(prompt, options, ct);

        public Task<GenerationResult> GenerateCompleteResultAsync(string prompt, GenerationOptions? options = null, CancellationToken ct = default)
            => inner.GenerateCompleteResultAsync(prompt, options, ct);

        public Task<string> GenerateChatCompleteAsync(IEnumerable<ChatMessage> messages, GenerationOptions? options = null, CancellationToken ct = default)
            => inner.GenerateChatCompleteAsync(messages, options, ct);

        public Task WarmupAsync(CancellationToken ct = default) => inner.WarmupAsync(ct);

        public Task<ChatCompletionResult> GenerateChatWithToolsAsync(IEnumerable<ChatMessage> messages, GenerationOptions? options = null, CancellationToken ct = default)
            => inner.GenerateChatWithToolsAsync(messages, options, ct);

        public IAsyncEnumerable<ChatStreamChunk> GenerateChatStreamAsync(IEnumerable<ChatMessage> messages, GenerationOptions? options = null, CancellationToken ct = default)
            => inner.GenerateChatStreamAsync(messages, options, ct);

        public Task<int> CountTokensAsync(string text, CancellationToken ct = default)
            => inner.CountTokensAsync(text, ct);

        public Task<int> CountTokensAsync(IEnumerable<ChatMessage> messages, CancellationToken ct = default)
            => inner.CountTokensAsync(messages, ct);

        public ValueTask DisposeAsync() => ValueTask.CompletedTask; // Does NOT dispose the underlying model
    }
}
