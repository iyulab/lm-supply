using System.Text.Json;
using LMSupply.Generator.Abstractions;
using Microsoft.Extensions.AI;
using AIChatMessage = Microsoft.Extensions.AI.ChatMessage;
using AIChatRole = Microsoft.Extensions.AI.ChatRole;
using LmChatMessage = LMSupply.Generator.Models.ChatMessage;
using LmChatRole = LMSupply.Generator.Models.ChatRole;
using LmChatToolCall = LMSupply.Generator.Models.ChatToolCall;
using LmChatToolDefinition = LMSupply.Generator.Models.ChatToolDefinition;
using LmContentPart = LMSupply.Generator.Models.ContentPart;
using LmGenerationOptions = LMSupply.Generator.Models.GenerationOptions;
using LmImageContentPart = LMSupply.Generator.Models.ImageContentPart;
using LmTextContentPart = LMSupply.Generator.Models.TextContentPart;
using LmThinkingMode = LMSupply.Generator.Models.ThinkingMode;
using LmToolChoice = LMSupply.Generator.Models.ToolChoice;

namespace LMSupply.Generator;

/// <summary>
/// A loaded <see cref="ITextGenerator"/> as a Microsoft.Extensions.AI <see cref="IChatClient"/>, so a library that takes
/// the standard contract — a pipeline built with <c>ChatClientBuilder</c>, an agent framework — uses a local model with no
/// adapter of its own. Create it with <see cref="TextGeneratorChatClientExtensions.AsChatClient"/>.
/// </summary>
/// <remarks>
/// <para>
/// The client does not own the generator: disposing it leaves the model loaded, and the model is disposed by whoever
/// loaded it. <see cref="ChatOptions.ModelId"/>, when set, must name this model: there is no other model to route to.
/// </para>
/// <para>
/// <b>Messages.</b> <see cref="ChatOptions.Instructions"/> becomes a leading system message. Text is carried as text and
/// images (<see cref="DataContent"/> or <see cref="UriContent"/> with an <c>image/*</c> media type) as image parts, for a
/// vision model. An assistant message's <see cref="FunctionCallContent"/> becomes its tool calls, and a
/// <see cref="FunctionResultContent"/> becomes a tool message: a text result as it is, a structured one block by block —
/// text as text, anything else as a note naming what was left out, since the model reads the result as text.
/// </para>
/// <para>
/// <b>Options.</b> <see cref="ChatOptions.MaxOutputTokens"/> is the cap on generated tokens
/// (<see cref="LmGenerationOptions.MaxNewTokens"/>); temperature, top-p, top-k, the frequency and presence penalties, the
/// seed and the stop sequences map to their namesakes. <see cref="ChatOptions.Reasoning"/> drives
/// <see cref="LmGenerationOptions.Thinking"/>: an effort of <see cref="ReasoningEffort.None"/> turns thinking off, any
/// other effort turns it on, and no reasoning options keep the model's default; the reasoning the model writes comes back
/// as <see cref="TextReasoningContent"/> unless <see cref="ReasoningOptions.Output"/> is
/// <see cref="ReasoningOutput.None"/>. Every <see cref="AIFunctionDeclaration"/> in <see cref="ChatOptions.Tools"/> is
/// offered to the model (the client never invokes a tool — wrap it in a function-invoking client for that), with
/// <see cref="ChatOptions.ToolMode"/> as the tool choice. A JSON <see cref="ChatOptions.ResponseFormat"/> constrains the
/// output to its schema (to a JSON object when it has none; llama-server backend). Options with no standard name — the
/// repetition penalty, min-p, a grammar — come from <see cref="ChatOptions.RawRepresentationFactory"/> returning a
/// <see cref="LmGenerationOptions"/>, which the standard options are then applied over.
/// </para>
/// <para>
/// <b>Responses.</b> A response carries the model id, the finish reason (stop, tool calls or length) and, where the
/// backend counts them (llama-server; not ONNX), the token usage with the prompt tokens served from the server's cache.
/// A streamed response yields text and reasoning as they come and the tool calls, assembled, with the final update, which
/// also carries the usage. Each response and update keeps the library's own result as its raw representation. A tool
/// call the model wrote without a function name is left out; one without an id is given a <c>call_</c> id, so a
/// function-invoking client can pair its result with it; arguments that are not a JSON object come back as no
/// arguments.
/// </para>
/// <para>
/// <b>Services.</b> <see cref="GetService"/> hands out the <see cref="ChatClientMetadata"/> and the generator under any
/// type it implements — <see cref="ITextGenerator"/>, or <see cref="IGeneratorModel"/> for a loaded model, which tells
/// a caller sizing a token budget the model's <see cref="IGeneratorModel.MaxContextLength"/>.
/// </para>
/// </remarks>
public sealed class LocalChatClient : IChatClient
{
    // The standard options (their type information resolves without reflection), written compact: what is serialized
    // here goes into the prompt.
    private static readonly JsonSerializerOptions Json = Compact();

    private readonly ITextGenerator _generator;

    internal LocalChatClient(ITextGenerator generator)
    {
        ArgumentNullException.ThrowIfNull(generator);
        _generator = generator;
        Metadata = new ChatClientMetadata(providerName: "LMSupply", providerUri: null, defaultModelId: generator.ModelId);
    }

    /// <summary>The generator this client answers with.</summary>
    public ITextGenerator Generator => _generator;

    /// <summary>Provider <c>"LMSupply"</c> and the model id.</summary>
    public ChatClientMetadata Metadata { get; }

    /// <inheritdoc />
    public async Task<ChatResponse> GetResponseAsync(
        IEnumerable<AIChatMessage> messages,
        ChatOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(messages);
        var request = ToMessages(messages, options);
        var generation = ToGenerationOptions(options);

        var result = await _generator.GenerateChatWithToolsAsync(request, generation, cancellationToken).ConfigureAwait(false);

        var contents = new List<AIContent>();
        if (CarriesReasoning(options) && !string.IsNullOrEmpty(result.Reasoning))
            contents.Add(new TextReasoningContent(result.Reasoning));
        if (!string.IsNullOrEmpty(result.Content))
            contents.Add(new TextContent(result.Content));
        foreach (var call in result.ToolCalls ?? [])
        {
            if (ToCallContent(call.Id, call.FunctionName, call.Arguments) is { } content)
                contents.Add(content);
        }

        return new ChatResponse(new AIChatMessage(AIChatRole.Assistant, contents))
        {
            FinishReason = ToFinishReason(result.FinishReason),
            ModelId = _generator.ModelId,
            Usage = result.Usage is { } usage ? ToUsage(usage, result.Timings) : null,
            RawRepresentation = result,
        };
    }

    /// <inheritdoc />
    public async IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(
        IEnumerable<AIChatMessage> messages,
        ChatOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(messages);
        var request = ToMessages(messages, options);
        var generation = ToGenerationOptions(options);
        var carriesReasoning = CarriesReasoning(options);
        var modelId = _generator.ModelId;

        // Tool calls arrive as deltas, by index; they are handed over whole, with the final update.
        SortedDictionary<int, (string? Id, string? Name, StringBuilder Arguments)>? calls = null;

        await foreach (var chunk in _generator.GenerateChatStreamAsync(request, generation, cancellationToken).ConfigureAwait(false))
        {
            var contents = new List<AIContent>();
            if (carriesReasoning && !string.IsNullOrEmpty(chunk.ReasoningDelta))
                contents.Add(new TextReasoningContent(chunk.ReasoningDelta));
            if (!string.IsNullOrEmpty(chunk.Text))
                contents.Add(new TextContent(chunk.Text));

            foreach (var delta in chunk.ToolCalls ?? [])
            {
                calls ??= [];
                var call = calls.TryGetValue(delta.Index, out var known) ? known : (null, null, new StringBuilder());
                calls[delta.Index] = (delta.Id ?? call.Id, delta.Name ?? call.Name, call.Arguments.Append(delta.Arguments));
            }

            if (chunk.FinishReason is not null)
            {
                contents.AddRange(TakeCalls(ref calls));
                if (chunk.Usage is { } usage)
                    contents.Add(new UsageContent(ToUsage(usage, chunk.Timings)));
            }

            if (contents.Count == 0 && chunk.FinishReason is null)
                continue;

            yield return new ChatResponseUpdate(AIChatRole.Assistant, contents)
            {
                ModelId = modelId,
                FinishReason = ToFinishReason(chunk.FinishReason),
                RawRepresentation = chunk,
            };
        }

        // A stream that ended without a finish reason still hands over the calls it carried.
        if (calls is { Count: > 0 })
        {
            yield return new ChatResponseUpdate(AIChatRole.Assistant, TakeCalls(ref calls))
            {
                ModelId = modelId,
                FinishReason = ChatFinishReason.ToolCalls,
            };
        }
    }

    /// <inheritdoc />
    public object? GetService(Type serviceType, object? serviceKey = null)
    {
        ArgumentNullException.ThrowIfNull(serviceType);
        if (serviceKey is not null)
            return null;

        if (serviceType == typeof(ChatClientMetadata))
            return Metadata;
        if (serviceType.IsInstanceOfType(this))
            return this;
        if (serviceType.IsInstanceOfType(_generator))
            return _generator;
        return null;
    }

    /// <summary>Does nothing: the client does not own the generator (see the remarks on the type).</summary>
    public void Dispose()
    {
    }

    private List<LmChatMessage> ToMessages(IEnumerable<AIChatMessage> messages, ChatOptions? options)
    {
        if (options?.ModelId is { } requestedModel &&
            !string.Equals(requestedModel, _generator.ModelId, StringComparison.OrdinalIgnoreCase))
        {
            throw new ArgumentException(
                $"This client answers with model '{_generator.ModelId}'; the request names '{requestedModel}'. " +
                "Load that model and use its own client.", nameof(options));
        }

        var converted = new List<LmChatMessage>();
        if (!string.IsNullOrEmpty(options?.Instructions))
            converted.Add(LmChatMessage.System(options.Instructions));

        foreach (var message in messages)
        {
            var results = message.Contents.OfType<FunctionResultContent>().ToList();
            if (results.Count > 0)
            {
                foreach (var result in results)
                    converted.Add(LmChatMessage.ToolResult(result.CallId, ResultText(result.Result)));
                continue;
            }

            var calls = message.Contents.OfType<FunctionCallContent>()
                .Select(call => new LmChatToolCall(call.CallId, call.Name, SerializeArguments(call.Arguments)))
                .ToList();
            var role = ToRole(message.Role);
            if (calls.Count > 0)
            {
                converted.Add(new LmChatMessage(role, message.Text) { ToolCalls = calls });
                continue;
            }

            var images = message.Contents.Select(ImageUrl).Where(url => url is not null).ToList();
            if (images.Count == 0)
            {
                converted.Add(new LmChatMessage(role, message.Text));
                continue;
            }

            // Text and images in the order they were given.
            var parts = new List<LmContentPart>();
            foreach (var content in message.Contents)
            {
                if (content is TextContent { Text: { Length: > 0 } text })
                    parts.Add(new LmTextContentPart(text));
                else if (ImageUrl(content) is { } url)
                    parts.Add(new LmImageContentPart { Url = url });
            }

            converted.Add(new LmChatMessage(role, message.Text) { ContentParts = parts });
        }

        return converted;
    }

    private static LmChatRole ToRole(AIChatRole role) =>
        role == AIChatRole.System ? LmChatRole.System
        : role == AIChatRole.Assistant ? LmChatRole.Assistant
        : role == AIChatRole.Tool ? LmChatRole.Tool
        : LmChatRole.User;

    private static string? ImageUrl(AIContent content) => content switch
    {
        DataContent data when data.HasTopLevelMediaType("image") => data.Uri,
        UriContent uri when uri.HasTopLevelMediaType("image") => uri.Uri.ToString(),
        _ => null,
    };

    /// <summary>A tool's result as the text the model reads.</summary>
    internal static string ResultText(object? result) => result switch
    {
        null => string.Empty,
        string text => text,
        AIContent content => ContentsText([content]),
        IEnumerable<AIContent> contents => ContentsText(contents),
        JsonElement { ValueKind: JsonValueKind.String } element => element.GetString() ?? string.Empty,
        JsonElement element => element.GetRawText(),
        _ => JsonSerializer.Serialize(result, Json.GetTypeInfo(result.GetType())),
    };

    private static string ContentsText(IEnumerable<AIContent> contents) =>
        string.Join("\n", contents.Select(content => content switch
        {
            TextContent text => text.Text,
            DataContent data => $"[{data.MediaType} content left out ({data.Data.Length} bytes): the model reads text only]",
            _ => $"[{content.GetType().Name} left out: the model reads text only]",
        }));

    private LmGenerationOptions ToGenerationOptions(ChatOptions? options)
    {
        // Options without a standard name come from the raw representation; the standard ones are applied over them.
        var generation = options?.RawRepresentationFactory?.Invoke(this) is LmGenerationOptions raw
            ? raw.Clone()
            : new LmGenerationOptions();
        generation.ExtractReasoningTokens = CarriesReasoning(options);
        if (options is null)
            return generation;

        if (options.MaxOutputTokens is { } maxOutputTokens)
            generation.MaxNewTokens = maxOutputTokens;
        if (options.Temperature is { } temperature)
            generation.Temperature = temperature;
        if (options.TopP is { } topP)
            generation.TopP = topP;
        if (options.TopK is { } topK)
            generation.TopK = topK;
        if (options.FrequencyPenalty is { } frequencyPenalty)
            generation.FrequencyPenalty = frequencyPenalty;
        if (options.PresencePenalty is { } presencePenalty)
            generation.PresencePenalty = presencePenalty;
        if (options.Seed is { } seed)
            generation.Seed = unchecked((int)seed);
        if (options.StopSequences is { Count: > 0 } stops)
            generation.StopSequences = [.. stops];

        if (options.Reasoning is { } reasoning)
        {
            generation.Thinking = reasoning.Effort switch
            {
                null => generation.Thinking,
                ReasoningEffort.None => LmThinkingMode.Off,
                _ => LmThinkingMode.On,
            };
        }

        var tools = options.Tools?.OfType<AIFunctionDeclaration>()
            .Select(tool => new LmChatToolDefinition(
                tool.Name,
                tool.Description,
                tool.JsonSchema.ValueKind == JsonValueKind.Undefined ? null : tool.JsonSchema))
            .ToList();
        if (tools is { Count: > 0 })
        {
            generation.Tools = tools;
            generation.ToolChoice = options.ToolMode switch
            {
                NoneChatToolMode => LmToolChoice.None,
                RequiredChatToolMode { RequiredFunctionName: { } name } => LmToolChoice.Function(name),
                RequiredChatToolMode => LmToolChoice.Required,
                AutoChatToolMode => LmToolChoice.Auto,
                _ => generation.ToolChoice,
            };
        }

        if (options.ResponseFormat is ChatResponseFormatJson json)
            generation.JsonSchema = json.Schema is { } schema ? schema.GetRawText() : """{"type":"object"}""";

        return generation;
    }

    private static bool CarriesReasoning(ChatOptions? options) =>
        options?.Reasoning?.Output is not ReasoningOutput.None;

    private static List<AIContent> TakeCalls(ref SortedDictionary<int, (string? Id, string? Name, StringBuilder Arguments)>? calls)
    {
        var contents = new List<AIContent>();
        if (calls is null)
            return contents;

        foreach (var (id, name, arguments) in calls.Values)
        {
            if (ToCallContent(id, name, arguments.ToString()) is { } content)
                contents.Add(content);
        }

        calls = null;
        return contents;
    }

    // A call that never named its function cannot be invoked and is left out. A call without an id gets one, since a
    // function-invoking client pairs each result with its call by id.
    private static FunctionCallContent? ToCallContent(string? id, string? name, string? arguments) =>
        string.IsNullOrEmpty(name)
            ? null
            : new FunctionCallContent(string.IsNullOrEmpty(id) ? $"call_{Guid.NewGuid():N}" : id, name, ParseArguments(arguments));

    private static string SerializeArguments(IDictionary<string, object?>? arguments) =>
        arguments is null
            ? "{}"
            : JsonSerializer.Serialize(arguments, Json.GetTypeInfo(typeof(IDictionary<string, object?>)));

    // Arguments the model wrote that are not a JSON object are left out rather than thrown on: the caller sees the call
    // with no arguments, which a function-invoking client reports back to the model as a failed call.
    private static Dictionary<string, object?>? ParseArguments(string? json)
    {
        if (string.IsNullOrWhiteSpace(json))
            return null;
        try
        {
            return (Dictionary<string, object?>?)JsonSerializer.Deserialize(
                json, Json.GetTypeInfo(typeof(Dictionary<string, object?>)));
        }
        catch (JsonException)
        {
            return null;
        }
    }

    private static JsonSerializerOptions Compact()
    {
        var options = new JsonSerializerOptions(AIJsonUtilities.DefaultOptions) { WriteIndented = false };
        options.MakeReadOnly();
        return options;
    }

    private static UsageDetails ToUsage(Models.ChatTokenUsage usage, Models.GenerationTimings? timings) => new()
    {
        InputTokenCount = usage.PromptTokens,
        OutputTokenCount = usage.CompletionTokens,
        TotalTokenCount = usage.TotalTokens,
        CachedInputTokenCount = timings?.CachedPromptTokens,
    };

    private static ChatFinishReason? ToFinishReason(string? reason) => reason switch
    {
        "stop" => ChatFinishReason.Stop,
        "tool_calls" => ChatFinishReason.ToolCalls,
        "length" => ChatFinishReason.Length,
        "content_filter" => ChatFinishReason.ContentFilter,
        _ => null,
    };
}

/// <summary>Microsoft.Extensions.AI interop for <see cref="ITextGenerator"/>.</summary>
public static class TextGeneratorChatClientExtensions
{
    /// <summary>
    /// The generator as a Microsoft.Extensions.AI <see cref="IChatClient"/>. The client does not own the generator;
    /// dispose the generator when it is no longer used.
    /// </summary>
    /// <param name="generator">A loaded text generator.</param>
    public static LocalChatClient AsChatClient(this ITextGenerator generator) => new(generator);
}
