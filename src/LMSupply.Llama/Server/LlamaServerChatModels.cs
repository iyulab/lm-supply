using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.Json.Serialization;

namespace LMSupply.Llama.Server;

/// <summary>
/// Chat completion message.
/// </summary>
public sealed class ChatCompletionMessage
{
    [JsonPropertyName("role")]
    public required string Role { get; init; }

    [JsonPropertyName("content")]
    public string? Content { get; init; }

    [JsonPropertyName("tool_calls")]
    public List<ToolCallMessage>? ToolCalls { get; init; }

    [JsonPropertyName("tool_call_id")]
    public string? ToolCallId { get; init; }

    public static ChatCompletionMessage System(string content) => new() { Role = "system", Content = content };
    public static ChatCompletionMessage User(string content) => new() { Role = "user", Content = content };
    public static ChatCompletionMessage Assistant(string content) => new() { Role = "assistant", Content = content };
    public static ChatCompletionMessage Tool(string toolCallId, string content) => new() { Role = "tool", Content = content, ToolCallId = toolCallId };
}

/// <summary>
/// Options for chat completion.
/// </summary>
public sealed class ChatCompletionOptions
{
    public int MaxTokens { get; init; } = 256;
    public float Temperature { get; init; } = 0.7f;
    public float TopP { get; init; } = 0.9f;
    public int TopK { get; init; } = 50;
    public float MinP { get; init; } = 0.05f;
    public float RepeatPenalty { get; init; } = 1.1f;
    public float FrequencyPenalty { get; init; }
    public float PresencePenalty { get; init; }

    // Advanced anti-repetition (standard llama-server samplers; null = server default, omitted from request)
    public float? DryMultiplier { get; init; }
    public float? DryBase { get; init; }
    public int? DryAllowedLength { get; init; }
    public int? DryPenaltyLastN { get; init; }
    public int? RepeatLastN { get; init; }

    public int Seed { get; init; } = -1;
    public IReadOnlyList<string>? StopSequences { get; init; }

    /// <summary>
    /// Grammar constraint in GBNF format (Phase 3).
    /// </summary>
    public string? Grammar { get; init; }

    /// <summary>
    /// JSON schema for structured output (Phase 3).
    /// When set, output will be constrained to match this schema.
    /// </summary>
    public string? JsonSchema { get; init; }

    /// <summary>
    /// Tool definitions available for the model.
    /// When provided, the model may respond with tool_calls.
    /// </summary>
    public IReadOnlyList<LlamaToolDefinition>? Tools { get; init; }

    /// <summary>
    /// Controls whether/which tool the model must call. Null = auto (model decides). Only
    /// meaningful when <see cref="Tools"/> is set.
    /// </summary>
    public LlamaToolChoice? ToolChoice { get; init; }

    /// <summary>
    /// Thinking control forwarded to the GGUF chat template via <c>chat_template_kwargs.enable_thinking</c>.
    /// Null = omit (model template default: Qwen3 thinks, Gemma does not). False = suppress reasoning
    /// (direct answer); True = force reasoning on. Only applies to the chat path (/v1/chat/completions),
    /// which is where the template runs — the raw /completion path has no template to control.
    /// </summary>
    public bool? EnableThinking { get; init; }

    /// <summary>
    /// Per-token log probabilities to return: null (the default) returns none; 0 returns each generated token's own log
    /// probability; n returns it with the n most likely candidates at that position. The values are the model's raw
    /// next-token distribution (a softmax over the logits), taken before sampling: temperature, top-k/top-p/min-p,
    /// penalties, <c>logit_bias</c> and a grammar or JSON schema do not change them, though they still decide which
    /// token is sampled. Sent as <c>logprobs</c>/<c>top_logprobs</c>; cannot be combined with <see cref="Tools"/> on
    /// a streamed completion.
    /// </summary>
    public int? TopLogprobs { get; init; }
}

/// <summary>
/// Options for text completion.
/// </summary>
public sealed class CompletionOptions
{
    public int MaxTokens { get; init; } = 256;
    public float Temperature { get; init; } = 0.7f;
    public float TopP { get; init; } = 0.9f;
    public int TopK { get; init; } = 50;
    public float MinP { get; init; } = 0.05f;
    public float RepeatPenalty { get; init; } = 1.1f;
    public float FrequencyPenalty { get; init; }
    public float PresencePenalty { get; init; }

    // Advanced anti-repetition (standard llama-server samplers; null = server default, omitted from request)
    public float? DryMultiplier { get; init; }
    public float? DryBase { get; init; }
    public int? DryAllowedLength { get; init; }
    public int? DryPenaltyLastN { get; init; }
    public int? RepeatLastN { get; init; }

    public int Seed { get; init; } = -1;
    public IReadOnlyList<string>? StopSequences { get; init; }

    /// <summary>
    /// Grammar constraint in GBNF format (Phase 3).
    /// </summary>
    public string? Grammar { get; init; }

    /// <summary>
    /// JSON schema for structured output (Phase 3).
    /// When set, output will be constrained to match this schema.
    /// </summary>
    public string? JsonSchema { get; init; }

    /// <summary>
    /// Per-token log probabilities to return: null (the default) returns none; 0 returns each generated token's own log
    /// probability; n returns it with the n most likely candidates at that position. The values are the model's raw
    /// next-token distribution (a softmax over the logits), taken before sampling: temperature, top-k/top-p/min-p,
    /// penalties, <c>logit_bias</c> and a grammar or JSON schema do not change them, though they still decide which
    /// token is sampled. Sent as <c>n_probs</c>.
    /// </summary>
    public int? TopLogprobs { get; init; }
}

internal sealed class ChatCompletionRequest
{
    public List<ChatCompletionMessage>? Messages { get; set; }
    public ChatStreamOptions? StreamOptions { get; set; }
    public int? MaxTokens { get; set; }
    public float? Temperature { get; set; }
    public float? TopP { get; set; }
    public int? TopK { get; set; }
    public float? MinP { get; set; }
    public float? RepeatPenalty { get; set; }
    public float? FrequencyPenalty { get; set; }
    public float? PresencePenalty { get; set; }

    // Advanced anti-repetition (snake_case: dry_multiplier, dry_base, dry_allowed_length,
    // dry_penalty_last_n, repeat_last_n). Null -> omitted (WhenWritingNull) -> server default.
    public float? DryMultiplier { get; set; }
    public float? DryBase { get; set; }
    public int? DryAllowedLength { get; set; }
    public int? DryPenaltyLastN { get; set; }
    public int? RepeatLastN { get; set; }

    public int? Seed { get; set; }
    public bool Stream { get; set; }

    /// <summary>
    /// Asks a streamed completion for <c>prompt_progress</c> chunks while the server reads the prompt; a build that
    /// predates the option ignores it.
    /// </summary>
    public bool? ReturnProgress { get; set; }

    public List<string>? Stop { get; set; }

    /// <summary>
    /// Grammar constraint in GBNF format (Phase 3).
    /// </summary>
    public string? Grammar { get; set; }

    /// <summary>
    /// Structured-output constraint for the OpenAI-compatible chat endpoint. llama-server's
    /// <c>/v1/chat/completions</c> reads the schema from <c>response_format.json_schema.schema</c>
    /// (OpenAI form), NOT a root-level <c>json_schema</c> field — sending the latter (as a string) is
    /// rejected with HTTP 400. Built from <c>GenerationOptions.JsonSchema</c> in BuildChatRequest.
    /// </summary>
    [JsonPropertyName("response_format")]
    public ChatResponseFormat? ResponseFormat { get; set; }

    /// <summary>
    /// Tool definitions (OpenAI-compatible).
    /// </summary>
    public List<LlamaToolDefinition>? Tools { get; set; }

    /// <summary>
    /// OpenAI-compatible <c>tool_choice</c> — a bare string (<c>"none"</c>/<c>"required"</c>) or
    /// <c>{"type":"function","function":{"name":...}}</c>, or omitted entirely for the server's
    /// "auto" default. Built from <see cref="ChatCompletionOptions.ToolChoice"/> in BuildChatRequest
    /// via <see cref="LlamaServerClient.BuildToolChoiceNode"/>.
    /// </summary>
    public JsonNode? ToolChoice { get; set; }

    /// <summary>
    /// Re-use KV cache from previous request if possible.
    /// Reduces first token latency for prompts with common prefixes.
    /// </summary>
    public bool CachePrompt { get; set; } = true;

    /// <summary>
    /// Extra kwargs forwarded to the GGUF chat template by llama-server. Used to pass
    /// <c>enable_thinking</c> to thinking-aware templates (Qwen3) so a thinking-default-on model can
    /// be told to emit a direct answer instead of a reasoning block. Null = omit (model template
    /// default applies). Unknown to older llama-server builds, which ignore unrecognized fields.
    /// </summary>
    public Dictionary<string, object>? ChatTemplateKwargs { get; set; }

    /// <summary>Whether to return per-token log probabilities (OpenAI-compatible). Null -> omitted.</summary>
    public bool? Logprobs { get; set; }

    /// <summary>
    /// Candidates per token (OpenAI-compatible). llama-server maps it to <c>n_probs</c> and computes the values before
    /// sampling — its default <c>post_sampling_probs: false</c>, which is left unsent.
    /// </summary>
    public int? TopLogprobs { get; set; }
}

internal sealed class CompletionRequest
{
    public string? Prompt { get; set; }
    public int? NPredict { get; set; }
    public float? Temperature { get; set; }
    public float? TopP { get; set; }
    public int? TopK { get; set; }
    public float? MinP { get; set; }
    public float? RepeatPenalty { get; set; }
    public float? FrequencyPenalty { get; set; }
    public float? PresencePenalty { get; set; }

    // Advanced anti-repetition (snake_case: dry_multiplier, dry_base, dry_allowed_length,
    // dry_penalty_last_n, repeat_last_n). Null -> omitted (WhenWritingNull) -> server default.
    public float? DryMultiplier { get; set; }
    public float? DryBase { get; set; }
    public int? DryAllowedLength { get; set; }
    public int? DryPenaltyLastN { get; set; }
    public int? RepeatLastN { get; set; }

    public int? Seed { get; set; }
    public bool Stream { get; set; }
    public List<string>? Stop { get; set; }

    /// <summary>
    /// Grammar constraint in GBNF format (Phase 3).
    /// </summary>
    public string? Grammar { get; set; }

    /// <summary>
    /// JSON schema (as a JSON <b>object</b>) for structured output on the native <c>/completion</c>
    /// endpoint. Serialized to the root <c>json_schema</c> field, which llama-server expects to be an
    /// object — so <c>GenerationOptions.JsonSchema</c> (a JSON string) is parsed to a node via
    /// <see cref="LlamaServerClient.ParseStructuredSchema"/> before assignment.
    /// </summary>
    public JsonNode? JsonSchema { get; set; }

    /// <summary>
    /// Candidates per token whose log probabilities to return (<c>n_probs</c>); null or 0 returns none. Computed before
    /// sampling — the server's default <c>post_sampling_probs: false</c>, which is left unsent.
    /// </summary>
    public int? NProbs { get; set; }

    /// <summary>
    /// Re-use KV cache from previous request if possible.
    /// Reduces first token latency for prompts with common prefixes.
    /// </summary>
    public bool CachePrompt { get; set; } = true;
}

/// <summary>
/// OpenAI-compatible <c>response_format</c> for structured chat output. llama-server's
/// <c>/v1/chat/completions</c> reads the schema from <c>response_format.json_schema.schema</c>.
/// </summary>
internal sealed class ChatResponseFormat
{
    [JsonPropertyName("type")]
    public string Type { get; set; } = "json_schema";

    [JsonPropertyName("json_schema")]
    public ChatJsonSchema? JsonSchema { get; set; }
}

/// <summary>
/// The nested schema payload of <see cref="ChatResponseFormat"/> (OpenAI form:
/// <c>{ "name": ..., "schema": {...} }</c>).
/// </summary>
internal sealed class ChatJsonSchema
{
    [JsonPropertyName("name")]
    public string Name { get; set; } = "response";

    [JsonPropertyName("schema")]
    public JsonNode? Schema { get; set; }
}

internal sealed class ChatCompletionChunk
{
    public List<ChatCompletionChoice>? Choices { get; set; }

    /// <summary>OpenAI-compatible usage — on the last chunk when <c>stream_options.include_usage</c> is set.</summary>
    public ChatCompletionUsage? Usage { get; set; }

    /// <summary>llama.cpp's own accounting, on the last chunk of every build that has it.</summary>
    public LlamaServerTimings? Timings { get; set; }

    /// <summary>How far the server has read the prompt, on the chunks a request with <c>return_progress</c> gets before the first token.</summary>
    public LlamaPromptProgress? PromptProgress { get; set; }
}

/// <summary>
/// llama.cpp's <c>prompt_progress</c>: sent while the server reads the prompt of a streamed completion that set
/// <c>return_progress</c> — when it starts and after each batch it evaluates.
/// </summary>
public sealed class LlamaPromptProgress
{
    /// <summary>Tokens in the prompt (<c>total</c>).</summary>
    [JsonPropertyName("total")]
    public int Total { get; set; }

    /// <summary>Prompt tokens reused from the server's prompt cache (<c>cache</c>); they count as processed from the start.</summary>
    [JsonPropertyName("cache")]
    public int Cache { get; set; }

    /// <summary>Prompt tokens read so far, the cached ones included (<c>processed</c>).</summary>
    [JsonPropertyName("processed")]
    public int Processed { get; set; }

    /// <summary>Milliseconds since the server started reading the prompt (<c>time_ms</c>).</summary>
    [JsonPropertyName("time_ms")]
    public double TimeMs { get; set; }
}

internal sealed class ChatStreamOptions
{
    public bool IncludeUsage { get; set; }
}

/// <summary>
/// llama.cpp's <c>timings</c> object, as the server reports it. Prompt figures cover only the tokens evaluated for this
/// request (<c>prompt_n</c>), not those reused from the prompt cache (<c>cache_n</c>), so a rate derived from the
/// OpenAI-compatible <c>usage.prompt_tokens</c> is wrong whenever the cache hits — use the server's rates.
/// </summary>
public sealed class LlamaServerTimings
{
    /// <summary>Prompt tokens reused from the server's prompt cache (<c>cache_n</c>).</summary>
    [JsonPropertyName("cache_n")]
    public int? CacheN { get; set; }

    /// <summary>Prompt tokens evaluated for this request (<c>prompt_n</c>).</summary>
    [JsonPropertyName("prompt_n")]
    public int? PromptN { get; set; }

    /// <summary>Time spent evaluating the prompt, in milliseconds (<c>prompt_ms</c>).</summary>
    [JsonPropertyName("prompt_ms")]
    public double? PromptMs { get; set; }

    /// <summary>Prompt evaluation rate the server measured (<c>prompt_per_second</c>).</summary>
    [JsonPropertyName("prompt_per_second")]
    public double? PromptPerSecond { get; set; }

    /// <summary>Tokens generated, reasoning included (<c>predicted_n</c>).</summary>
    [JsonPropertyName("predicted_n")]
    public int? PredictedN { get; set; }

    /// <summary>Time spent generating, in milliseconds (<c>predicted_ms</c>).</summary>
    [JsonPropertyName("predicted_ms")]
    public double? PredictedMs { get; set; }

    /// <summary>Generation rate the server measured (<c>predicted_per_second</c>).</summary>
    [JsonPropertyName("predicted_per_second")]
    public double? PredictedPerSecond { get; set; }
}

internal sealed class ChatCompletionChoice
{
    public ChatCompletionDelta? Delta { get; set; }
    public string? FinishReason { get; set; }

    /// <summary>The log probabilities of the tokens this chunk carries, beside the delta (not inside it).</summary>
    public LlamaChoiceLogprobs? Logprobs { get; set; }
}

internal sealed class ChatCompletionDelta
{
    public string? Content { get; set; }

    [JsonPropertyName("reasoning_content")]
    public string? ReasoningContent { get; set; }

    [JsonPropertyName("tool_calls")]
    public List<ToolCallDelta>? ToolCalls { get; set; }
}

/// <summary>
/// Tool call delta in streaming response.
/// </summary>
public sealed class ToolCallDelta
{
    [JsonPropertyName("index")]
    public int Index { get; set; }

    [JsonPropertyName("id")]
    public string? Id { get; set; }

    [JsonPropertyName("type")]
    public string? Type { get; set; }

    [JsonPropertyName("function")]
    public FunctionCallDelta? Function { get; set; }
}

/// <summary>
/// Function call delta in streaming response.
/// </summary>
public sealed class FunctionCallDelta
{
    [JsonPropertyName("name")]
    public string? Name { get; set; }

    [JsonPropertyName("arguments")]
    public string? Arguments { get; set; }
}

internal sealed class CompletionChunk
{
    public string? Content { get; set; }
    public bool Stop { get; set; }

    /// <summary>On the final chunk: <c>eos</c>, <c>limit</c>, <c>word</c> or <c>none</c>.</summary>
    public string? StopType { get; set; }

    /// <summary>On the final chunk: tokens generated.</summary>
    public int? TokensPredicted { get; set; }

    /// <summary>On the final chunk: prompt tokens evaluated.</summary>
    public int? TokensEvaluated { get; set; }

    /// <summary>On the final chunk: llama.cpp's timings.</summary>
    public LlamaServerTimings? Timings { get; set; }

    /// <summary>The log probabilities of the tokens this chunk carries, when <c>n_probs</c> asked for them.</summary>
    public List<LlamaTokenLogprob>? CompletionProbabilities { get; set; }
}

/// <summary>
/// Structured streaming data from a raw text completion (<c>/completion</c>).
/// </summary>
public sealed class CompletionStreamData
{
    /// <summary>
    /// Text content delta.
    /// </summary>
    public string? TextDelta { get; init; }

    /// <summary>
    /// Finish reason (present only on the final chunk): <c>"length"</c> or <c>"stop"</c>, or <c>null</c> when the
    /// server did not say.
    /// </summary>
    public string? FinishReason { get; init; }

    /// <summary>Prompt tokens the server evaluated (final chunk), or null when it did not say.</summary>
    public int? PromptTokens { get; init; }

    /// <summary>Tokens the server generated (final chunk), or null when it did not say.</summary>
    public int? CompletionTokens { get; init; }

    /// <summary>The server's timings for the whole completion (final chunk), or null when it did not send them.</summary>
    public LlamaServerTimings? Timings { get; init; }

    /// <summary>
    /// The log probabilities of the tokens this chunk carries, when <see cref="CompletionOptions.TopLogprobs"/> asked
    /// for them; null otherwise.
    /// </summary>
    public IReadOnlyList<LlamaTokenLogprob>? Logprobs { get; init; }
}

/// <summary>
/// Tool call in a message (OpenAI format).
/// </summary>
public sealed class ToolCallMessage
{
    [JsonPropertyName("id")]
    public string? Id { get; set; }

    [JsonPropertyName("type")]
    public string Type { get; set; } = "function";

    [JsonPropertyName("function")]
    public FunctionCallMessage? Function { get; set; }
}

/// <summary>
/// Function call details.
/// </summary>
public sealed class FunctionCallMessage
{
    [JsonPropertyName("name")]
    public string? Name { get; set; }

    [JsonPropertyName("arguments")]
    public string? Arguments { get; set; }
}

/// <summary>
/// Tool definition for the request (OpenAI format).
/// </summary>
public sealed class LlamaToolDefinition
{
    [JsonPropertyName("type")]
    public string Type { get; set; } = "function";

    [JsonPropertyName("function")]
    public FunctionDefinition? Function { get; set; }
}

/// <summary>
/// Function definition for a tool.
/// </summary>
public sealed class FunctionDefinition
{
    [JsonPropertyName("name")]
    public string? Name { get; set; }

    [JsonPropertyName("description")]
    public string? Description { get; set; }

    [JsonPropertyName("parameters")]
    public JsonElement? Parameters { get; set; }
}

/// <summary>
/// Which <see cref="LlamaToolChoice"/> mode a value represents, mirroring the OpenAI-compatible
/// <c>tool_choice</c> wire values llama-server accepts.
/// </summary>
public enum LlamaToolChoiceMode
{
    /// <summary>Model decides freely whether to call a tool.</summary>
    Auto = 0,

    /// <summary>Suppress tool calls even though <see cref="ChatCompletionOptions.Tools"/> is set.</summary>
    None = 1,

    /// <summary>Force the model to call at least one tool.</summary>
    Required = 2,

    /// <summary>Force the model to call one specific, named function.</summary>
    Function = 3
}

/// <summary>
/// Controls whether/which tool the model must call. Only meaningful when
/// <see cref="ChatCompletionOptions.Tools"/> is set.
/// </summary>
public sealed class LlamaToolChoice
{
    /// <summary>Model decides freely whether to call a tool. Equivalent to leaving <see cref="ChatCompletionOptions.ToolChoice"/> unset.</summary>
    public static readonly LlamaToolChoice Auto = new(LlamaToolChoiceMode.Auto, null);

    /// <summary>Suppress tool calls even though <see cref="ChatCompletionOptions.Tools"/> is set.</summary>
    public static readonly LlamaToolChoice None = new(LlamaToolChoiceMode.None, null);

    /// <summary>Force the model to call at least one tool.</summary>
    public static readonly LlamaToolChoice Required = new(LlamaToolChoiceMode.Required, null);

    /// <summary>Force the model to call the named function.</summary>
    public static LlamaToolChoice Function(string name) =>
        new(LlamaToolChoiceMode.Function, name ?? throw new ArgumentNullException(nameof(name)));

    /// <summary>Which mode this instance represents.</summary>
    public LlamaToolChoiceMode Mode { get; }

    /// <summary>The forced function name when <see cref="Mode"/> is <see cref="LlamaToolChoiceMode.Function"/>; null otherwise.</summary>
    public string? FunctionName { get; }

    private LlamaToolChoice(LlamaToolChoiceMode mode, string? functionName)
    {
        Mode = mode;
        FunctionName = functionName;
    }
}

/// <summary>
/// Structured streaming data from a chat completion.
/// Exposed to LMSupply.Generator via InternalsVisibleTo for conversion to ChatStreamChunk.
/// </summary>
public sealed class ChatStreamData
{
    /// <summary>
    /// Text content delta.
    /// </summary>
    public string? TextDelta { get; init; }

    /// <summary>
    /// Reasoning/thinking content delta (b8994+: emitted in separate field before content).
    /// </summary>
    public string? ReasoningDelta { get; init; }

    /// <summary>
    /// Tool call deltas (raw SSE format).
    /// </summary>
    public List<ToolCallDelta>? ToolCallDeltas { get; init; }

    /// <summary>
    /// Finish reason (present only on the final chunk).
    /// </summary>
    public string? FinishReason { get; init; }

    /// <summary>
    /// The server's token accounting for the whole completion (reasoning included), on the chunk that carries it.
    /// </summary>
    public ChatCompletionUsage? Usage { get; init; }

    /// <summary>
    /// The server's timings for the whole completion, on the chunk that carries them. Null on builds that do not send them.
    /// </summary>
    public LlamaServerTimings? Timings { get; init; }

    /// <summary>
    /// The log probabilities of the tokens this chunk carries — reasoning tokens included — when
    /// <see cref="ChatCompletionOptions.TopLogprobs"/> asked for them; null otherwise.
    /// </summary>
    public IReadOnlyList<LlamaTokenLogprob>? Logprobs { get; init; }

    /// <summary>
    /// How far the server has read the prompt, on the data the stream yields before the first token; null otherwise.
    /// </summary>
    public LlamaPromptProgress? PromptProgress { get; init; }
}

/// <summary>
/// Full (non-streaming) chat completion response.
/// </summary>
public sealed class ChatCompletionFullResponse
{
    public List<ChatCompletionFullChoice>? Choices { get; set; }

    /// <summary>
    /// Token accounting the server reports for the completion (OpenAI-compatible <c>usage</c> object).
    /// </summary>
    public ChatCompletionUsage? Usage { get; set; }

    /// <summary>
    /// llama.cpp's <c>timings</c> for the completion. Null on builds that do not send them.
    /// </summary>
    public LlamaServerTimings? Timings { get; set; }
}

/// <summary>
/// OpenAI-compatible <c>usage</c> object of a chat completion response.
/// </summary>
public sealed class ChatCompletionUsage
{
    [JsonPropertyName("prompt_tokens")]
    public int PromptTokens { get; set; }

    [JsonPropertyName("completion_tokens")]
    public int CompletionTokens { get; set; }

    [JsonPropertyName("total_tokens")]
    public int TotalTokens { get; set; }
}

/// <summary>
/// Choice in a full chat completion response.
/// </summary>
public sealed class ChatCompletionFullChoice
{
    public ChatCompletionResponseMessage? Message { get; set; }

    [JsonPropertyName("finish_reason")]
    public string? FinishReason { get; set; }

    /// <summary>
    /// The log probabilities of the generated tokens — reasoning tokens included — when
    /// <see cref="ChatCompletionOptions.TopLogprobs"/> asked for them; null otherwise.
    /// </summary>
    [JsonPropertyName("logprobs")]
    public LlamaChoiceLogprobs? Logprobs { get; set; }
}

/// <summary>
/// The OpenAI-compatible <c>logprobs</c> object of a chat completion choice.
/// </summary>
public sealed class LlamaChoiceLogprobs
{
    /// <summary>One entry per generated token, in order.</summary>
    [JsonPropertyName("content")]
    public List<LlamaTokenLogprob>? Content { get; set; }
}

/// <summary>
/// A token's log probability as llama-server reports it: an entry of the OpenAI-compatible <c>logprobs.content</c>, or
/// of the native <c>completion_probabilities</c>, which has the same shape. The value is the natural log of the
/// token's probability in the model's raw next-token distribution, before sampling.
/// </summary>
public sealed class LlamaTokenLogprob
{
    /// <summary>The token's id in the model's vocabulary; null when the server does not send it.</summary>
    [JsonPropertyName("id")]
    public int? Id { get; set; }

    /// <summary>The token's text. Empty for a token with no text, such as an end-of-generation token.</summary>
    [JsonPropertyName("token")]
    public string? Token { get; set; }

    /// <summary>The token's UTF-8 bytes. A token can hold part of a character, which <see cref="Token"/> cannot show.</summary>
    [JsonPropertyName("bytes")]
    public List<int>? Bytes { get; set; }

    /// <summary>The natural log of the token's probability.</summary>
    [JsonPropertyName("logprob")]
    public double Logprob { get; set; }

    /// <summary>
    /// The most likely tokens at this position, most likely first. Null on a candidate, which has no list of its own.
    /// </summary>
    [JsonPropertyName("top_logprobs")]
    public List<LlamaTokenLogprob>? TopLogprobs { get; set; }
}

/// <summary>
/// Message in a full chat completion response.
/// </summary>
public sealed class ChatCompletionResponseMessage
{
    [JsonPropertyName("role")]
    public string? Role { get; set; }

    [JsonPropertyName("content")]
    public string? Content { get; set; }

    [JsonPropertyName("reasoning_content")]
    public string? ReasoningContent { get; set; }

    [JsonPropertyName("tool_calls")]
    public List<ToolCallMessage>? ToolCalls { get; set; }
}
