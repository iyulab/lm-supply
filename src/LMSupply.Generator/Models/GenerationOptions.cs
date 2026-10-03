namespace LMSupply.Generator.Models;

/// <summary>
/// Configuration options for text generation (inference parameters).
/// </summary>
public sealed class GenerationOptions
{
    /// <summary>
    /// Default hard cap on generated tokens, applied when no positive limit is supplied.
    /// Guarantees generation can never be unbounded even if a caller passes a non-positive value.
    /// </summary>
    public const int DefaultMaxTokens = 512;

    /// <summary>
    /// Gets or sets the maximum number of tokens to generate.
    /// Enforced both server-side (llama-server) and client-side as a safety net.
    /// Defaults to 512.
    /// </summary>
    public int MaxTokens { get; set; } = DefaultMaxTokens;

    /// <summary>
    /// Gets or sets the temperature for sampling.
    /// Higher values produce more random outputs. Range: 0.0 to 2.0.
    /// Defaults to 0.7.
    /// </summary>
    public float Temperature { get; set; } = 0.7f;

    /// <summary>
    /// Gets or sets the top-p (nucleus) sampling parameter.
    /// Considers tokens with cumulative probability mass up to this value. Range: 0.0 to 1.0.
    /// Defaults to 0.9.
    /// </summary>
    public float TopP { get; set; } = 0.9f;

    /// <summary>
    /// Gets or sets the top-k sampling parameter.
    /// Considers only the top k tokens. Set to 0 to disable.
    /// Defaults to 50.
    /// </summary>
    public int TopK { get; set; } = 50;

    /// <summary>
    /// Gets or sets the repetition penalty applied to tokens that have already appeared.
    /// Values greater than 1.0 discourage repetition; 1.0 disables the penalty entirely.
    /// Typical range: 1.0 to 1.3. Values above 1.5 may degrade output quality.
    /// Defaults to 1.1.
    /// </summary>
    /// <remarks>
    /// This is the primary defense against repetition loops. If the model produces
    /// repetitive text, increase this value (e.g., 1.2–1.3). For creative writing
    /// or deterministic tasks, adjust accordingly via <see cref="Creative"/> or <see cref="Precise"/> presets.
    /// Sent to llama-server as <c>repeat_penalty</c>.
    /// </remarks>
    public float RepetitionPenalty { get; set; } = 1.1f;

    /// <summary>
    /// Gets or sets the min-p (minimum probability) sampling parameter.
    /// Filters out tokens below this probability threshold relative to the top token.
    /// Range: 0.0 to 1.0. Set to 0 to disable.
    /// Defaults to 0.05 (5% of top token probability).
    /// </summary>
    /// <remarks>
    /// Min-p is a dynamic cutoff that adapts to confidence levels:
    /// - When the model is confident (high top token probability), more tokens are filtered
    /// - When uncertain (low top token probability), more tokens are considered
    /// Works well with TopK for improved output quality.
    /// </remarks>
    public float MinP { get; set; } = 0.05f;

    /// <summary>
    /// Gets or sets the random seed for reproducible generation.
    /// Set to a specific value for deterministic outputs across identical inputs.
    /// Defaults to -1 (random seed each time).
    /// </summary>
    public int Seed { get; set; } = -1;

    /// <summary>
    /// Gets or sets the frequency penalty (OpenAI-style).
    /// Penalizes tokens proportionally to how often they appear in the text so far.
    /// Complements <see cref="RepetitionPenalty"/> for fine-grained repetition control.
    /// Range: 0.0 to 2.0. Defaults to 0.0 (disabled).
    /// </summary>
    public float FrequencyPenalty { get; set; }

    /// <summary>
    /// Gets or sets the presence penalty (OpenAI-style).
    /// Applies a flat penalty to any token that has appeared at least once.
    /// Encourages the model to introduce new topics rather than revisiting earlier ones.
    /// Range: 0.0 to 2.0. Defaults to 0.0 (disabled).
    /// </summary>
    public float PresencePenalty { get; set; }

    #region Advanced anti-repetition (standard samplers; null = backend default)

    /// <summary>
    /// DRY (Don't Repeat Yourself) sampler multiplier. 0.0 disables DRY; typical enabled value ~0.8.
    /// DRY penalizes repeated token sequences and is the most effective standard defense against
    /// degenerate run-on / repetition loops, especially on low-end and quantized models where a scalar
    /// <see cref="RepetitionPenalty"/> alone is insufficient. Null = use the backend default.
    /// </summary>
    /// <remarks>Backend support: llama-server only (sent as <c>dry_multiplier</c>). Ignored by the ONNX backend.</remarks>
    public float? DryMultiplier { get; set; }

    /// <summary>
    /// DRY sampler base — controls how steeply the penalty grows with repeated-sequence length.
    /// llama-server default is 1.75. Only meaningful when <see cref="DryMultiplier"/> &gt; 0. Null = backend default.
    /// </summary>
    /// <remarks>Backend support: llama-server only (sent as <c>dry_base</c>).</remarks>
    public float? DryBase { get; set; }

    /// <summary>
    /// DRY sampler allowed length — sequences up to this length are not penalized. llama-server default is 2.
    /// Only meaningful when <see cref="DryMultiplier"/> &gt; 0. Null = backend default.
    /// </summary>
    /// <remarks>Backend support: llama-server only (sent as <c>dry_allowed_length</c>).</remarks>
    public int? DryAllowedLength { get; set; }

    /// <summary>
    /// DRY sampler look-back window in tokens. llama-server default is -1 (context size); 0 disables.
    /// Only meaningful when <see cref="DryMultiplier"/> &gt; 0. Null = backend default.
    /// </summary>
    /// <remarks>Backend support: llama-server only (sent as <c>dry_penalty_last_n</c>).</remarks>
    public int? DryPenaltyLastN { get; set; }

    /// <summary>
    /// Number of most-recent tokens that <see cref="RepetitionPenalty"/> considers (the penalty window).
    /// llama-server default is 64; 0 disables, -1 = context size. Null = backend default.
    /// </summary>
    /// <remarks>Backend support: llama-server only (sent as <c>repeat_last_n</c>).</remarks>
    public int? RepeatLastN { get; set; }

    /// <summary>
    /// Blocks any n-gram of this size from being generated more than once (hard repetition cut).
    /// 0 disables. A standard HuggingFace/ONNX defense against verbatim phrase loops. Null = backend default.
    /// </summary>
    /// <remarks>Backend support: ONNX backend only (set as <c>no_repeat_ngram_size</c>). Not supported by llama-server, which omits it.</remarks>
    public int? NoRepeatNgramSize { get; set; }

    #endregion

    /// <summary>
    /// Gets or sets the stop sequences that will terminate generation.
    /// </summary>
    public IReadOnlyList<string>? StopSequences { get; set; }

    /// <summary>
    /// Gets or sets whether to enable random sampling.
    /// When false, uses greedy decoding (always picks highest probability token).
    /// Defaults to true.
    /// </summary>
    public bool DoSample { get; set; } = true;

    /// <summary>
    /// Gets or sets the number of beams for beam search.
    /// Set to 1 to disable beam search.
    /// Note: Beam search disables KV cache sharing for better quality.
    /// Defaults to 1.
    /// </summary>
    public int NumBeams { get; set; } = 1;

    /// <summary>
    /// Gets or sets whether to share buffer between past and present KV cache.
    /// Improves memory efficiency but incompatible with beam search (num_beams > 1).
    /// Defaults to true.
    /// </summary>
    public bool PastPresentShareBuffer { get; set; } = true;

    /// <summary>
    /// Gets or sets the maximum number of new tokens to generate (excluding prompt).
    /// If null, limited only by MaxTokens (prompt + generated).
    /// </summary>
    public int? MaxNewTokens { get; set; }

    /// <summary>
    /// Gets or sets whether to filter reasoning tokens from output.
    /// When true, content within &lt;think&gt;...&lt;/think&gt; tags is removed from streamed output.
    /// Useful for DeepSeek R1 and other reasoning models to show only final answers.
    /// Defaults to false (reasoning tokens are included in output).
    /// </summary>
    public bool FilterReasoningTokens { get; set; }

    /// <summary>
    /// Gets or sets whether to extract reasoning tokens to a separate property.
    /// When true, reasoning content is captured but not included in main output.
    /// Use with GenerateChatWithReasoningAsync to access reasoning content.
    /// Defaults to false.
    /// </summary>
    public bool ExtractReasoningTokens { get; set; }

    /// <summary>
    /// Controls the model's reasoning ("thinking") behavior on the chat path.
    /// <see cref="ThinkingMode.Auto"/> (default) keeps each model's built-in default, which varies by
    /// model and llama.cpp build (Qwen3 thinks by default; some Gemma 4 GGUF builds emit reasoning too).
    /// <see cref="ThinkingMode.On"/> activates thinking (injects the formatter thinking token and requests
    /// <c>enable_thinking=true</c>); <see cref="ThinkingMode.Off"/> suppresses it (requests
    /// <c>enable_thinking=false</c>) so the model answers directly instead of spending tokens on a
    /// reasoning block. Honored only on the chat path (the raw completion path has no chat template to drive).
    /// Note for small token budgets: a thinking model can exhaust <see cref="MaxTokens"/> on reasoning
    /// before emitting any answer (empty content, finish=length). Use <see cref="ThinkingMode.Off"/> to
    /// get a direct answer in a tight budget — <see cref="FilterReasoningTokens"/> only strips reasoning
    /// from output, it does not stop reasoning from consuming the budget.
    /// </summary>
    public ThinkingMode Thinking { get; set; } = ThinkingMode.Auto;

    /// <summary>
    /// Tool definitions available for the model to call.
    /// When set, the model may respond with tool calls instead of text.
    /// </summary>
    public IReadOnlyList<ChatToolDefinition>? Tools { get; set; }

    /// <summary>
    /// Controls whether, and which, tool the model must call. Null (default) leaves the model free
    /// to decide, same as <see cref="Models.ToolChoice.Auto"/>. Only meaningful when <see cref="Tools"/>
    /// is non-empty.
    /// </summary>
    public ToolChoice? ToolChoice { get; set; }

    #region Phase 3: Grammar Constraints

    /// <summary>
    /// Gets or sets a grammar constraint in GBNF (GGML BNF) format.
    /// Constrains generation to match the specified grammar rules.
    /// Use for enforcing specific output formats like JSON, markdown, etc.
    /// Mutually exclusive with <see cref="JsonSchema"/>: llama-server rejects a request that sets both,
    /// so setting both throws <see cref="System.ArgumentException"/> at request time.
    /// </summary>
    /// <example>
    /// // Simple grammar for yes/no answers:
    /// Grammar = "root ::= (\"yes\" | \"no\")"
    /// </example>
    /// <remarks>Backend support: llama-server only (sent as the GBNF <c>grammar</c> field). Ignored by the ONNX backend.</remarks>
    public string? Grammar { get; set; }

    /// <summary>
    /// Gets or sets a JSON schema (as a JSON string) to constrain generation.
    /// When set, output will be valid JSON matching this schema.
    /// On the chat path the schema is sent via the OpenAI-compatible
    /// <c>response_format: { type: "json_schema", json_schema: { schema } }</c>; on the raw completion
    /// path it is sent as the native root <c>json_schema</c> object. The string is parsed to a JSON
    /// object before sending — an unparseable value throws <see cref="System.ArgumentException"/> rather
    /// than producing an opaque HTTP 400. Mutually exclusive with <see cref="Grammar"/>.
    /// </summary>
    /// <example>
    /// JsonSchema = "{\"type\":\"object\",\"properties\":{\"answer\":{\"type\":\"string\"}}}"
    /// </example>
    /// <remarks>Backend support: llama-server only. The ONNX backend does not constrain output to the schema.</remarks>
    public string? JsonSchema { get; set; }

    #endregion

    /// <summary>
    /// Resolves the effective hard cap on output (generated) tokens that every backend must enforce.
    /// Prefers <see cref="MaxNewTokens"/> when set, otherwise falls back to <see cref="MaxTokens"/>.
    /// A non-positive resolved value is normalized to <see cref="DefaultMaxTokens"/> so generation
    /// is never unbounded — a non-positive limit is treated as "unset", not "infinite".
    /// </summary>
    /// <remarks>
    /// This is the single decision point shared by the llama-server and ONNX backends, ensuring the
    /// <see cref="MaxTokens"/> safety-net contract holds identically across both. The previous ONNX
    /// path ignored <see cref="MaxTokens"/> entirely (fell back to <c>int.MaxValue</c>), allowing a
    /// low-end model to run on to the full context window — this method closes that gap.
    /// </remarks>
    public int ResolveMaxOutputTokens()
    {
        var effective = MaxNewTokens ?? MaxTokens;
        return effective > 0 ? effective : DefaultMaxTokens;
    }

    /// <summary>
    /// Creates a default instance of GenerationOptions.
    /// </summary>
    public static GenerationOptions Default => new();

    /// <summary>
    /// Creates options optimized for creative text generation.
    /// </summary>
    public static GenerationOptions Creative => new()
    {
        Temperature = 0.9f,
        TopP = 0.95f,
        TopK = 100,
        RepetitionPenalty = 1.2f
    };

    /// <summary>
    /// Creates options optimized for deterministic/precise outputs.
    /// </summary>
    /// <remarks>
    /// Ships <see cref="RepetitionPenalty"/> = 1.0 (defense off), which is fine for full-precision models
    /// but risky on low-end/quantized ones. See <see cref="AdaptiveSamplingPolicy"/> to restore a safe floor.
    /// </remarks>
    public static GenerationOptions Precise => new()
    {
        Temperature = 0.1f,
        TopP = 0.5f,
        TopK = 10,
        RepetitionPenalty = 1.0f
    };

    /// <summary>
    /// Creates options tuned for Gemma 4 models per Google's published recommendations.
    /// temperature=1.0, top_p=0.95, top_k=64 — required for stable tool-call generation
    /// on E4B and larger. Use instead of <see cref="Default"/> when loading a Gemma 4 model.
    /// Ships <see cref="RepetitionPenalty"/> = 1.0 (defense off) per Google's recommendation; on a
    /// low-end/quantized Gemma 4 this raises run-on risk — see <see cref="AdaptiveSamplingPolicy"/>.
    /// </summary>
    public static GenerationOptions Gemma4 => new()
    {
        Temperature = 1.0f,
        TopP = 0.95f,
        TopK = 64,
        RepetitionPenalty = 1.0f,
    };

    /// <summary>
    /// Sampling parameters per official Qwen3 recommendation for thinking mode.
    /// Temperature = 0.6, TopP = 0.95, TopK = 20, MinP = 0.0, RepetitionPenalty = 1.0.
    /// Note: RepetitionPenalty = 1.0 disables the defense; on a low-end/quantized Qwen3 this raises
    /// run-on risk — see <see cref="AdaptiveSamplingPolicy"/> to restore a safe floor.
    /// </summary>
    public static GenerationOptions Qwen3 => new()
    {
        Temperature = 0.6f,
        TopP = 0.95f,
        TopK = 20,
        MinP = 0.0f,
        RepetitionPenalty = 1.0f,
    };
}
