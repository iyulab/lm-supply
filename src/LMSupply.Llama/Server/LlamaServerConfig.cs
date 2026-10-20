namespace LMSupply.Llama.Server;

/// <summary>
/// Server operation mode - determines which llama-server endpoints are available.
/// </summary>
public enum ServerMode
{
    /// <summary>
    /// Text generation mode (default). Uses /completion and /v1/chat/completions endpoints.
    /// </summary>
    Generation,

    /// <summary>
    /// Embedding mode. Uses /v1/embeddings endpoint. Requires --embedding flag.
    /// </summary>
    Embedding,

    /// <summary>
    /// Reranking mode. Uses /v1/rerank endpoint. Requires --embedding and --pooling rank flags.
    /// </summary>
    Reranking
}

/// <summary>
/// Pooling type for embedding/reranking operations.
/// </summary>
public enum PoolingType
{
    /// <summary>
    /// No pooling specified (use model default).
    /// </summary>
    None,

    /// <summary>
    /// Mean pooling - average all token embeddings.
    /// </summary>
    Mean,

    /// <summary>
    /// CLS token pooling - use first token embedding.
    /// </summary>
    Cls,

    /// <summary>
    /// Last token pooling - use last token embedding.
    /// </summary>
    Last,

    /// <summary>
    /// Rank pooling - for reranking models (cross-encoder output).
    /// </summary>
    Rank
}

/// <summary>
/// Configuration for llama-server process.
/// </summary>
public sealed class LlamaServerConfig
{
    /// <summary>
    /// Path to the GGUF model file.
    /// </summary>
    public required string ModelPath { get; init; }

    /// <summary>
    /// Port to run the server on (0 for auto-assign).
    /// </summary>
    public int Port { get; init; }

    /// <summary>
    /// Context size.
    /// </summary>
    public int ContextSize { get; init; } = 4096;

    /// <summary>
    /// Number of GPU layers to offload (-1 for all).
    /// </summary>
    public int GpuLayers { get; init; } = -1;

    /// <summary>
    /// Batch size for prompt processing (logical).
    /// Higher values speed up prompt evaluation but use more memory.
    /// </summary>
    public int BatchSize { get; init; } = 512;

    /// <summary>
    /// Physical batch size (ubatch). Controls VRAM usage during processing.
    /// Must be less than or equal to BatchSize. Default: 512.
    /// </summary>
    public int? UBatchSize { get; init; }

    /// <summary>
    /// Number of parallel sequences.
    /// </summary>
    public int Parallel { get; init; } = 1;

    /// <summary>
    /// Enable Flash Attention.
    /// </summary>
    public bool FlashAttention { get; init; }

    #region KV Cache Options

    /// <summary>
    /// KV cache type for keys (f16, q8_0, q4_0, f32).
    /// Reduces memory usage at the cost of potential quality loss.
    /// </summary>
    public string? CacheTypeK { get; init; }

    /// <summary>
    /// KV cache type for values (f16, q8_0, q4_0, f32).
    /// Reduces memory usage at the cost of potential quality loss.
    /// </summary>
    public string? CacheTypeV { get; init; }

    #endregion

    #region Memory Options

    /// <summary>
    /// Use memory mapping for model loading (mmap).
    /// Enables faster loading and sharing between processes. Default: true.
    /// </summary>
    public bool? UseMemoryMap { get; init; }

    /// <summary>
    /// Lock model memory to prevent swapping (mlock).
    /// Improves latency but may require elevated privileges.
    /// </summary>
    public bool? UseMemoryLock { get; init; }

    #endregion

    #region GPU Options

    /// <summary>
    /// Main GPU index for multi-GPU systems (0-based).
    /// </summary>
    public int? MainGpu { get; init; }

    #endregion

    #region RoPE Options

    /// <summary>
    /// RoPE frequency base for context extension.
    /// Use with RoPE-scaling-aware models.
    /// </summary>
    public float? RopeFreqBase { get; init; }

    /// <summary>
    /// RoPE frequency scale factor.
    /// Use with RoPE-scaling-aware models.
    /// </summary>
    public float? RopeFreqScale { get; init; }

    #endregion

    #region Speculative Decoding Options

    /// <summary>
    /// Speculative decoding type (--spec-type). "ngram" for N-gram based speculation.
    /// </summary>
    public string? SpecType { get; init; }

    /// <summary>
    /// Draft model path for speculative decoding (--model-draft).
    /// </summary>
    public string? ModelDraft { get; init; }

    #endregion

    #region YaRN RoPE Scaling Options

    /// <summary>
    /// RoPE scaling mode (--rope-scaling). Values: linear, yarn, longrope.
    /// </summary>
    public string? RopeScaling { get; init; }

    /// <summary>
    /// YaRN: original context window size the model was trained with (--yarn-orig-ctx).
    /// </summary>
    public uint? YarnOriginalContext { get; init; }

    /// <summary>
    /// YaRN: extrapolation mix factor (--yarn-ext-factor). -1 = model default.
    /// </summary>
    public float? YarnExtensionFactor { get; init; }

    /// <summary>
    /// YaRN: attention magnitude scaling factor (--yarn-attn-factor).
    /// </summary>
    public float? YarnAttentionFactor { get; init; }

    /// <summary>
    /// YaRN: low-frequency correction ramp parameter (--yarn-beta-fast).
    /// </summary>
    public float? YarnBetaFast { get; init; }

    /// <summary>
    /// YaRN: high-frequency correction ramp parameter (--yarn-beta-slow).
    /// </summary>
    public float? YarnBetaSlow { get; init; }

    #endregion

    #region Multimodal Options (Phase 3)

    /// <summary>
    /// Path to multimodal projector file (--mmproj).
    /// Required for vision models like LLaVA.
    /// </summary>
    public string? MultimodalProjector { get; init; }

    #endregion

    #region LoRA Options (Phase 3)

    /// <summary>
    /// Path to LoRA adapter file (--lora).
    /// </summary>
    public string? LoraPath { get; init; }

    /// <summary>
    /// LoRA adapter scale (--lora-scaled).
    /// </summary>
    public float? LoraScale { get; init; }

    #endregion

    /// <summary>
    /// Additional command line arguments.
    /// </summary>
    public IReadOnlyList<string>? AdditionalArgs { get; init; }

    /// <summary>
    /// Whether the server requires a key on every request. Default: <see langword="true"/>.
    /// </summary>
    /// <remarks>
    /// The server listens on loopback, which keeps other machines out but not other processes or other users of the same
    /// machine. With this on, each launched server gets its own random key (<see cref="LlamaServerProcess.ApiKey"/>),
    /// handed to the process through its environment rather than its command line, and every request LMSupply sends
    /// carries it. Turn it off only to let another program call the server without a key. While it is on,
    /// <c>--api-key</c> or <c>--api-key-file</c> in <see cref="AdditionalArgs"/> is refused at start.
    /// </remarks>
    public bool RequireApiKey { get; init; } = true;

    /// <summary>
    /// The llama-server build tag the binary at the server path was resolved as (e.g. <c>b10298</c>),
    /// when known. Used to pick the argument spelling the binary understands where llama.cpp has
    /// renamed a flag — currently <c>--load-mode</c> (b10105+) versus the removed
    /// <c>--mmap</c>/<c>--no-mmap</c>/<c>--mlock</c>. A consumer-provisioned binary is probed with
    /// <c>--version</c>; <c>null</c> or an unparsable value means its build is unknown, and only
    /// spellings that parse on every build are then used where one exists.
    /// </summary>
    public string? ServerVersion { get; init; }

    /// <summary>
    /// Server operation mode. Default: Generation.
    /// Embedding mode: enables --embedding flag
    /// Reranking mode: enables --embedding and --pooling rank
    /// </summary>
    public ServerMode Mode { get; init; } = ServerMode.Generation;

    /// <summary>
    /// Pooling type for embedding/reranking modes.
    /// Only applicable when Mode is Embedding or Reranking.
    /// </summary>
    public PoolingType Pooling { get; init; } = PoolingType.None;

    /// <summary>
    /// Default for <see cref="StartupTimeout"/>: 10 minutes.
    /// </summary>
    public static readonly TimeSpan DefaultStartupTimeout = TimeSpan.FromMinutes(10);

    /// <summary>
    /// Default for <see cref="StartupStallTimeout"/>: 120 seconds.
    /// </summary>
    public static readonly TimeSpan DefaultStartupStallTimeout = TimeSpan.FromSeconds(120);

    /// <summary>
    /// Absolute limit on server startup, counted from process launch regardless of activity.
    /// Default: <see cref="DefaultStartupTimeout"/> (10 minutes).
    /// </summary>
    /// <remarks>
    /// This is the far-out cap, not the working limit. A start is normally declared failed by
    /// <see cref="StartupStallTimeout"/> — the process going quiet — because how long a legitimate
    /// start takes depends on model size and disk speed (a 4 GB model read through mmap from a cold
    /// page cache takes minutes on a shared CI runner), and a fixed budget cannot tell slow from
    /// stuck. The cap only bounds a process that keeps showing activity without ever answering
    /// <c>/health</c>. Must be at least <see cref="StartupStallTimeout"/>.
    /// </remarks>
    public TimeSpan StartupTimeout { get; init; } = DefaultStartupTimeout;

    /// <summary>
    /// How long the starting server may go without any observable activity before it is declared
    /// stuck. Default: <see cref="DefaultStartupStallTimeout"/> (120 seconds).
    /// </summary>
    /// <remarks>
    /// Activity is any of: a new stderr line, a new working-set high-water mark (pages being faulted
    /// in during the model load), or CPU time consumed (context creation and warm-up). The model load
    /// itself emits no line-delimited output, so stderr alone would read as silence for the whole
    /// read; the process metrics are what keep a slow-but-busy start alive. Raise this only for
    /// environments where the process can legitimately show none of the three for longer — e.g. a
    /// network filesystem that blocks the first page fault.
    /// </remarks>
    public TimeSpan StartupStallTimeout { get; init; } = DefaultStartupStallTimeout;

    /// <summary>
    /// Timeout for graceful shutdown.
    /// </summary>
    public TimeSpan ShutdownTimeout { get; init; } = TimeSpan.FromSeconds(10);

    /// <summary>
    /// Default for <see cref="RequestTimeout"/>: 5 minutes.
    /// </summary>
    public static readonly TimeSpan DefaultRequestTimeout = TimeSpan.FromMinutes(5);

    /// <summary>
    /// Limit on a single HTTP request to the running server (a non-streamed completion as a whole,
    /// a streamed one up to its first response headers). Default: <see cref="DefaultRequestTimeout"/>
    /// (5 minutes), not HttpClient's 100 seconds — local/CPU-bound inference, especially a
    /// multi-step tool-calling loop where each round's prompt grows with prior tool results,
    /// routinely exceeds 100 seconds per completion on hardware without a GPU.
    /// <see cref="Timeout.InfiniteTimeSpan"/> means no limit.
    /// </summary>
    /// <remarks>
    /// Applies to the lease that asked for it, not to the server process: callers that share a
    /// pooled server (same model, backend, context size and mode) each keep their own limit.
    /// </remarks>
    public TimeSpan RequestTimeout { get; init; } = DefaultRequestTimeout;
}

/// <summary>
/// Information about a running llama-server process.
/// </summary>
public sealed record LlamaServerInfo
{
    /// <summary>
    /// Process ID.
    /// </summary>
    public required int ProcessId { get; init; }

    /// <summary>
    /// Port the server is listening on.
    /// </summary>
    public required int Port { get; init; }

    /// <summary>
    /// Base URL for API calls.
    /// </summary>
    public string BaseUrl => $"http://{LlamaServerProcess.LoopbackHost}:{Port}";

    /// <summary>
    /// Model path being served.
    /// </summary>
    public required string ModelPath { get; init; }

    /// <summary>
    /// llama-server version.
    /// </summary>
    public string? Version { get; init; }

    /// <summary>
    /// GPU backend being used.
    /// </summary>
    public LlamaServerBackend Backend { get; init; }

    /// <summary>
    /// Time when server was started.
    /// </summary>
    public DateTimeOffset StartTime { get; init; }

    /// <summary>
    /// Startup log from the server (stderr output during initialization).
    /// </summary>
    public string? StartupLog { get; init; }
}
