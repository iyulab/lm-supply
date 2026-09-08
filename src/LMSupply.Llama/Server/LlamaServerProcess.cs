using System.Diagnostics;
using System.Globalization;
using System.Net;
using System.Net.Sockets;
using System.Text.RegularExpressions;
using ChildProcessGuard;

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
    /// The llama-server build tag the binary at the server path was resolved as (e.g. <c>b10298</c>),
    /// when known. Used to pick the argument spelling the binary understands where llama.cpp has
    /// renamed a flag — currently <c>--load-mode</c> (b10105+) versus the deprecated
    /// <c>--mmap</c>/<c>--no-mmap</c>/<c>--mlock</c>. <c>null</c> (a consumer-provisioned binary of
    /// unknown build) keeps the legacy spelling, which parses on both sides of the gate today.
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
    /// Timeout for a single chat completion HTTP request to the running server.
    /// Defaults to 5 minutes, not HttpClient's 100-second default — local/CPU-bound inference,
    /// especially a multi-step tool-calling loop where each round's prompt grows with prior tool
    /// results, routinely exceeds 100 seconds per completion on hardware without a GPU.
    /// </summary>
    public TimeSpan RequestTimeout { get; init; } = TimeSpan.FromMinutes(5);
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
    public string BaseUrl => $"http://localhost:{Port}";

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

/// <summary>
/// Manages llama-server process lifecycle with automatic cleanup.
/// </summary>
public sealed class LlamaServerProcess : IAsyncDisposable
{
    private readonly ProcessGuardian _guardian;
    private readonly LlamaServerConfig _config;
    private readonly string _serverPath;
    private readonly LlamaServerBackend _backend;
    private readonly HttpClient _httpClient;

    private Process? _process;
    private int _port;
    private bool _disposed;

    /// <summary>
    /// Gets information about the running server.
    /// </summary>
    public LlamaServerInfo? Info { get; private set; }

    /// <summary>
    /// Gets whether the server is running.
    /// </summary>
    public bool IsRunning => _process is { HasExited: false };

    private LlamaServerProcess(
        string serverPath,
        LlamaServerConfig config,
        LlamaServerBackend backend)
    {
        _serverPath = serverPath;
        _config = config;
        _backend = backend;

        _guardian = new ProcessGuardian(new ProcessGuardianOptions
        {
            ProcessKillTimeout = config.ShutdownTimeout
        });

        _httpClient = new HttpClient
        {
            Timeout = TimeSpan.FromSeconds(5)
        };
    }

    /// <summary>
    /// Starts a new llama-server process.
    /// </summary>
    public static async Task<LlamaServerProcess> StartAsync(
        string serverPath,
        LlamaServerConfig config,
        LlamaServerBackend backend,
        CancellationToken cancellationToken = default)
    {
        var server = new LlamaServerProcess(serverPath, config, backend);

        try
        {
            await server.StartInternalAsync(cancellationToken);
            return server;
        }
        catch
        {
            await server.DisposeAsync();
            throw;
        }
    }

    private async Task StartInternalAsync(CancellationToken cancellationToken)
    {
        // Find available port if not specified
        _port = _config.Port > 0 ? _config.Port : FindAvailablePort();

        // Build arguments
        var args = BuildArguments();

        // Get the directory containing llama-server for DLL resolution
        var workingDir = Path.GetDirectoryName(_serverPath)!;

        // Start process via guardian
        var startInfo = new ProcessStartInfo
        {
            FileName = _serverPath,
            Arguments = args,
            WorkingDirectory = workingDir,
            UseShellExecute = false,
            CreateNoWindow = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true
        };

        // Enable CUDA graph optimization for NVIDIA GPUs (reduces token generation latency)
        if (_backend == LlamaServerBackend.Cuda12 || _backend == LlamaServerBackend.Cuda13)
        {
            startInfo.Environment["GGML_CUDA_GRAPH_OPT"] = "1";
        }

        // A launch that never happened yields "No process is associated with this object" -- a bare
        // message with no binary path, no backend, nothing to act on. It replaces the diagnostic a
        // few lines further down, which is the one written for exactly this situation, so the caller
        // is told the LEAST precisely when the failure is most opaque.
        //
        // The throw comes from two places and both have to be covered. The guardian reads Id while
        // starting, so the exception can surface before this method ever gets a Process back; and if
        // a future guardian returns without touching it, the first member here that needs the
        // handle throws instead. Guarding only the returned object -- as this did between 0.37.2 and
        // 0.38.1 -- covers the second and misses the first, which is the one that actually occurs.
        try
        {
            _process = _guardian.StartProcessWithStartInfo(startInfo);
        }
        catch (InvalidOperationException ex)
        {
            throw new InvalidOperationException(DidNotLaunchMessage(_serverPath, _backend, workingDir), ex);
        }

        if (!HasLaunched(_process))
        {
            throw new InvalidOperationException(DidNotLaunchMessage(_serverPath, _backend, workingDir));
        }

        // Capture stderr output for diagnostics
        var stderrBuilder = new System.Text.StringBuilder();
        _process.ErrorDataReceived += (_, e) =>
        {
            if (e.Data != null)
            {
                stderrBuilder.AppendLine(e.Data);
            }
        };
        _process.BeginErrorReadLine();

        // Wait for server to be ready
        var startTime = DateTimeOffset.UtcNow;
        var failure = await WaitForServerReadyAsync(startTime.UtcDateTime, stderrBuilder, cancellationToken);

        if (failure != null)
        {
            // Collect error output
            var error = stderrBuilder.ToString();
            if (string.IsNullOrEmpty(error) && _process.HasExited)
            {
                error = "Process exited without error output";
            }

            throw new InvalidOperationException(
                $"llama-server failed to start: {failure}. " +
                $"Exit code: {(_process.HasExited ? _process.ExitCode : "still running")}. " +
                $"Error: {error}");
        }

        var startupLog = stderrBuilder.ToString();

        Info = new LlamaServerInfo
        {
            ProcessId = _process.Id,
            Port = _port,
            ModelPath = _config.ModelPath,
            Backend = _backend,
            StartTime = startTime,
            StartupLog = startupLog
        };

        // Silent-fallback guard: a GPU backend whose runtime cannot load (e.g. cudart/cublas missing
        // alongside the cuda binary, or a driver/runtime mismatch) starts and serves normally but
        // llama.cpp enumerates only a CPU device and runs on CPU with no error. Surface it as a
        // warning so the consumer is not silently downgraded to CPU performance.
        if (IsGpuBackend(_backend) && !StartupLogShowsGpuDevice(startupLog))
        {
            Trace.TraceWarning(
                $"[LlamaServerProcess] {_backend} backend was selected but llama-server initialized " +
                "CPU-only (no GPU device enumerated). The GPU runtime likely failed to load " +
                "(e.g. CUDA runtime cudart/cublas missing alongside the server binary, or a " +
                "driver/runtime mismatch). Inference will run on CPU. Install the GPU runtime or " +
                "ensure the runtime libraries are present next to the llama-server binary.");
        }
    }

    /// <summary>
    /// Translates the memory-map / memory-lock options into the flags the given build parses.
    /// llama.cpp b10105 replaced <c>--mmap</c>/<c>--no-mmap</c>/<c>--mlock</c> with one
    /// <c>--load-mode</c> (<c>none</c> | <c>mmap</c> | <c>mlock</c> = mmap + lock | <c>dio</c>);
    /// the old flags still parse there but log <c>DEPRECATED</c> on every start and will be removed,
    /// at which point passing them is a fatal "unknown argument" before the port is open. Builds
    /// below the gate, and binaries of unknown build, get the legacy flags — the only spelling
    /// guaranteed to parse on both sides today.
    /// </summary>
    internal static IReadOnlyList<string> ResolveLoadModeArgs(bool? useMemoryMap, bool? useMemoryLock, string? serverVersion)
    {
        var args = new List<string>(2);
        var build = LlamaServerVersionRequirements.ParseBuildNumber(serverVersion);
        var gate = LlamaServerVersionRequirements.GetMinimumBuild("load-mode");

        if (build.HasValue && gate.HasValue && build.Value >= gate.Value)
        {
            // The new vocabulary has no "lock without mapping" mode: mlock implies mmap. Locking
            // is the stronger intent, so it wins over an explicit UseMemoryMap = false.
            string? mode = useMemoryLock == true ? "mlock"
                : useMemoryMap == true ? "mmap"
                : useMemoryMap == false ? "none"
                : null;

            if (mode != null)
            {
                args.Add("--load-mode");
                args.Add(mode);
            }

            return args;
        }

        if (useMemoryMap.HasValue)
        {
            args.Add(useMemoryMap.Value ? "--mmap" : "--no-mmap");
        }

        if (useMemoryLock == true)
        {
            args.Add("--mlock");
        }

        return args;
    }

    private string BuildArguments()
    {
        var args = new List<string>
        {
            "--model", $"\"{_config.ModelPath}\"",
            "--port", _port.ToString(CultureInfo.InvariantCulture),
            "--ctx-size", _config.ContextSize.ToString(CultureInfo.InvariantCulture),
            "--n-gpu-layers", _config.GpuLayers.ToString(CultureInfo.InvariantCulture),
            "--batch-size", _config.BatchSize.ToString(CultureInfo.InvariantCulture),
            "--parallel", _config.Parallel.ToString(CultureInfo.InvariantCulture),
            "--host", "127.0.0.1", // Only listen on localhost for security
            "--cont-batching",     // Enable continuous batching for better throughput
            "--jinja"              // Enable Jinja template processing for native tool calling support
        };

        // Physical batch size for VRAM efficiency
        if (_config.UBatchSize.HasValue)
        {
            args.Add("--ubatch-size");
            args.Add(_config.UBatchSize.Value.ToString(CultureInfo.InvariantCulture));
        }

        if (_config.FlashAttention)
        {
            // llama.cpp b8795+ requires an explicit value (on|off|auto).
            // Older builds that accepted the boolean form are no longer distributed via LMSupply.
            args.Add("--flash-attn");
            args.Add("on");
        }

        // KV cache quantization (Phase 1)
        if (!string.IsNullOrEmpty(_config.CacheTypeK))
        {
            args.Add("--cache-type-k");
            args.Add(_config.CacheTypeK);
        }

        if (!string.IsNullOrEmpty(_config.CacheTypeV))
        {
            args.Add("--cache-type-v");
            args.Add(_config.CacheTypeV);
        }

        // Memory options (Phase 1) — spelling depends on the build, see ResolveLoadModeArgs.
        args.AddRange(ResolveLoadModeArgs(_config.UseMemoryMap, _config.UseMemoryLock, _config.ServerVersion));

        // GPU options (Phase 1)
        if (_config.MainGpu.HasValue)
        {
            args.Add("--main-gpu");
            args.Add(_config.MainGpu.Value.ToString(CultureInfo.InvariantCulture));
        }

        // RoPE options (Phase 1)
        if (_config.RopeFreqBase.HasValue)
        {
            args.Add("--rope-freq-base");
            args.Add(_config.RopeFreqBase.Value.ToString("F1", CultureInfo.InvariantCulture));
        }

        if (_config.RopeFreqScale.HasValue)
        {
            args.Add("--rope-freq-scale");
            args.Add(_config.RopeFreqScale.Value.ToString("F4", CultureInfo.InvariantCulture));
        }

        // Speculative decoding
        if (!string.IsNullOrEmpty(_config.SpecType))
        {
            args.Add("--spec-type");
            args.Add(_config.SpecType);
        }

        if (!string.IsNullOrEmpty(_config.ModelDraft))
        {
            args.Add("--model-draft");
            args.Add($"\"{_config.ModelDraft}\"");
        }

        // YaRN RoPE scaling
        if (!string.IsNullOrEmpty(_config.RopeScaling))
        {
            args.Add("--rope-scaling");
            args.Add(_config.RopeScaling);
        }

        if (_config.YarnOriginalContext.HasValue)
        {
            args.Add("--yarn-orig-ctx");
            args.Add(_config.YarnOriginalContext.Value.ToString(CultureInfo.InvariantCulture));
        }

        if (_config.YarnExtensionFactor.HasValue)
        {
            args.Add("--yarn-ext-factor");
            args.Add(_config.YarnExtensionFactor.Value.ToString("F4", CultureInfo.InvariantCulture));
        }

        if (_config.YarnAttentionFactor.HasValue)
        {
            args.Add("--yarn-attn-factor");
            args.Add(_config.YarnAttentionFactor.Value.ToString("F4", CultureInfo.InvariantCulture));
        }

        if (_config.YarnBetaFast.HasValue)
        {
            args.Add("--yarn-beta-fast");
            args.Add(_config.YarnBetaFast.Value.ToString("F4", CultureInfo.InvariantCulture));
        }

        if (_config.YarnBetaSlow.HasValue)
        {
            args.Add("--yarn-beta-slow");
            args.Add(_config.YarnBetaSlow.Value.ToString("F4", CultureInfo.InvariantCulture));
        }

        // Multimodal projector (Phase 3)
        if (!string.IsNullOrEmpty(_config.MultimodalProjector))
        {
            args.Add("--mmproj");
            args.Add($"\"{_config.MultimodalProjector}\"");
        }

        // LoRA adapter (Phase 3)
        if (!string.IsNullOrEmpty(_config.LoraPath))
        {
            if (_config.LoraScale.HasValue)
            {
                args.Add("--lora-scaled");
                args.Add($"\"{_config.LoraPath}\"");
                args.Add(_config.LoraScale.Value.ToString("F2", CultureInfo.InvariantCulture));
            }
            else
            {
                args.Add("--lora");
                args.Add($"\"{_config.LoraPath}\"");
            }
        }

        // Embedding/Reranking mode flags
        if (_config.Mode == ServerMode.Embedding || _config.Mode == ServerMode.Reranking)
        {
            args.Add("--embedding");
        }

        // Pooling type
        var poolingType = _config.Pooling;

        // For reranking mode, force rank pooling if not explicitly set
        if (_config.Mode == ServerMode.Reranking && poolingType == PoolingType.None)
        {
            poolingType = PoolingType.Rank;
        }

        if (poolingType != PoolingType.None)
        {
            args.Add("--pooling");
            args.Add(poolingType.ToString().ToLowerInvariant());
        }

        if (_config.AdditionalArgs != null)
        {
            args.AddRange(_config.AdditionalArgs);
        }

        return string.Join(" ", args);
    }

    /// <summary>
    /// Polls <c>/health</c> until the server answers, the process dies, stderr shows a fatal CLI
    /// error, or <see cref="StartupProgressTracker"/> declares the start stalled or capped.
    /// Returns <c>null</c> when ready, otherwise a one-line reason for the failure.
    /// </summary>
    private async Task<string?> WaitForServerReadyAsync(
        DateTime startedAt,
        System.Text.StringBuilder stderrBuilder,
        CancellationToken cancellationToken)
    {
        var tracker = new StartupProgressTracker(startedAt, _config.StartupStallTimeout, _config.StartupTimeout);

        while (true)
        {
            cancellationToken.ThrowIfCancellationRequested();

            if (_process?.HasExited == true)
            {
                return "the process exited before answering /health";
            }

            // Fail fast on fatal CLI-parse errors — llama.cpp prints these before the HTTP port
            // is up and the process may linger briefly, so don't wait for HasExited or the full timeout.
            if (HasFatalStartupError(stderrBuilder))
            {
                return "llama-server rejected its command line (see stderr)";
            }

            try
            {
                var response = await _httpClient.GetAsync($"http://localhost:{_port}/health", cancellationToken);
                if (response.IsSuccessStatusCode)
                {
                    return null;
                }
            }
            catch (HttpRequestException)
            {
                // Server not ready yet
            }
            catch (TaskCanceledException)
            {
                // Timeout, retry
            }

            var now = DateTime.UtcNow;
            var sample = SampleProcess(_process);
            if (sample == null)
            {
                // The process went away between the HasExited check and the read.
                return "the process exited before answering /health";
            }

            var verdict = tracker.Observe(now, stderrBuilder.Length, sample.Value.WorkingSet, sample.Value.CpuTime);
            if (verdict != StartupWaitVerdict.Waiting)
            {
                return tracker.Describe(now);
            }

            await Task.Delay(100, cancellationToken);
        }
    }

    /// <summary>
    /// Reads the process metrics the stall detector watches. <c>null</c> when the process is gone —
    /// <see cref="Process.Refresh"/> and the counters throw once the process has exited, and the
    /// caller's earlier <see cref="Process.HasExited"/> check can race that.
    /// </summary>
    private static (long WorkingSet, TimeSpan CpuTime)? SampleProcess(Process? process)
    {
        if (process == null)
            return null;

        try
        {
            process.Refresh();
            return (process.WorkingSet64, process.TotalProcessorTime);
        }
        catch (InvalidOperationException)
        {
            return null;
        }
    }

    private static bool HasFatalStartupError(System.Text.StringBuilder stderrBuilder)
    {
        if (stderrBuilder.Length == 0)
            return false;

        var text = stderrBuilder.ToString();
        return text.Contains("error while handling argument", StringComparison.Ordinal)
            || text.Contains("error: invalid argument", StringComparison.Ordinal)
            || text.Contains("unknown argument", StringComparison.Ordinal);
    }

    /// <summary>
    /// Checks if the server is healthy.
    /// </summary>
    public async Task<bool> CheckHealthAsync(CancellationToken cancellationToken = default)
    {
        if (!IsRunning)
            return false;

        try
        {
            var response = await _httpClient.GetAsync($"http://localhost:{_port}/health", cancellationToken);
            return response.IsSuccessStatusCode;
        }
        catch (Exception ex)
        {
            Trace.TraceInformation($"[LlamaServerProcess] Health check failed: {ex.Message}");
            return false;
        }
    }

    /// <summary>
    /// Stops the server gracefully.
    /// </summary>
    public async Task StopAsync(CancellationToken cancellationToken = default)
    {
        if (_process == null || _process.HasExited)
            return;

        try
        {
            // Try graceful shutdown first
            _process.Kill(entireProcessTree: true);
            await _process.WaitForExitAsync(cancellationToken);
        }
        catch (InvalidOperationException)
        {
            // Process already exited
        }
    }

    private static int FindAvailablePort()
    {
        using var listener = new TcpListener(IPAddress.Loopback, 0);
        listener.Start();
        var port = ((IPEndPoint)listener.LocalEndpoint).Port;
        listener.Stop();
        return port;
    }

    // device_info bullet line for an accelerated (non-CPU) compute device, e.g.
    // "  - CUDA0   : NVIDIA GeForce RTX 4060 ..." or "  - Vulkan0 : ...". A CPU-only line
    // ("  - CPU : ...") does not match. The leading "- " anchors to the device list so model
    // names or system_info text containing a backend keyword are not mistaken for a device.
    private static readonly Regex GpuDeviceLineRegex = new(
        @"-\s*(?:CUDA|Vulkan|Metal|ROCm|HIP|SYCL|CANN|OpenCL)\d*\s*:",
        RegexOptions.IgnoreCase | RegexOptions.Compiled);

    /// <summary>
    /// Returns true if the llama-server startup log shows an accelerated (non-CPU) compute device
    /// was enumerated. A GPU llama-server binary that cannot load its runtime (e.g. missing
    /// cudart/cublas) starts and serves normally but enumerates only a CPU device — llama.cpp
    /// silently runs on CPU. This predicate lets callers detect that silent fallback and warn
    /// instead of failing silently. Conservative: a null/empty log returns false (no GPU evidence).
    /// </summary>
    internal static bool StartupLogShowsGpuDevice(string? startupLog)
        => !string.IsNullOrWhiteSpace(startupLog) && GpuDeviceLineRegex.IsMatch(startupLog);

    /// <summary>
    /// The message for a launch that produced no process. Shared by both detection points so they
    /// cannot drift into saying different things about the same failure.
    /// </summary>
    internal static string DidNotLaunchMessage(string serverPath, LlamaServerBackend backend, string workingDir) =>
        $"llama-server did not launch: '{serverPath}' (backend {backend}, working directory " +
        $"'{workingDir}'). The process was never created, so there is no exit code or error output " +
        "to report. Check that the binary exists at that path, is executable, and that its runtime " +
        "dependencies are resolvable on this machine.";

    /// <summary>
    /// True when the object actually has an OS process behind it. A failed launch still yields a
    /// <see cref="Process"/> instance, and every member that needs the underlying handle -- including
    /// <c>Id</c> -- throws <see cref="InvalidOperationException"/> on it. Probing <c>Id</c> is the
    /// discriminator: it succeeds for a process that has already exited, and fails only for one that
    /// was never created.
    /// </summary>
    internal static bool HasLaunched(Process? process)
    {
        if (process is null)
            return false;

        try
        {
            _ = process.Id;
            return true;
        }
        catch (InvalidOperationException)
        {
            return false;
        }
    }

    /// <summary>True for llama-server backends that offload to a GPU (i.e. should engage a device).</summary>
    internal static bool IsGpuBackend(LlamaServerBackend backend) => backend switch
    {
        LlamaServerBackend.Cuda12 => true,
        LlamaServerBackend.Cuda13 => true,
        LlamaServerBackend.Vulkan => true,
        LlamaServerBackend.Hip => true,
        LlamaServerBackend.Sycl => true,
        LlamaServerBackend.Metal => true,
        _ => false
    };

    public async ValueTask DisposeAsync()
    {
        if (_disposed)
            return;

        _disposed = true;

        await StopAsync();
        _httpClient.Dispose();
        _guardian.Dispose();
        _process?.Dispose();
    }
}
