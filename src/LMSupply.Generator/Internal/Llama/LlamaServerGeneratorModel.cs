using System.Diagnostics;
using LMSupply.Exceptions;
using LMSupply.Generator.Abstractions;
using LMSupply.Generator.ChatFormatters;
using LMSupply.Hardware;
using LMSupply.Generator.Models;
using LMSupply.Llama.Server;
using LMSupply.Runtime;

namespace LMSupply.Generator.Internal.Llama;

/// <summary>
/// GGUF model implementation using llama-server (standalone llama.cpp HTTP server).
/// Uses LlamaServerPool for server instance reuse across model loads.
/// </summary>
internal sealed partial class LlamaServerGeneratorModel : IGeneratorModel, IDiagnosticsSink
{
    // The server lease, replaced with a new server (same configuration) when the held one has exited.
    private readonly RestartingLease<ServerLease> _lease;
    private readonly IChatFormatter _chatFormatter;
    private readonly GeneratorOptions _options;
    private readonly string _modelPath;
    private readonly SemaphoreSlim _concurrencyLimiter;
    private readonly GgufMetadata? _ggufMetadata;
    private readonly string _serverVersion;
    private SelectionDiagnostics? _diagnostics;
    private readonly int _requestedContextLength;
    private readonly int _effectiveContextLength;
    private readonly LoadSizing _sizing;

    public void SetDiagnostics(SelectionDiagnostics diagnostics) => _diagnostics = diagnostics;
    private bool _disposed;

    private LlamaServerGeneratorModel(
        GgufLoadIdentity identity,
        string modelPath,
        ServerLease serverLease,
        Func<CancellationToken, Task<ServerLease>> leaseServer,
        IChatFormatter chatFormatter,
        GeneratorOptions options,
        int requestedContextLength,
        int effectiveContextLength,
        GgufMetadata? ggufMetadata,
        string serverVersion,
        LoadSizing sizing)
    {
        ModelId = identity.ModelId;
        _identity = identity;
        _modelPath = modelPath;
        _lease = RestartingServerLease.Create(serverLease, leaseServer, $"LlamaServerGeneratorModel '{identity.ModelId}'");
        _chatFormatter = chatFormatter;
        _options = options;
        MaxContextLength = effectiveContextLength;
        _requestedContextLength = requestedContextLength;
        _effectiveContextLength = effectiveContextLength;
        _ggufMetadata = ggufMetadata;
        _serverVersion = serverVersion;
        _sizing = sizing;

        // Initialize concurrency limiter
        _concurrencyLimiter = new SemaphoreSlim(
            Math.Max(1, options.MaxConcurrentRequests),
            Math.Max(1, options.MaxConcurrentRequests));
    }

    /// <summary>
    /// Loads a GGUF model using llama-server.
    /// </summary>
    public static async Task<LlamaServerGeneratorModel> LoadAsync(
        GgufLoadIdentity identity,
        string modelPath,
        IChatFormatter chatFormatter,
        GeneratorOptions options,
        IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        // 1. The llama-server build for this model's chat format. The opening report is the runtime step's, like every
        // report the update service makes.
        var updateService = LlamaServerUpdateService.Resolve(options.ServerUpdateOptions);
        var (serverPath, backend, serverVersion) = await AcquireServerAsync(updateService, chatFormatter, options, progress, cancellationToken);

        // 1c. On a GPU, stop the idle servers of other models before sizing this one: a server whose
        // model was disposed stays pooled for reuse, and on one GPU it keeps the memory this load is
        // about to budget for. A server a live model uses is never stopped; an idle server of this same
        // model is kept, because this load may reuse it.
        if (backend != LlamaServerBackend.Cpu)
        {
            var released = await LlamaServerPool.Instance.ReleaseIdleAsync(
                s => s.Backend != LlamaServerBackend.Cpu
                     && !string.Equals(s.ModelPath, modelPath, StringComparison.OrdinalIgnoreCase), cancellationToken);
            if (released > 0)
            {
                Trace.TraceInformation(
                    $"[LlamaServerGeneratorModel] Stopped {released} idle llama-server(s) of other models to free GPU memory before loading '{Path.GetFileName(modelPath)}'.");
            }
        }

        // 2. Read GGUF metadata (best effort)
        GgufMetadata? ggufMetadata = null;
        try
        {
            ggufMetadata = await GgufMetadataReader.ReadAsync(modelPath, false, cancellationToken);
        }
        catch (Exception ex) when (ex is not OperationCanceledException || !cancellationToken.IsCancellationRequested)
        {
            Trace.TraceInformation($"[LlamaServerGeneratorModel] GGUF metadata reading failed: {ex.Message}");
        }

        // 3. Configure and start llama-server. The pool reports Loading when it starts a server.
        var llamaOpts = ChooseLlamaOptions(options, modelPath, ggufMetadata);
        var contextLength = ResolveRequestedContextLength(options.MaxContextLength, ggufMetadata?.ContextLength);
        var requestedContextLength = contextLength;

        // 3a. A running server of this model that already holds this context is shared as it is — sizing exists
        // to start a new server, and measuring free memory while this model's own server holds it would count
        // that server against itself (a reload of a model whose server is still pooled would come up smaller, or
        // on CPU). An idle server of this model that is too small is superseded: stopped before memory is read.
        var sharedContext = backend != LlamaServerBackend.Cpu
            ? LlamaServerPool.Instance.FindSharableContext(modelPath, backend, ServerMode.Generation, contextLength)
            : null;
        if (backend != LlamaServerBackend.Cpu && sharedContext is null)
        {
            await LlamaServerPool.Instance.ReleaseIdleAsync(
                s => s.Backend == backend
                     && s.Mode == ServerMode.Generation
                     && s.ContextSize < contextLength
                     && string.Equals(s.ModelPath, modelPath, StringComparison.OrdinalIgnoreCase), cancellationToken);
        }

        if (sharedContext is { } shared)
        {
            Trace.TraceInformation(
                $"[LlamaServerGeneratorModel] Sharing the running llama-server of '{Path.GetFileName(modelPath)}' " +
                $"(context {shared}, requested {contextLength}).");
            contextLength = shared;
        }

        // 3b. The GPU as it is now: free memory changes as models load and unload, so the start-up reading is
        // refreshed once all idle servers this load may displace have been stopped.
        var loadGpu = backend != LlamaServerBackend.Cpu && sharedContext is null
            ? GpuDetector.WithCurrentFreeMemory(Hardware.HardwareProfile.Current.GpuInfo)
            : Hardware.HardwareProfile.Current.GpuInfo;

        // What the load settles before the server starts, one step at a time: the GPU layers, then the context the
        // GPU can hold (falling back to CPU where Auto would get no usable one), then the context RAM can hold.
        var plan = new GgufLoadPlan(serverPath, backend, serverVersion, llamaOpts, contextLength, LoadSizing.None);
        if (NeedsVramFit(llamaOpts, backend) && sharedContext is null)
        {
            var (fitted, fittedSizing) = FitGpuLayers(modelPath, llamaOpts, ggufMetadata, contextLength, backend, serverVersion, options, loadGpu);
            plan = plan with { Options = fitted, Sizing = fittedSizing };
        }

        if (backend != LlamaServerBackend.Cpu && sharedContext is null)
        {
            plan = await BoundContextByVramAsync(plan, modelPath, ggufMetadata, options, loadGpu, updateService, progress, cancellationToken);
        }

        if (plan.Backend == LlamaServerBackend.Cpu && options.MaxContextLength is null)
        {
            plan = BoundContextByRam(plan, modelPath, ggufMetadata, options);
        }

        (serverPath, backend, serverVersion, llamaOpts, contextLength, var sizingDecided) = plan;

        var serverConfig = BuildServerConfig(modelPath, contextLength, llamaOpts, backend, serverVersion, options, ggufMetadata);

        // 4. Lease server from pool with OOM retry (reduces GPU layers on failure)
        var plannedGpuLayers = serverConfig.GpuLayers;
        (var serverLease, serverConfig) = await LeaseWithOomRetryAsync(serverPath, serverConfig, backend, progress, cancellationToken);
        var sizing = sizingDecided.AfterRetry(plannedGpuLayers, serverConfig.GpuLayers);

        var restartConfig = serverConfig;
        var model = new LlamaServerGeneratorModel(
            identity,
            modelPath,
            serverLease,
            ct => LlamaServerPool.Instance.LeaseAsync(serverPath, restartConfig, backend, progress: null, ct),
            chatFormatter,
            options,
            requestedContextLength,
            contextLength,
            ggufMetadata,
            serverVersion ?? "unknown",
            sizing);

        // W1: Gemma 4 tool-use risk advisory (llama.cpp #21375 / #21882 not yet merged).
        // Emitted once at load time so operators see the warning before any inference request.
        if (chatFormatter is Gemma4ChatFormatter)
        {
            Trace.TraceWarning(
                "[LlamaServerGeneratorModel] Gemma 4 model loaded. " +
                "llama.cpp #21375 (chat-template/rope) and #21882 (instruction-following) " +
                "are not yet in a stable release. " +
                "Tool-use + Korean instructional prompts may produce empty responses (Q4_K_M). " +
                "Consider Qwen2.5-7B-Instruct GGUF for reliable tool-use until upstream PRs land.");
        }

        return model;
    }

    /// <summary>
    /// What a GGUF load has settled before its server starts: the server build and backend, the llama options, the
    /// context length and what sizing decided.
    /// </summary>
    private sealed record GgufLoadPlan(
        string ServerPath,
        LlamaServerBackend Backend,
        string? ServerVersion,
        LlamaOptions Options,
        int ContextLength,
        LoadSizing Sizing);

    /// <summary>
    /// Bounds the context by what the GPU can hold after the model's layers. When it can offer only the unusable floor
    /// (n_ctx clamped to 512 below the request), Auto falls back to CPU (RAM-bound, no VRAM clamp) and an explicit GPU
    /// pin fails fast instead of silently loading an unusable 512-token context.
    /// </summary>
    private static async Task<GgufLoadPlan> BoundContextByVramAsync(
        GgufLoadPlan plan,
        string modelPath,
        GgufMetadata? ggufMetadata,
        GeneratorOptions options,
        GpuInfo loadGpu,
        LlamaServerUpdateService updateService,
        IProgress<DownloadProgress>? progress,
        CancellationToken cancellationToken)
    {
        var (_, backend, serverVersion, llamaOpts, contextLength, _) = plan;
        var (safeContext, contextFloored) = EstimateSafeContextLengthDetailed(
            modelPath,
            contextLength,
            llamaOpts.GpuLayerCount ?? -1,
            ggufMetadata,
            ResolveKvCacheType(llamaOpts.TypeK, backend, serverVersion),
            ResolveKvCacheType(llamaOpts.TypeV, backend, serverVersion),
            sequences: Math.Max(1, options.MaxConcurrentRequests),
            ubatch: (int)(llamaOpts.UBatchSize ?? KvCacheGeometry.DefaultUBatch),
            gpu: loadGpu);

        // VRAM telemetry is taken before any CPU fallback switches the backend below.
        var vramBudget = VramBudget.GetAvailableBytes(loadGpu);
        plan = plan with
        {
            Sizing = plan.Sizing with
            {
                ContextFlooredByVram = contextFloored,
                VramBudgetBytes = vramBudget > 0 ? vramBudget : null,
                VramTotalBytes = loadGpu.TotalMemoryBytes,
                VramFreeBytes = loadGpu.FreeMemoryBytes ?? loadGpu.TotalMemoryBytes,
            },
        };

        switch (DecideFlooredContextAction(options.Provider, backend, safeContext, contextLength))
        {
            case FlooredContextAction.FailFast:
                throw new InvalidOperationException(
                    $"[LlamaServerGeneratorModel] GPU backend '{backend}' can only provide a {safeContext}-token context " +
                    $"(requested {contextLength}) - VRAM is insufficient for a usable context. " +
                    $"Pin ExecutionProvider.Cpu or free VRAM.");

            case FlooredContextAction.FallBackToCpu:
                Trace.TraceWarning(
                    $"[LlamaServerGeneratorModel] Auto provider: GPU backend '{backend}' yields only {safeContext}-token context " +
                    $"(requested {contextLength}); falling back to CPU for a usable RAM-bound context.");

                // The CPU build replaces the GPU one. CPU is RAM-bound, so the full requested context is kept
                // (EstimateSafeContextLength returns it for GpuLayerCount=0).
                LlamaServerInstallation? cpu = null;
                string? cpuError = null;
                try
                {
                    cpu = await updateService.AcquireServerAsync(LlamaServerBackend.Cpu, options.DisableAutoDownload, progress, cancellationToken);
                }
                catch (Exception ex) when (ex is ModelNotFoundException or InvalidOperationException)
                {
                    cpuError = ex.Message;
                }

                if (cpu is not null)
                {
                    return plan with
                    {
                        ServerPath = cpu.ServerPath,
                        Backend = LlamaServerBackend.Cpu,
                        ServerVersion = cpu.Version ?? serverVersion,
                        Options = CloneLlamaOptionsForCpuFallback(llamaOpts),
                    };
                }

                // CPU binary unavailable — degrade to the floored context rather than fail the load.
                Trace.TraceWarning(
                    $"[LlamaServerGeneratorModel] CPU fallback unavailable ({cpuError}); " +
                    $"using floored {safeContext}-token context.");
                return plan with { ContextLength = safeContext };

            default:
                if (safeContext >= contextLength)
                {
                    return plan;
                }

                const double mb = 1024.0 * 1024.0;
                var budgetMb = vramBudget / mb;
                var totalMb = (loadGpu.TotalMemoryBytes ?? 0) / mb;
                var freeMb = (loadGpu.FreeMemoryBytes ?? loadGpu.TotalMemoryBytes ?? 0) / mb;
                var ctxMsg = $"[LlamaServerGeneratorModel] ctx-size adjusted: requested={contextLength}, actual={safeContext} " +
                    $"(VRAM budget={budgetMb:F0}MB, free={freeMb:F0}MB, total={totalMb:F0}MB)";

                // Warn when the caller explicitly set MaxContextLength and it was silently reduced.
                if (options.MaxContextLength is not null)
                    Trace.TraceWarning(ctxMsg);
                else
                    Trace.TraceInformation(ctxMsg);
                return plan with { ContextLength = safeContext };
        }
    }

    /// <summary>
    /// Bounds a context the caller did not choose by RAM, where the KV cache lives on the CPU backend (chosen or fallen
    /// back to); the GPU path bounds it by VRAM. An explicit length is the caller's.
    /// </summary>
    private static GgufLoadPlan BoundContextByRam(GgufLoadPlan plan, string modelPath, GgufMetadata? ggufMetadata, GeneratorOptions options)
    {
        var ramContext = EstimateRamBoundContextLength(
            CacheManager.GetContentLength(modelPath),
            plan.ContextLength,
            ggufMetadata,
            ResolveKvCacheType(plan.Options.TypeK, plan.Backend, plan.ServerVersion),
            ResolveKvCacheType(plan.Options.TypeV, plan.Backend, plan.ServerVersion),
            Math.Max(1, options.MaxConcurrentRequests),
            (int)(plan.Options.UBatchSize ?? KvCacheGeometry.DefaultUBatch),
            Hardware.HardwareProfile.Current.SystemMemoryBytes);
        if (ramContext >= plan.ContextLength)
        {
            return plan;
        }

        Trace.TraceInformation(
            $"[LlamaServerGeneratorModel] ctx-size bounded by RAM: requested={plan.ContextLength}, actual={ramContext}");
        return plan with { ContextLength = ramContext };
    }

    /// <summary>
    /// What sizing decided for a load, reported by <see cref="GetModelInfo"/>: the GPU layers of a partial offload and
    /// the estimates behind them, and the VRAM budget the context was sized against.
    /// </summary>
    internal sealed record LoadSizing(
        int? GpuLayers = null,
        int? TotalLayers = null,
        long? EstimatedVramBytes = null,
        long? EstimatedRamBytes = null,
        long? VramBudgetBytes = null,
        long? VramFreeBytes = null,
        long? VramTotalBytes = null,
        bool ContextFlooredByVram = false)
    {
        /// <summary>Nothing was decided: the model loads as configured.</summary>
        public static LoadSizing None { get; } = new();

        /// <summary>
        /// The sizing once the server has started: when out-of-memory retries started it with fewer GPU layers than
        /// planned, those are the layers it runs with.
        /// </summary>
        public LoadSizing AfterRetry(int plannedGpuLayers, int startedGpuLayers) =>
            startedGpuLayers == plannedGpuLayers ? this : this with { GpuLayers = startedGpuLayers };
    }

    /// <summary>
    /// Gets the llama-server build for <paramref name="chatFormatter"/>'s format. A cached build too old for the format is
    /// updated once before <see cref="LlamaServerVersionRequirements.Validate"/> rejects it — avoids requiring manual
    /// cache deletion. A pinned version is never exceeded: the update short-circuits to the same pinned binary, and
    /// Validate then throws, correctly. With downloads disabled the server comes from the cache only, with no update
    /// check, and no newer build is fetched.
    /// </summary>
    private static async Task<(string ServerPath, LlamaServerBackend Backend, string? Version)> AcquireServerAsync(
        LlamaServerUpdateService updateService,
        IChatFormatter chatFormatter,
        GeneratorOptions options,
        IProgress<DownloadProgress>? progress,
        CancellationToken cancellationToken)
    {
        progress?.Report(new DownloadProgress
        {
            FileName = "llama-server",
            Kind = DownloadKind.Runtime,
            Phase = DownloadPhase.Preparing
        });

        var preferredBackend = global::LMSupply.Llama.LlamaBackendSelector.MapProvider(
            options.Provider, Hardware.HardwareProfile.For(options.Provider).GpuInfo);
        var installation = await updateService.AcquireServerAsync(
            preferredBackend,
            options.DisableAutoDownload,
            progress,
            cancellationToken);

        var serverPath = installation.ServerPath;
        var backend = installation.Backend;
        var serverVersion = installation.Version;

        if (!options.DisableAutoDownload
            && !LlamaServerVersionRequirements.MeetsMinimum(serverVersion, chatFormatter.FormatName))
        {
            var retryResult = await updateService.CheckAndApplyUpdateAsync(
                preferredBackend, progress, cancellationToken);
            serverPath    = retryResult.ServerPath;
            backend       = retryResult.Backend;
            serverVersion = retryResult.NewVersion ?? retryResult.PreviousVersion;
        }

        LlamaServerVersionRequirements.Validate(serverVersion, chatFormatter.FormatName);
        return (serverPath, backend, serverVersion);
    }

    /// <summary>
    /// Leases a server for <paramref name="serverConfig"/>. A start that runs out of memory with GPU layers left is
    /// retried with about a quarter fewer (at least one), until it starts or no GPU layer is left. Returns the lease and
    /// the configuration it was started with.
    /// </summary>
    private static async Task<(ServerLease Lease, LlamaServerConfig Config)> LeaseWithOomRetryAsync(
        string serverPath,
        LlamaServerConfig serverConfig,
        LlamaServerBackend backend,
        IProgress<DownloadProgress>? progress,
        CancellationToken cancellationToken)
    {
        var currentGpuLayers = serverConfig.GpuLayers;
        var leaseProgress = progress;

        while (true)
        {
            try
            {
                var lease = await LlamaServerPool.Instance.LeaseAsync(
                    serverPath,
                    serverConfig,
                    backend,
                    leaseProgress,
                    cancellationToken);
                return (lease, serverConfig);
            }
            catch (Exception ex) when ((ex is not OperationCanceledException || !cancellationToken.IsCancellationRequested) && (IsOomError(ex) && currentGpuLayers > 0))
            {
                // The start that just failed reported Loading; the retries are the same load.
                leaseProgress = null;

                // Reduce GPU layers by ~25% (minimum 1 layer reduction)
                var reduction = Math.Max(1, currentGpuLayers / 4);
                currentGpuLayers -= reduction;

                Trace.TraceInformation(
                    $"[LlamaServerGeneratorModel] OOM detected, retrying with {currentGpuLayers} GPU layers " +
                    $"(was {serverConfig.GpuLayers}): {ex.Message}");

                serverConfig = CloneConfigWithGpuLayers(serverConfig, currentGpuLayers);
            }
        }
    }

    /// <inheritdoc />
    public string ModelId { get; }

    private readonly GgufLoadIdentity _identity;

    /// <inheritdoc />
    public int MaxContextLength { get; }

    /// <inheritdoc />
    public IChatFormatter ChatFormatter => _chatFormatter;

    /// <inheritdoc />
    public bool IsGpuActive => _lease.Current.Backend != LlamaServerBackend.Cpu;

    /// <inheritdoc />
    public IReadOnlyList<string> ActiveProviders => IsGpuActive
        ? new[] { $"llama-server-{_lease.Current.Backend}", "CPU" }
        : new[] { "llama-server-CPU" };

    /// <inheritdoc />
    public ExecutionProvider RequestedProvider => _options.Provider;

    /// <inheritdoc />
    public long? EstimatedMemoryBytes => CacheManager.TryGetContentLength(_modelPath, out var length) ? length * 2 : null;

    /// <summary>
    /// Gets the startup log from the llama-server process for diagnostics.
    /// </summary>
    public string? ServerStartupLog => _lease.Current.Server.Info?.StartupLog;

    /// <inheritdoc />
    public IAsyncEnumerable<string> GenerateAsync(
        string prompt,
        GenerationOptions? options = null,
        CancellationToken cancellationToken = default)
        => GenerateCoreAsync(prompt, options, outcome: null, cancellationToken);

    /// <summary>
    /// Why a raw completion ended, filled in by <see cref="GenerateCoreAsync"/> once its stream completes.
    /// </summary>
    private sealed class CompletionOutcome
    {
        public string? FinishReason { get; set; }

        /// <summary>The server's own token counts, when it reported them.</summary>
        public int? PromptTokens { get; set; }
        public int? CompletionTokens { get; set; }

        /// <summary>The server's timings, when it reported them; null once the client-side cap cut the stream.</summary>
        public GenerationTimings? Timings { get; set; }

        /// <summary>The generated tokens' log probabilities, when the options asked for them.</summary>
        public List<TokenLogprob>? Logprobs { get; private set; }

        public void AddLogprobs(IReadOnlyList<LlamaTokenLogprob>? logprobs)
        {
            if (ToTokenLogprobs(logprobs) is { } converted)
                (Logprobs ??= []).AddRange(converted);
        }

        /// <summary>
        /// The server's counts where it reported them, else an estimate from the text — marked as one, since a
        /// reasoning model's hidden reasoning never appears in the text it is estimated from.
        /// </summary>
        public TokenUsage ToUsage(string promptText, string content)
            => PromptTokens is { } prompt && CompletionTokens is { } completion
                ? new TokenUsage(prompt, completion)
                : new TokenUsage(TokenUsage.EstimateTokens(promptText), TokenUsage.EstimateTokens(content)) { IsEstimated = true };
    }

    private async IAsyncEnumerable<string> GenerateCoreAsync(
        string prompt,
        GenerationOptions? options,
        CompletionOutcome? outcome,
        [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        await foreach (var chunk in StreamCompletionAsync(prompt, options, outcome ?? new CompletionOutcome(), cancellationToken))
        {
            if (!string.IsNullOrEmpty(chunk.Text))
                yield return chunk.Text;
        }
    }

    /// <inheritdoc />
    public IAsyncEnumerable<TextStreamChunk> GenerateStreamAsync(
        string prompt,
        GenerationOptions? options = null,
        CancellationToken cancellationToken = default)
        => StreamCompletionAsync(prompt, options, new CompletionOutcome(), cancellationToken);

    /// <summary>
    /// A raw-prompt completion as structured chunks: the server's prompt-reading progress, the text (through the
    /// reasoning filter when the options ask for one) with its log probabilities, and a final chunk with the ending,
    /// usage and timings, which <paramref name="outcome"/> also records.
    /// </summary>
    private async IAsyncEnumerable<TextStreamChunk> StreamCompletionAsync(
        string prompt,
        GenerationOptions? options,
        CompletionOutcome outcome,
        [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        ThrowIfDisposed();
        options ??= GenerationOptions.Default;

        await _concurrencyLimiter.WaitAsync(cancellationToken);
        try
        {
            var completionOptions = new CompletionOptions
            {
                MaxTokens = options.ResolveMaxOutputTokens(),
                Temperature = options.Temperature,
                TopP = options.TopP,
                TopK = options.TopK,
                MinP = options.MinP,
                RepeatPenalty = options.RepetitionPenalty,
                FrequencyPenalty = options.FrequencyPenalty,
                PresencePenalty = options.PresencePenalty,
                Seed = options.Seed,
                StopSequences = MergeStopSequences(options.StopSequences),
                Grammar = options.Grammar,
                JsonSchema = options.JsonSchema,
                TopLogprobs = options.TopLogprobs
            };

            // Client-side token limit as safety net (server enforces too, but may fail)
            var maxTokens = completionOptions.MaxTokens;
            var tokenCount = 0;

            // Initialize reasoning token filter if needed
            var useReasoningFilter = options.FilterReasoningTokens || options.ExtractReasoningTokens;
            var reasoningFilter = useReasoningFilter
                ? new ReasoningTokenFilter(options.ExtractReasoningTokens)
                : null;

            await foreach (var data in (await ClientAsync(cancellationToken)).GenerateStreamAsync(prompt, completionOptions, cancellationToken))
            {
                // Before the first token: nothing to filter, and the consumer is waiting for any sign of life.
                if (data.PromptProgress is { } progress)
                {
                    yield return new TextStreamChunk { PromptProgress = ToPromptProgress(progress) };
                    continue;
                }

                if (data.FinishReason is not null)
                    outcome.FinishReason = data.FinishReason;
                if (data.PromptTokens is { } evaluated && data.CompletionTokens is { } predicted)
                {
                    outcome.PromptTokens = evaluated;
                    outcome.CompletionTokens = predicted;
                }
                if (data.Timings is { } rawTimings)
                    outcome.Timings = ToGenerationTimings(rawTimings);

                if (data.TextDelta is not { } token)
                {
                    outcome.AddLogprobs(data.Logprobs);
                    if (ToTokenLogprobs(data.Logprobs) is { } onlyLogprobs)
                        yield return new TextStreamChunk { Logprobs = onlyLogprobs };
                    continue;
                }

                if (maxTokens > 0 && ++tokenCount > maxTokens)
                {
                    // The client-side cap cut the stream before the server said why: that is the output limit.
                    outcome.FinishReason = "length";
                    outcome.PromptTokens = null;
                    outcome.CompletionTokens = null;
                    outcome.Timings = null;
                    break;
                }

                outcome.AddLogprobs(data.Logprobs);
                var text = reasoningFilter is null ? token : reasoningFilter.Process(token);
                var logprobs = ToTokenLogprobs(data.Logprobs);
                if (!string.IsNullOrEmpty(text) || logprobs is not null)
                    yield return new TextStreamChunk { Text = string.IsNullOrEmpty(text) ? null : text, Logprobs = logprobs };
            }

            // Flush remaining content
            if (reasoningFilter?.Flush() is { Length: > 0 } remaining)
                yield return new TextStreamChunk { Text = remaining };

            yield return new TextStreamChunk
            {
                FinishReason = outcome.FinishReason ?? "stop",
                Usage = outcome.PromptTokens is { } prompts && outcome.CompletionTokens is { } completions
                    ? new ChatTokenUsage { PromptTokens = prompts, CompletionTokens = completions, TotalTokens = prompts + completions }
                    : null,
                Timings = outcome.Timings,
            };
        }
        finally
        {
            _concurrencyLimiter.Release();
        }
    }

    /// <inheritdoc />
    public IAsyncEnumerable<string> GenerateChatAsync(
        IEnumerable<ChatMessage> messages,
        GenerationOptions? options = null,
        CancellationToken cancellationToken = default)
        => GenerateChatCoreAsync(messages, options, outcome: null, cancellationToken);

    private async IAsyncEnumerable<string> GenerateChatCoreAsync(
        IEnumerable<ChatMessage> messages,
        GenerationOptions? options,
        CompletionOutcome? outcome,
        [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        ThrowIfDisposed();
        options ??= GenerationOptions.Default;

        await _concurrencyLimiter.WaitAsync(cancellationToken);
        try
        {
            var trimmedMessages = await TrimToFitContextAsync(messages, options, cancellationToken);
            // Convert to llama-server format
            var serverMessages = ConvertMessages(PrepareServerMessages(trimmedMessages, options, _chatFormatter));
            var chatOptions = CreateChatOptions(options);

            // Client-side token limit as safety net
            var maxTokens = options.ResolveMaxOutputTokens();
            var tokenCount = 0;

            // Initialize reasoning token filter if needed
            var useReasoningFilter = options.FilterReasoningTokens || options.ExtractReasoningTokens;
            var reasoningFilter = useReasoningFilter
                ? new ReasoningTokenFilter(options.ExtractReasoningTokens)
                : null;

            // The structured stream carries the server's finish_reason on its last chunk; the text stream dropped it.
            // Text is the same: content deltas only (reasoning_content is not part of the answer).
            await foreach (var data in (await ClientAsync(cancellationToken)).GenerateChatStreamAsync(serverMessages, chatOptions, cancellationToken))
            {
                if (data.FinishReason is not null && outcome is not null)
                    outcome.FinishReason = data.FinishReason;
                if (data.Usage is { } usage && outcome is not null)
                {
                    outcome.PromptTokens = usage.PromptTokens;
                    outcome.CompletionTokens = usage.CompletionTokens;
                }
                if (data.Timings is { } chatTimings && outcome is not null)
                    outcome.Timings = ToGenerationTimings(chatTimings);

                var token = data.TextDelta;
                if (string.IsNullOrEmpty(token))
                {
                    // Reasoning tokens carry log probabilities on chunks with no answer text.
                    outcome?.AddLogprobs(data.Logprobs);
                    continue;
                }

                if (maxTokens > 0 && ++tokenCount > maxTokens)
                {
                    if (outcome is not null)
                    {
                        outcome.FinishReason = "length";
                        outcome.Timings = null;
                    }
                    break;
                }

                outcome?.AddLogprobs(data.Logprobs);

                if (reasoningFilter != null)
                {
                    var filtered = reasoningFilter.Process(token);
                    if (!string.IsNullOrEmpty(filtered))
                    {
                        yield return filtered;
                    }
                }
                else
                {
                    yield return token;
                }
            }

            // Flush remaining content
            if (reasoningFilter != null)
            {
                var remaining = reasoningFilter.Flush();
                if (!string.IsNullOrEmpty(remaining))
                {
                    yield return remaining;
                }
            }
        }
        finally
        {
            _concurrencyLimiter.Release();
        }
    }

    /// <inheritdoc />
    public async Task<string> GenerateCompleteAsync(
        string prompt,
        GenerationOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        var sb = new StringBuilder();

        await foreach (var token in GenerateAsync(prompt, options, cancellationToken))
        {
            sb.Append(token);
        }

        return sb.ToString();
    }

    /// <inheritdoc />
    public async Task<GenerationResult> GenerateCompleteResultAsync(
        string prompt,
        GenerationOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        var outcome = new CompletionOutcome();
        var sb = new StringBuilder();

        await foreach (var token in GenerateCoreAsync(prompt, options, outcome, cancellationToken))
        {
            sb.Append(token);
        }

        var content = sb.ToString();
        return new GenerationResult(content, outcome.ToUsage(prompt, content), outcome.FinishReason)
        {
            Timings = outcome.Timings,
            Logprobs = outcome.Logprobs,
        };
    }

    /// <inheritdoc />
    public async Task<string> GenerateChatCompleteAsync(
        IEnumerable<ChatMessage> messages,
        GenerationOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        var sb = new StringBuilder();

        await foreach (var token in GenerateChatAsync(messages, options, cancellationToken))
        {
            sb.Append(token);
        }

        return sb.ToString();
    }

    /// <inheritdoc />
    public async Task<GenerationResult> GenerateChatCompleteResultAsync(
        IEnumerable<ChatMessage> messages,
        GenerationOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        var messageList = messages.ToList();
        var outcome = new CompletionOutcome();
        var sb = new StringBuilder();

        await foreach (var token in GenerateChatCoreAsync(messageList, options, outcome, cancellationToken))
        {
            sb.Append(token);
        }

        var content = sb.ToString();
        return new GenerationResult(
            content,
            outcome.ToUsage(string.Concat(messageList.Select(m => m.Content)), content),
            outcome.FinishReason)
        {
            Timings = outcome.Timings,
            Logprobs = outcome.Logprobs,
        };
    }

    /// <summary>
    /// The client of a running server. A server that has exited — killed, crashed, or out of GPU memory because another
    /// process took it — is replaced by a new one from the pool with the configuration this model was loaded with, so
    /// the call that saw it die fails but the next one works. Before 0.77.0 every later call failed on the dead port
    /// with "connection refused", and nothing said the server was gone.
    /// </summary>
    private async ValueTask<LlamaServerClient> ClientAsync(CancellationToken cancellationToken) =>
        (await _lease.GetAsync(cancellationToken).ConfigureAwait(false)).Client;

    /// <inheritdoc />
    public Task WarmupAsync(CancellationToken cancellationToken = default)
    {
        // Server is already warmed up during StartAsync health check
        // Optionally perform a minimal generation
        return GenerateCompleteAsync(
            "Hi",
            new GenerationOptions { MaxTokens = 5 },
            cancellationToken);
    }

    /// <inheritdoc />
    public GeneratorModelInfo GetModelInfo() => new(
        ModelId,
        _modelPath,
        MaxContextLength,
        _chatFormatter.FormatName,
        $"llama-server-{_lease.Current.Backend}")
    {
        GgufMetadata = _ggufMetadata,
        BackendLog = _lease.Current.Server.Info?.StartupLog,
        RuntimeVersion = _serverVersion,
        Diagnostics = _diagnostics,
        AdjustedContextLength = ResolveAdjustedContextLength(_requestedContextLength, _effectiveContextLength),
        KnownIssues = _identity.KnownIssues,
        RequestedModelId = _identity.RequestedModelId,
        LoadedFile = Path.GetFileName(_modelPath),
        RequestedFile = _identity.RequestedFile,
        GpuLayers = _sizing.GpuLayers,
        TotalLayers = _sizing.TotalLayers,
        EstimatedVramBytes = _sizing.EstimatedVramBytes,
        EstimatedRamBytes = _sizing.EstimatedRamBytes,
        VramBudgetBytes = _sizing.VramBudgetBytes,
        VramFreeBytes = _sizing.VramFreeBytes,
        VramTotalBytes = _sizing.VramTotalBytes,
        ContextFlooredByVram = _sizing.ContextFlooredByVram,
    };

    /// <inheritdoc />
    public async Task<int> CountTokensAsync(string text, CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();

        if (string.IsNullOrEmpty(text))
            return 0;

        return await (await ClientAsync(cancellationToken)).CountTokensAsync(text, cancellationToken);
    }

    /// <inheritdoc />
    public async Task<IReadOnlyList<int>> TokenizeAsync(string text, CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();

        if (string.IsNullOrEmpty(text))
            return [];

        return await (await ClientAsync(cancellationToken)).TokenizeAsync(text, addSpecial: false, cancellationToken);
    }

    /// <inheritdoc />
    public async Task<int> CountTokensAsync(IEnumerable<ChatMessage> messages, CancellationToken cancellationToken = default)
    {
        var prompt = _chatFormatter.FormatPrompt(messages);
        return await CountTokensAsync(prompt, cancellationToken);
    }

    /// <inheritdoc />
    public async Task<int> CountTokensAsync(
        IEnumerable<ChatMessage> messages,
        GenerationOptions? options,
        CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();
        options ??= GenerationOptions.Default;
        return await CountRenderedPromptTokensAsync(
            PrepareServerMessages(messages, options, _chatFormatter), options, cancellationToken);
    }

    /// <summary>
    /// The token count of the prompt the server renders for <paramref name="prepared"/> (already through
    /// <see cref="PrepareServerMessages"/>) with <paramref name="options"/>: the server's own chat template, with the
    /// tools, tool choice and thinking setting the request sends. A server without <c>/apply-template</c> falls back
    /// to the formatter's prompt, which does not include the tools the template would add.
    /// </summary>
    private async Task<int> CountRenderedPromptTokensAsync(
        IReadOnlyList<ChatMessage> prepared,
        GenerationOptions options,
        CancellationToken cancellationToken)
    {
        var client = await ClientAsync(cancellationToken);
        if (await client.CountChatPromptTokensAsync(ConvertMessages(prepared), CreateChatOptions(options), cancellationToken)
            is { } rendered)
        {
            return rendered;
        }

        return await client.CountTokensAsync(_chatFormatter.FormatPrompt(prepared), cancellationToken);
    }

    private void ThrowIfDisposed()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
    }

    public async ValueTask DisposeAsync()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        _concurrencyLimiter.Dispose();

        // Return server to pool (does not terminate the server)
        await _lease.DisposeAsync();
    }
}
