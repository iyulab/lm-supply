using System.Diagnostics;
using System.Globalization;
using System.Runtime.CompilerServices;
using LMSupply.Download;
using LMSupply.Exceptions;
using LMSupply.Generator.Abstractions;
using LMSupply.Generator.ChatFormatters;
using LMSupply.Hardware;
using LMSupply.Generator.Models;
using LMSupply.Llama.Server;

namespace LMSupply.Generator.Internal.Llama;

/// <summary>
/// GGUF model implementation using llama-server (standalone llama.cpp HTTP server).
/// Uses LlamaServerPool for server instance reuse across model loads.
/// </summary>
internal sealed class LlamaServerGeneratorModel : IGeneratorModel, IDiagnosticsSink
{
    private readonly ServerLease _serverLease;
    private readonly IChatFormatter _chatFormatter;
    private readonly GeneratorOptions _options;
    private readonly string _modelPath;
    private readonly SemaphoreSlim _concurrencyLimiter;
    private readonly GgufMetadata? _ggufMetadata;
    private readonly string _serverVersion;
    private SelectionDiagnostics? _diagnostics;
    private readonly int _effectiveContextLength;
    private readonly int? _gpuLayers;
    private readonly int? _totalLayers;
    private readonly long? _estimatedVramBytes;
    private readonly long? _estimatedRamBytes;
    private readonly long? _vramBudgetBytes;
    private readonly long? _vramFreeBytes;
    private readonly long? _vramTotalBytes;
    private readonly bool _contextFlooredByVram;

    public void SetDiagnostics(SelectionDiagnostics diagnostics) => _diagnostics = diagnostics;
    private bool _disposed;

    private LlamaServerGeneratorModel(
        string modelId,
        string modelPath,
        ServerLease serverLease,
        IChatFormatter chatFormatter,
        GeneratorOptions options,
        int maxContextLength,
        int effectiveContextLength,
        GgufMetadata? ggufMetadata,
        string serverVersion,
        int? gpuLayers = null,
        int? totalLayers = null,
        long? estimatedVramBytes = null,
        long? estimatedRamBytes = null,
        long? vramBudgetBytes = null,
        long? vramFreeBytes = null,
        long? vramTotalBytes = null,
        bool contextFlooredByVram = false)
    {
        ModelId = modelId;
        _modelPath = modelPath;
        _serverLease = serverLease;
        _chatFormatter = chatFormatter;
        _options = options;
        MaxContextLength = maxContextLength;
        _effectiveContextLength = effectiveContextLength;
        _ggufMetadata = ggufMetadata;
        _serverVersion = serverVersion;
        _gpuLayers = gpuLayers;
        _totalLayers = totalLayers;
        _estimatedVramBytes = estimatedVramBytes;
        _estimatedRamBytes = estimatedRamBytes;
        _vramBudgetBytes = vramBudgetBytes;
        _vramFreeBytes = vramFreeBytes;
        _vramTotalBytes = vramTotalBytes;
        _contextFlooredByVram = contextFlooredByVram;

        // Initialize concurrency limiter
        _concurrencyLimiter = new SemaphoreSlim(
            Math.Max(1, options.MaxConcurrentRequests),
            Math.Max(1, options.MaxConcurrentRequests));
    }

    /// <summary>
    /// Loads a GGUF model using llama-server.
    /// </summary>
    public static async Task<LlamaServerGeneratorModel> LoadAsync(
        string modelId,
        string modelPath,
        IChatFormatter chatFormatter,
        GeneratorOptions options,
        IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        // 1. Get llama-server via update service (handles caching, updates, rollback)
        progress?.Report(new DownloadProgress
        {
            FileName = "llama-server",
            BytesDownloaded = 0,
            TotalBytes = 0,
            Phase = DownloadPhase.Downloading
        });

        var preferredBackend = global::LMSupply.Llama.LlamaBackendSelector.MapProvider(
            options.Provider, Hardware.HardwareProfile.Current.GpuInfo);
        var updateService = LlamaServerUpdateService.Resolve(options.ServerUpdateOptions);
        var updateResult = await updateService.GetServerPathAsync(
            preferredBackend,
            progress,
            cancellationToken);

        if (!updateResult.Success)
        {
            throw new InvalidOperationException(
                $"Failed to get llama-server: {updateResult.Error}");
        }

        var serverPath = updateResult.ServerPath;
        var backend = updateResult.Backend;
        var serverVersion = updateResult.NewVersion ?? updateResult.PreviousVersion;

        // 1b. Validate server version meets model requirements.
        // If the cached version is too old, trigger an immediate update and retry once
        // before throwing — avoids requiring manual cache deletion. When options.ServerUpdateOptions
        // pins a version, CheckAndApplyUpdateAsync intentionally short-circuits to the same pinned
        // binary (a pin must never be silently exceeded), so this retry is a no-op and Validate below
        // still throws — correctly, since the pin genuinely doesn't meet the requirement.
        if (!LlamaServerVersionRequirements.MeetsMinimum(serverVersion, chatFormatter.FormatName))
        {
            var retryResult = await updateService.CheckAndApplyUpdateAsync(
                preferredBackend, progress, cancellationToken);
            serverPath    = retryResult.ServerPath;
            backend       = retryResult.Backend;
            serverVersion = retryResult.NewVersion ?? retryResult.PreviousVersion;
        }

        LlamaServerVersionRequirements.Validate(serverVersion, chatFormatter.FormatName);

        // 2. Read GGUF metadata (best effort)
        GgufMetadata? ggufMetadata = null;
        try
        {
            ggufMetadata = await GgufMetadataReader.ReadAsync(modelPath, false, cancellationToken);
        }
        catch (Exception ex)
        {
            Trace.TraceInformation($"[LlamaServerGeneratorModel] GGUF metadata reading failed: {ex.Message}");
        }

        // 3. Configure and start llama-server
        progress?.Report(new DownloadProgress
        {
            FileName = Path.GetFileName(modelPath),
            BytesDownloaded = 30,
            TotalBytes = 100,
            Phase = DownloadPhase.Extracting
        });

        var llamaOpts = options.LlamaOptions ?? GetVramAwareLlamaOptions(modelPath, ggufMetadata);
        var contextLength = options.MaxContextLength ?? 4096;

        // Captured when partial offload occurs — exposed via GetModelInfo() for diagnostics.
        int? capturedGpuLayers = null;
        int? capturedTotalLayers = null;
        long? capturedVramBytes = null;
        long? capturedRamBytes = null;

        // VRAM-budget telemetry — exposed via GetModelInfo() so consumers can classify
        // "accurately-small VRAM" vs "under-reported budget" without parsing log magic numbers.
        long? capturedVramBudgetBytes = null;
        long? capturedVramFreeBytes = null;
        long? capturedVramTotalBytes = null;
        bool capturedContextFloored = false;

        // Auto-calculate GPU layer count based on actual VRAM budget when using default (-1 = all)
        if (llamaOpts.GpuLayerCount == -1 && backend != LlamaServerBackend.Cpu)
        {
            var fileSize = new FileInfo(modelPath).Length;
            var profile = Hardware.HardwareProfile.Current;
            // Use VramBudget so LMSUPPLY_VRAM_BUDGET_MB override + safety margins
            // flow into the offload decision. Raw EffectiveAvailableBytes ignored both.
            var budgetVram = VramBudget.GetAvailableBytes(profile.GpuInfo);
            var estimate = MemoryEstimator.EstimateForGguf(
                fileSize,
                contextLength,
                availableVramBytes: budgetVram > 0 ? budgetVram : profile.GpuInfo.EffectiveAvailableBytes,
                availableRamBytes: profile.SystemMemoryBytes);

            if (!estimate.CanFitInVram && estimate.RecommendedGpuLayers < estimate.TotalLayers)
            {
                capturedGpuLayers = estimate.RecommendedGpuLayers;
                capturedTotalLayers = estimate.TotalLayers;
                capturedVramBytes = estimate.EstimatedVramBytes;
                capturedRamBytes = estimate.EstimatedRamBytes;

                llamaOpts = CloneLlamaOptionsWithGpuLayers(llamaOpts, estimate.RecommendedGpuLayers);
                // Severity-aware trace: TraceWarning for full CPU fallback (0 GPU layers),
                // TraceInformation for partial offload. See LlamaOffloadTraceHelper.
                LlamaOffloadTraceHelper.TraceOffloadDecision(
                    estimate,
                    freeVramBytes: profile.GpuInfo.FreeMemoryBytes ?? 0,
                    totalVramBytes: profile.GpuInfo.TotalMemoryBytes ?? 0);
            }
        }

        // Auto-cap context length based on remaining VRAM after model load.
        // When the GPU can only offer the unusable floor (n_ctx clamped to 512 below request),
        // Auto recovers by falling back to CPU (RAM-bound, no VRAM clamp); an explicit GPU pin
        // fails fast instead of silently loading an unusable 512-token context.
        if (backend != LlamaServerBackend.Cpu)
        {
            var (safeContext, contextFloored) = EstimateSafeContextLengthDetailed(
                modelPath, contextLength, llamaOpts.GpuLayerCount ?? -1, ggufMetadata);

            // Capture VRAM telemetry before any CPU fallback switches the backend below.
            capturedContextFloored = contextFloored;
            var gpuInfo = Hardware.HardwareProfile.Current.GpuInfo;
            var vramBudget = VramBudget.GetAvailableBytes(gpuInfo);
            capturedVramBudgetBytes = vramBudget > 0 ? vramBudget : null;
            capturedVramTotalBytes = gpuInfo.TotalMemoryBytes;
            capturedVramFreeBytes = gpuInfo.FreeMemoryBytes ?? gpuInfo.TotalMemoryBytes;

            var action = DecideFlooredContextAction(options.Provider, backend, safeContext, contextLength);

            if (action == FlooredContextAction.FailFast)
            {
                throw new InvalidOperationException(
                    $"[LlamaServerGeneratorModel] GPU backend '{backend}' can only provide a {safeContext}-token context " +
                    $"(requested {contextLength}) — VRAM is insufficient for a usable context. " +
                    $"Pin ExecutionProvider.Cpu or free VRAM.");
            }

            if (action == FlooredContextAction.FallBackToCpu)
            {
                Trace.TraceWarning(
                    $"[LlamaServerGeneratorModel] Auto provider: GPU backend '{backend}' yields only {safeContext}-token context " +
                    $"(requested {contextLength}); falling back to CPU for a usable RAM-bound context.");

                // Re-acquire the CPU server binary and switch the backend. CPU is RAM-bound, so the
                // full requested context is kept (EstimateSafeContextLength returns it for GpuLayerCount=0).
                var cpuResult = await updateService.GetServerPathAsync(LlamaServerBackend.Cpu, progress, cancellationToken);
                if (cpuResult.Success)
                {
                    serverPath = cpuResult.ServerPath;
                    backend = LlamaServerBackend.Cpu;
                    serverVersion = cpuResult.NewVersion ?? cpuResult.PreviousVersion ?? serverVersion;
                    llamaOpts = CloneLlamaOptionsForCpuFallback(llamaOpts);
                }
                else
                {
                    // CPU binary unavailable — degrade to the floored context rather than fail the load.
                    Trace.TraceWarning(
                        $"[LlamaServerGeneratorModel] CPU fallback unavailable ({cpuResult.Error}); " +
                        $"using floored {safeContext}-token context.");
                    contextLength = safeContext;
                }
            }
            else if (safeContext < contextLength)
            {
                const double mb = 1024.0 * 1024.0;
                var gpu = Hardware.HardwareProfile.Current.GpuInfo;
                var budgetMb = VramBudget.GetAvailableBytes(gpu) / mb;
                var totalMb = (gpu.TotalMemoryBytes ?? 0) / mb;
                var freeMb = (gpu.FreeMemoryBytes ?? gpu.TotalMemoryBytes ?? 0) / mb;
                var ctxMsg = $"[LlamaServerGeneratorModel] ctx-size adjusted: requested={contextLength}, actual={safeContext} " +
                    $"(VRAM budget={budgetMb:F0}MB, free={freeMb:F0}MB, total={totalMb:F0}MB)";
                // Warn when the caller explicitly set MaxContextLength and it was silently reduced.
                if (options.MaxContextLength is not null)
                    Trace.TraceWarning(ctxMsg);
                else
                    Trace.TraceInformation(ctxMsg);
                contextLength = safeContext;
            }
        }

        // Build additional arguments
        var additionalArgs = BuildAdditionalArgs(llamaOpts, ggufMetadata?.Architecture);

        // Validate speculative decoding configuration
        if (llamaOpts.SpeculativeDecoding == SpeculativeDecodingMode.DraftModel
            && string.IsNullOrEmpty(llamaOpts.DraftModelPath))
        {
            throw new InvalidOperationException(
                "SpeculativeDecoding = DraftModel requires DraftModelPath to be set.");
        }

        // Validate YaRN configuration
        if (llamaOpts.RopeScaling == RopeScalingMode.YaRN && !llamaOpts.YarnOriginalContext.HasValue)
        {
            throw new InvalidOperationException(
                "RopeScaling = YaRN requires YarnOriginalContext to be set (original training context size, e.g., 4096).");
        }

        var serverConfig = new LlamaServerConfig
        {
            ModelPath = modelPath,
            Port = 0, // Auto-assign
            ContextSize = contextLength,
            GpuLayers = llamaOpts.GpuLayerCount ?? (backend == LlamaServerBackend.Cpu ? 0 : -1),
            BatchSize = (int)(llamaOpts.BatchSize ?? 512),
            UBatchSize = llamaOpts.UBatchSize.HasValue ? (int)llamaOpts.UBatchSize.Value : null,
            Parallel = Math.Max(1, options.MaxConcurrentRequests),
            FlashAttention = llamaOpts.FlashAttention ?? false,
            // Phase 1: KV cache quantization
            CacheTypeK = ResolveKvCacheType(llamaOpts.TypeK, backend, serverVersion),
            CacheTypeV = ResolveKvCacheType(llamaOpts.TypeV, backend, serverVersion),
            // Phase 1: Memory options
            UseMemoryMap = llamaOpts.UseMemoryMap,
            UseMemoryLock = llamaOpts.UseMemoryLock,
            // Phase 1: GPU options
            MainGpu = llamaOpts.MainGpu,
            // Phase 1: RoPE options
            RopeFreqBase = llamaOpts.RopeFrequencyBase,
            RopeFreqScale = llamaOpts.RopeFrequencyScale,
            // Speculative decoding
            SpecType = ResolveSpecType(llamaOpts.SpeculativeDecoding, serverVersion),
            ModelDraft = llamaOpts.SpeculativeDecoding == SpeculativeDecodingMode.DraftModel
                ? llamaOpts.DraftModelPath : null,
            // YaRN RoPE scaling
            RopeScaling         = MapRopeScaling(llamaOpts.RopeScaling),
            YarnOriginalContext  = llamaOpts.YarnOriginalContext,
            YarnExtensionFactor  = llamaOpts.YarnExtensionFactor,
            YarnAttentionFactor  = llamaOpts.YarnAttentionFactor,
            YarnBetaFast         = llamaOpts.YarnBetaFast,
            YarnBetaSlow         = llamaOpts.YarnBetaSlow,
            // Phase 3: Multimodal support
            MultimodalProjector = llamaOpts.MultimodalProjector,
            // Phase 3: LoRA support
            LoraPath = llamaOpts.LoraPath,
            LoraScale = llamaOpts.LoraScale,
            // Startup wait policy: the caller's values when given, otherwise LlamaServerConfig's
            // defaults (progress-based stall detection + a far-out absolute cap). No constant here —
            // the 120 s that used to live at this site (and in Embedder/Reranker) is what a cold
            // page-cache mmap load of a 4 GB model overran on a shared CI runner.
            StartupTimeout = llamaOpts.StartupTimeout ?? LlamaServerConfig.DefaultStartupTimeout,
            StartupStallTimeout = llamaOpts.StartupStallTimeout ?? LlamaServerConfig.DefaultStartupStallTimeout,
            ShutdownTimeout = TimeSpan.FromSeconds(10),
            AdditionalArgs = additionalArgs.Count > 0 ? additionalArgs : null
        };

        // 4. Lease server from pool with OOM retry (reduces GPU layers on failure)
        ServerLease serverLease;
        var currentGpuLayers = serverConfig.GpuLayers;

        while (true)
        {
            try
            {
                serverLease = await LlamaServerPool.Instance.LeaseAsync(
                    serverPath,
                    serverConfig,
                    backend,
                    progress,
                    cancellationToken);
                break; // Success
            }
            catch (Exception ex) when (IsOomError(ex) && currentGpuLayers > 0)
            {
                // Reduce GPU layers by ~25% (minimum 1 layer reduction)
                var reduction = Math.Max(1, currentGpuLayers / 4);
                currentGpuLayers -= reduction;

                Trace.TraceInformation(
                    $"[LlamaServerGeneratorModel] OOM detected, retrying with {currentGpuLayers} GPU layers " +
                    $"(was {serverConfig.GpuLayers}): {ex.Message}");

                serverConfig = CloneConfigWithGpuLayers(serverConfig, currentGpuLayers);
            }
        }

        progress?.Report(new DownloadProgress
        {
            FileName = Path.GetFileName(modelPath),
            BytesDownloaded = 100,
            TotalBytes = 100,
            Phase = DownloadPhase.Complete
        });

        // OOM retry may have further reduced GPU layers — update captured value.
        if (capturedGpuLayers.HasValue && currentGpuLayers != serverConfig.GpuLayers)
            capturedGpuLayers = currentGpuLayers;

        var model = new LlamaServerGeneratorModel(
            modelId,
            modelPath,
            serverLease,
            chatFormatter,
            options,
            SelectReportedContextLength(options, ggufMetadata, contextLength),
            contextLength,
            ggufMetadata,
            serverVersion ?? "unknown",
            capturedGpuLayers,
            capturedTotalLayers,
            capturedVramBytes,
            capturedRamBytes,
            capturedVramBudgetBytes,
            capturedVramFreeBytes,
            capturedVramTotalBytes,
            capturedContextFloored);

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

    /// <inheritdoc />
    public string ModelId { get; }

    /// <inheritdoc />
    public int MaxContextLength { get; }

    /// <inheritdoc />
    public IChatFormatter ChatFormatter => _chatFormatter;

    /// <inheritdoc />
    public bool IsGpuActive => _serverLease.Backend != LlamaServerBackend.Cpu;

    /// <inheritdoc />
    public IReadOnlyList<string> ActiveProviders => IsGpuActive
        ? new[] { $"llama-server-{_serverLease.Backend}", "CPU" }
        : new[] { "llama-server-CPU" };

    /// <inheritdoc />
    public ExecutionProvider RequestedProvider => _options.Provider;

    /// <inheritdoc />
    public long? EstimatedMemoryBytes => File.Exists(_modelPath) ? new FileInfo(_modelPath).Length * 2 : null;

    /// <summary>
    /// Gets the startup log from the llama-server process for diagnostics.
    /// </summary>
    public string? ServerStartupLog => _serverLease.Server.Info?.StartupLog;

    /// <inheritdoc />
    public async IAsyncEnumerable<string> GenerateAsync(
        string prompt,
        GenerationOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
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
                JsonSchema = options.JsonSchema
            };

            // Client-side token limit as safety net (server enforces too, but may fail)
            var maxTokens = completionOptions.MaxTokens;
            var tokenCount = 0;

            // Initialize reasoning token filter if needed
            var useReasoningFilter = options.FilterReasoningTokens || options.ExtractReasoningTokens;
            var reasoningFilter = useReasoningFilter
                ? new ReasoningTokenFilter(options.ExtractReasoningTokens)
                : null;

            await foreach (var token in _serverLease.Client.GenerateAsync(prompt, completionOptions, cancellationToken))
            {
                if (maxTokens > 0 && ++tokenCount > maxTokens)
                    break;

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
    public async IAsyncEnumerable<string> GenerateChatAsync(
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
            // Convert to llama-server format
            var augmentedMessages = MaybeInjectToolPromptFragment(trimmedMessages, options.Tools, _chatFormatter, options.Thinking == ThinkingMode.On);
            augmentedMessages = MaybeInjectThinkingToken(augmentedMessages, options.Thinking == ThinkingMode.On, _chatFormatter);
            var serverMessages = ConvertMessages(augmentedMessages);
            var chatOptions = CreateChatOptions(options);

            // Client-side token limit as safety net
            var maxTokens = options.ResolveMaxOutputTokens();
            var tokenCount = 0;

            // Initialize reasoning token filter if needed
            var useReasoningFilter = options.FilterReasoningTokens || options.ExtractReasoningTokens;
            var reasoningFilter = useReasoningFilter
                ? new ReasoningTokenFilter(options.ExtractReasoningTokens)
                : null;

            await foreach (var token in _serverLease.Client.GenerateChatAsync(serverMessages, chatOptions, cancellationToken))
            {
                if (maxTokens > 0 && ++tokenCount > maxTokens)
                    break;

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
        $"llama-server-{_serverLease.Backend}")
    {
        GgufMetadata = _ggufMetadata,
        BackendLog = _serverLease.Server.Info?.StartupLog,
        RuntimeVersion = _serverVersion,
        Diagnostics = _diagnostics,
        AdjustedContextLength = ResolveAdjustedContextLength(MaxContextLength, _effectiveContextLength),
        KnownIssues = GgufModelRegistry.Resolve(ModelId)?.KnownIssues ?? [],
        GpuLayers = _gpuLayers,
        TotalLayers = _totalLayers,
        EstimatedVramBytes = _estimatedVramBytes,
        EstimatedRamBytes = _estimatedRamBytes,
        VramBudgetBytes = _vramBudgetBytes,
        VramFreeBytes = _vramFreeBytes,
        VramTotalBytes = _vramTotalBytes,
        ContextFlooredByVram = _contextFlooredByVram,
    };

    internal static int? ResolveAdjustedContextLength(int maxContextLength, int effectiveContextLength)
        => effectiveContextLength != maxContextLength ? effectiveContextLength : null;

    /// <inheritdoc />
    public async Task<int> CountTokensAsync(string text, CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();

        if (string.IsNullOrEmpty(text))
            return 0;

        return await _serverLease.Client.CountTokensAsync(text, cancellationToken);
    }

    /// <inheritdoc />
    public async Task<int> CountTokensAsync(IEnumerable<ChatMessage> messages, CancellationToken cancellationToken = default)
    {
        var prompt = _chatFormatter.FormatPrompt(messages);
        return await CountTokensAsync(prompt, cancellationToken);
    }

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
            var augmentedMessages = MaybeInjectToolPromptFragment(trimmedMessages, options.Tools, _chatFormatter, options.Thinking == ThinkingMode.On);
            augmentedMessages = MaybeInjectThinkingToken(augmentedMessages, options.Thinking == ThinkingMode.On, _chatFormatter);
            var serverMessages = ConvertMessages(augmentedMessages);
            var chatOptions = CreateChatOptions(options);

            // Client-side token limit as safety net
            var maxTokens = options.ResolveMaxOutputTokens();
            var tokenCount = 0;

            // Initialize reasoning token filter if needed
            var useReasoningFilter = options.FilterReasoningTokens || options.ExtractReasoningTokens;
            var reasoningFilter = useReasoningFilter
                ? new ReasoningTokenFilter(options.ExtractReasoningTokens)
                : null;

            // Initialize formatter-supplied tool-call wrapper parser if any. When present AND
            // the formatter's grammar channel never produces a usable delta (the Gemma 4 case,
            // SuppressServerToolCallsWhenParserActive == true), the parser is the sole tool-call
            // source for the turn. When the formatter's grammar channel usually works (the
            // ChatML/Qwen case, == false), server deltas win per-chunk and the parser only fills
            // in the chunks where the server gave nothing — see ToolCallStreamCoexistence.
            // (ecosystem ISSUE Option D-5, 2026-05-01 — Gemma 4 wrapper extraction;
            // Option D-8, 2026-08-17 — ChatML coexist mode.)
            var toolStreamParser = _chatFormatter.CreateToolCallStreamParser();
            var suppressServerCallsWhenParserActive = _chatFormatter.SuppressServerToolCallsWhenParserActive;

            await foreach (var data in _serverLease.Client.GenerateChatStreamAsync(
                serverMessages, chatOptions, cancellationToken))
            {
                // Safety net: stop if token limit exceeded (finish_reason chunks still pass through)
                if (data.TextDelta is not null && maxTokens > 0 && ++tokenCount > maxTokens)
                {
                    yield return new ChatStreamChunk { FinishReason = "length" };
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
                {
                    if (suppressServerCallsWhenParserActive)
                    {
                        // Gemma 4 class: the parser is the sole source whenever it is registered.
                        if (text is not null)
                        {
                            var parsed = toolStreamParser.Feed(text);
                            text = parsed.Text;
                            if (parsed.ToolCalls is { Count: > 0 })
                            {
                                toolCallDeltas = parsed.ToolCalls;
                            }
                        }
                    }
                    else
                    {
                        // ChatML/Qwen class: server deltas win per chunk; the parser only fills
                        // in chunks where the server gave nothing.
                        var (resolvedText, resolvedCalls) =
                            ToolCallStreamCoexistence.Resolve(text, serverToolCallDeltas, toolStreamParser);
                        text = resolvedText;
                        toolCallDeltas = resolvedCalls;
                    }
                }

                // Yield structured chunk
                if (text is not null || reasoningDelta is not null || toolCallDeltas is not null || data.FinishReason is not null)
                {
                    yield return new ChatStreamChunk
                    {
                        Text = text,
                        ReasoningDelta = reasoningDelta,
                        ToolCalls = toolCallDeltas,
                        FinishReason = data.FinishReason
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
        }
        finally
        {
            _concurrencyLimiter.Release();
        }
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
            var augmentedMessages = MaybeInjectToolPromptFragment(trimmedMessages, options.Tools, _chatFormatter, options.Thinking == ThinkingMode.On);
            augmentedMessages = MaybeInjectThinkingToken(augmentedMessages, options.Thinking == ThinkingMode.On, _chatFormatter);
            var serverMessages = ConvertMessages(augmentedMessages);
            var chatOptions = CreateChatOptions(options);

            var response = await _serverLease.Client.GenerateChatWithToolsAsync(
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
                )).ToList()
            };
        }
        finally
        {
            _concurrencyLimiter.Release();
        }
    }

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
            Tools = options.Tools?.Select(t => new LMSupply.Llama.Server.ToolDefinition
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
            EnableThinking = ThinkingModeToEnableFlag(options.Thinking)
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
    /// Resolves KV cache type to llama-server CLI string.
    /// Auto selects based on backend and server version.
    /// </summary>
    /// <summary>
    /// Builds the raw llama-server CLI argument list derived from <see cref="LlamaOptions"/>
    /// properties that don't map to a dedicated <see cref="LlamaServerConfig"/> field
    /// (currently just Threads), an architecture-gated workaround (see below), plus any
    /// caller-supplied <see cref="LlamaOptions.AdditionalArgs"/> passthrough appended after them.
    /// </summary>
    /// <param name="llamaOpts">The caller's llama.cpp options.</param>
    /// <param name="ggufArchitecture">
    /// The loaded model's <c>general.architecture</c> GGUF field, if read. Used only to gate the
    /// <c>gemma4_assistant</c> workaround below.
    /// </param>
    internal static List<string> BuildAdditionalArgs(LlamaOptions llamaOpts, string? ggufArchitecture = null)
    {
        var args = new List<string>();
        if (llamaOpts.Threads.HasValue)
        {
            args.Add("--threads");
            args.Add(llamaOpts.Threads.Value.ToString(CultureInfo.InvariantCulture));
        }

        // llama.cpp's automatic memory-fitting (-fit, on by default) throws "Gemma4Assistant
        // requires ctx_other to be set" for this architecture — a confirmed upstream bug
        // (ggml-org/llama.cpp#24343, fix pending as of PR #24590, unreleased as of this writing).
        // Disabling fitting is safe here specifically: lm-supply already sets ctx-size/gpu-layers/
        // batch-size explicitly (never relies on auto-fit to choose them) and has its own
        // OOM-retry fallback (see LoadAsync's lease-retry loop), so this narrows to just the one
        // architecture known to crash rather than disabling fitting universally.
        if (string.Equals(ggufArchitecture, "gemma4_assistant", StringComparison.OrdinalIgnoreCase))
        {
            args.Add("-fit");
            args.Add("off");
        }

        if (llamaOpts.AdditionalArgs is { Count: > 0 })
            args.AddRange(llamaOpts.AdditionalArgs);

        return args;
    }

    internal static string? ResolveKvCacheType(
        KvCacheQuantizationType type,
        LlamaServerBackend backend,
        string? serverVersion)
    {
        if (type == KvCacheQuantizationType.Auto)
            type = SelectAutoKvCache(backend, serverVersion);

        return type switch
        {
            KvCacheQuantizationType.Q8_0 => "q8_0",
            KvCacheQuantizationType.Q4_0 => "q4_0",
            KvCacheQuantizationType.F32  => "f32",
            _                            => null   // F16 = llama-server default
        };
    }

    private static KvCacheQuantizationType SelectAutoKvCache(LlamaServerBackend backend, string? serverVersion)
    {
        return backend switch
        {
            LlamaServerBackend.Cuda12 or LlamaServerBackend.Cuda13
                or LlamaServerBackend.Metal or LlamaServerBackend.Hip
                => KvCacheQuantizationType.Q8_0,
            LlamaServerBackend.Vulkan
                => IsFeatureSupported("kv-q8-vulkan", serverVersion)
                    ? KvCacheQuantizationType.Q8_0
                    : KvCacheQuantizationType.F16,
            _ => KvCacheQuantizationType.F16   // Cpu, Sycl
        };
    }

    /// <summary>
    /// Resolves speculative decoding mode to llama-server --spec-type value.
    /// </summary>
    internal static string? ResolveSpecType(
        SpeculativeDecodingMode mode,
        string? serverVersion)
    {
        // b8994+ renamed --spec-type ngram to ngram-simple.
        // Auto probes the newer name first, falls back to the old name for b8500-b8993.
        bool hasNgramSimple = IsFeatureSupported("spec-ngram-simple", serverVersion);
        bool hasNgram       = IsFeatureSupported("spec-ngram",        serverVersion);

        return mode switch
        {
            SpeculativeDecodingMode.None       => null,
            SpeculativeDecodingMode.Ngram      => hasNgramSimple ? "ngram-simple" : "ngram",
            SpeculativeDecodingMode.DraftModel => null,  // handled via ModelDraft
            SpeculativeDecodingMode.Auto       => hasNgramSimple ? "ngram-simple" :
                                                  hasNgram       ? "ngram"        : null,
            _ => null
        };
    }

    private static bool IsFeatureSupported(string featureKey, string? serverVersion)
    {
        var build = LlamaServerVersionRequirements.ParseBuildNumber(serverVersion);
        var minBuild = LlamaServerVersionRequirements.GetMinimumBuild(featureKey);
        return build.HasValue && minBuild.HasValue && build.Value >= minBuild.Value;
    }

    private static string? MapRopeScaling(RopeScalingMode mode) => mode switch
    {
        RopeScalingMode.Linear   => "linear",
        RopeScalingMode.YaRN    => "yarn",
        RopeScalingMode.LongRoPE => "longrope",
        _                        => null  // Default = passthrough
    };

    /// <summary>
    /// Estimates the maximum safe context length based on VRAM remaining after model weights.
    /// Returns the original requestedContext if it fits, otherwise a capped value.
    /// For MoE models, applies an additional overhead margin to account for expert activation
    /// buffers and compute scratch space that llama.cpp pre-allocates independently of KV cache.
    /// </summary>
    internal static int EstimateSafeContextLength(
        string modelPath,
        int requestedContext,
        int gpuLayerCount,
        GgufMetadata? ggufMetadata = null)
        => EstimateSafeContextLengthDetailed(modelPath, requestedContext, gpuLayerCount, ggufMetadata).Context;

    /// <summary>
    /// Same VRAM-aware context estimate as <see cref="EstimateSafeContextLength"/>, but also reports
    /// whether the estimate was floored — i.e. the raw VRAM-derived value fell below
    /// <see cref="UnusableContextFloorTokens"/> and was raised to that floor. <c>Floored == true</c>
    /// means VRAM is insufficient for a usable context (the brick signal), distinct from a
    /// legitimately small request. Returns <c>Floored == false</c> on the CPU path.
    /// </summary>
    internal static (int Context, bool Floored) EstimateSafeContextLengthDetailed(
        string modelPath,
        int requestedContext,
        int gpuLayerCount,
        GgufMetadata? ggufMetadata = null)
    {
        var profile = Hardware.HardwareProfile.Current;
        // Use VramBudget so context cap honors LMSUPPLY_VRAM_BUDGET_MB override + safety margins.
        var budgetVram = VramBudget.GetAvailableBytes(profile.GpuInfo);
        var availableVram = budgetVram > 0 ? budgetVram : (profile.GpuInfo.EffectiveAvailableBytes ?? 0);
        if (availableVram <= 0 || gpuLayerCount == 0)
            return (requestedContext, false); // CPU-only, no VRAM constraint

        // MoE models (ExpertCount > 1) require significant additional VRAM for expert activation
        // buffers, routing computation, and compute scratch space. llama.cpp pre-allocates these
        // independently of the KV cache, and they are not captured by the weight-size estimate.
        // Empirical observation: RTX 3090 + Gemma 4 26B A4B Q4_K_M loses ~25% of effective VRAM
        // to MoE overhead. Apply a conservative 0.80 multiplier to budget accordingly.
        const double moeBudgetFactor = 0.80;
        if (ggufMetadata?.ExpertCount > 1)
            availableVram = (long)(availableVram * moeBudgetFactor);

        var modelFileSize = new FileInfo(modelPath).Length;
        var modelMemory = (long)(modelFileSize * 1.1);
        const long vramBuffer = 512L * 1024 * 1024; // 500MB safety buffer
        var remainingVram = Math.Max(0, availableVram - modelMemory - vramBuffer);

        // KV cache per token ≈ 2(K+V) × layers × hiddenSize × 2(FP16 bytes)
        var kvBytesPerToken = Core.Download.AvailableMemory.EstimateKvCacheBytes(modelFileSize, 1);
        if (kvBytesPerToken <= 0)
            return (requestedContext, false);

        var rawSafeContext = (int)(remainingVram / kvBytesPerToken);
        var floored = rawSafeContext < UnusableContextFloorTokens;
        var safeContext = Math.Max(UnusableContextFloorTokens, rawSafeContext); // minimum 512 tokens

        return (Math.Min(requestedContext, safeContext), floored);
    }

    /// <summary>
    /// The context-length floor (tokens) that <see cref="EstimateSafeContextLength"/> never goes below.
    /// A safe-context estimate at or below this floor that is also below the requested size means the
    /// GPU backend cannot offer a usable context (VRAM exhausted) — the brick threshold consumers reject.
    /// </summary>
    internal const int UnusableContextFloorTokens = 512;

    /// <summary>What to do when a GPU context estimate is clamped to an unusable floor.</summary>
    internal enum FlooredContextAction
    {
        /// <summary>Context is usable (or already CPU) — load as-is.</summary>
        Proceed,
        /// <summary>Provider was Auto: transparently fall back to CPU (RAM-bound, no VRAM clamp).</summary>
        FallBackToCpu,
        /// <summary>Provider was an explicit GPU pin: surface an error instead of silently loading 512.</summary>
        FailFast
    }

    /// <summary>
    /// Decides how to recover when the VRAM-aware context estimate is floored to an unusable value.
    /// Pure function (no HW access) so the policy is unit-testable by passing a low <paramref name="safeContext"/>.
    /// </summary>
    /// <param name="requestedProvider">The provider the caller requested (<see cref="ExecutionProvider.Auto"/> by default).</param>
    /// <param name="backend">The GPU backend that was actually selected.</param>
    /// <param name="safeContext">Result of <see cref="EstimateSafeContextLength"/>.</param>
    /// <param name="requestedContext">The context length the caller asked for.</param>
    internal static FlooredContextAction DecideFlooredContextAction(
        ExecutionProvider requestedProvider,
        LlamaServerBackend backend,
        int safeContext,
        int requestedContext)
    {
        // Floored == GPU backend offered only the unusable floor, below what was requested.
        var floored = backend != LlamaServerBackend.Cpu
            && safeContext <= UnusableContextFloorTokens
            && safeContext < requestedContext;
        if (!floored)
            return FlooredContextAction.Proceed;

        // Auto promised a *working* provider — recover to CPU. An explicit GPU pin must fail honestly.
        return requestedProvider == ExecutionProvider.Auto
            ? FlooredContextAction.FallBackToCpu
            : FlooredContextAction.FailFast;
    }

    /// <summary>
    /// Copies <paramref name="src"/> with GPU offload disabled (CPU-only). Used when Auto falls back
    /// to CPU after a floored GPU context — does not mutate the caller-supplied options object.
    /// </summary>
    internal static LlamaOptions CloneLlamaOptionsForCpuFallback(LlamaOptions src) =>
        CloneLlamaOptionsWithGpuLayers(src, gpuLayers: 0);

    /// <summary>
    /// Copies <paramref name="src"/> with <see cref="LlamaOptions.GpuLayerCount"/> replaced and
    /// <see cref="LlamaOptions.GpuOffloadRatio"/> cleared (it would otherwise override the count).
    /// Every other property is carried over — the VRAM auto-tune and the CPU fallback both go
    /// through here, so a property added to <see cref="LlamaOptions"/> has one place to be copied,
    /// and <c>LlamaOptionsCloneCompletenessTests</c> fails when it is not.
    /// </summary>
    internal static LlamaOptions CloneLlamaOptionsWithGpuLayers(LlamaOptions src, int gpuLayers) => new()
    {
        GpuLayerCount = gpuLayers,
        GpuOffloadRatio = null,
        BatchSize = src.BatchSize,
        UBatchSize = src.UBatchSize,
        RopeFrequencyBase = src.RopeFrequencyBase,
        RopeFrequencyScale = src.RopeFrequencyScale,
        FlashAttention = src.FlashAttention,
        UseMemoryMap = src.UseMemoryMap,
        UseMemoryLock = src.UseMemoryLock,
        MainGpu = src.MainGpu,
        Threads = src.Threads,
        TypeK = src.TypeK,
        TypeV = src.TypeV,
        SpeculativeDecoding = src.SpeculativeDecoding,
        DraftModelPath = src.DraftModelPath,
        RopeScaling = src.RopeScaling,
        YarnOriginalContext = src.YarnOriginalContext,
        YarnExtensionFactor = src.YarnExtensionFactor,
        YarnAttentionFactor = src.YarnAttentionFactor,
        YarnBetaFast = src.YarnBetaFast,
        YarnBetaSlow = src.YarnBetaSlow,
        MultimodalProjector = src.MultimodalProjector,
        LoraPath = src.LoraPath,
        LoraScale = src.LoraScale,
        StartupTimeout = src.StartupTimeout,
        StartupStallTimeout = src.StartupStallTimeout,
        AdditionalArgs = src.AdditionalArgs,
    };

    /// <summary>
    /// Creates a copy of a LlamaServerConfig with a different GpuLayers value.
    /// </summary>
    /// <remarks>
    /// Every property except <see cref="LlamaServerConfig.GpuLayers"/> is carried over;
    /// <c>LlamaOptionsCloneCompletenessTests</c> fails when a new one is not. The OOM-retry clone
    /// silently dropped <see cref="LlamaServerConfig.RequestTimeout"/> before that test existed.
    /// </remarks>
    internal static LlamaServerConfig CloneConfigWithGpuLayers(LlamaServerConfig source, int gpuLayers) => new()
    {
        ModelPath = source.ModelPath,
        Port = source.Port,
        ContextSize = source.ContextSize,
        GpuLayers = gpuLayers,
        BatchSize = source.BatchSize,
        UBatchSize = source.UBatchSize,
        Parallel = source.Parallel,
        FlashAttention = source.FlashAttention,
        CacheTypeK = source.CacheTypeK,
        CacheTypeV = source.CacheTypeV,
        UseMemoryMap = source.UseMemoryMap,
        UseMemoryLock = source.UseMemoryLock,
        MainGpu = source.MainGpu,
        RopeFreqBase = source.RopeFreqBase,
        RopeFreqScale = source.RopeFreqScale,
        SpecType            = source.SpecType,
        ModelDraft          = source.ModelDraft,
        RopeScaling         = source.RopeScaling,
        YarnOriginalContext = source.YarnOriginalContext,
        YarnExtensionFactor = source.YarnExtensionFactor,
        YarnAttentionFactor = source.YarnAttentionFactor,
        YarnBetaFast        = source.YarnBetaFast,
        YarnBetaSlow        = source.YarnBetaSlow,
        MultimodalProjector = source.MultimodalProjector,
        LoraPath = source.LoraPath,
        LoraScale = source.LoraScale,
        Mode = source.Mode,
        Pooling = source.Pooling,
        StartupTimeout = source.StartupTimeout,
        StartupStallTimeout = source.StartupStallTimeout,
        ShutdownTimeout = source.ShutdownTimeout,
        RequestTimeout = source.RequestTimeout,
        AdditionalArgs = source.AdditionalArgs
    };

    /// <summary>
    /// Checks if an exception is an out-of-memory error from the GPU runtime.
    /// </summary>
    internal static bool IsOomError(Exception ex)
    {
        var msg = ex.Message;
        return msg.Contains("out of memory", StringComparison.OrdinalIgnoreCase)
            || msg.Contains("CUDA_ERROR_OUT_OF_MEMORY", StringComparison.OrdinalIgnoreCase)
            || msg.Contains("Could not allocate")
            || msg.Contains("ggml_backend_cuda_buffer_type_alloc_buffer");
    }

    /// <summary>
    /// Gets VRAM-aware LlamaOptions using actual model size and GPU information.
    /// Uses GgufModelInfo metadata when available, falls back to file-size estimation.
    /// </summary>
    private static LlamaOptions GetVramAwareLlamaOptions(string modelPath, GgufMetadata? ggufMetadata)
    {
        var gpu = Hardware.HardwareProfile.Current.GpuInfo;

        // Determine model size: prefer GGUF metadata, fall back to file size
        long modelSizeBytes;
        if (ggufMetadata is not null)
        {
            // Use file size as the most accurate measure of on-disk model weight size
            modelSizeBytes = new FileInfo(modelPath).Length;
        }
        else
        {
            // No metadata: estimate from file size with runtime overhead factor
            var fileSize = new FileInfo(modelPath).Length;
            modelSizeBytes = (long)(fileSize * 1.1); // ~10% runtime overhead
        }

        // Determine total layers from metadata or estimate from file size
        var totalLayers = ggufMetadata?.LayerCount ?? EstimateTotalLayers(new FileInfo(modelPath).Length);

        return LlamaOptions.GetOptimalForHardware(gpu, modelSizeBytes, totalLayers);
    }

    /// <summary>
    /// Estimates total layer count from GGUF file size when metadata is not available.
    /// </summary>
    internal static int EstimateTotalLayers(long fileSizeBytes)
    {
        return fileSizeBytes switch
        {
            < 2L * 1024 * 1024 * 1024 => 22,
            < 5L * 1024 * 1024 * 1024 => 28,
            < 10L * 1024 * 1024 * 1024 => 32,
            _ => 40
        };
    }

    /// <summary>
    /// Conditionally prepends a textual reinforcement of the active tool schemas as a
    /// system message when the formatter opts in via
    /// <see cref="IChatFormatter.RenderToolPromptFragment"/>. Small/quantized models
    /// (Gemma 4 E4B at gguf:gemma4-default) misinterpret llama-server's raw JSON-schema
    /// rendering and emit empty tool args; the textual fragment raises first-attempt
    /// success (ecosystem ISSUE Option D-1, 2026-04-30).
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
    /// Selects the context length to report as the model's capability.
    /// Priority: explicit user cap &gt; GGUF metadata capability &gt; VRAM-capped session budget.
    /// </summary>
    internal static int SelectReportedContextLength(
        GeneratorOptions options,
        GgufMetadata? ggufMetadata,
        int vramCappedBudget)
        => options.MaxContextLength ?? ggufMetadata?.ContextLength ?? vramCappedBudget;

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
            var augmented = MaybeInjectToolPromptFragment(list, options.Tools, _chatFormatter, options.Thinking == ThinkingMode.On);
            augmented = MaybeInjectThinkingToken(augmented, options.Thinking == ThinkingMode.On, _chatFormatter);
            var prompt = _chatFormatter.FormatPrompt(augmented);
            var tokenCount = await _serverLease.Client.CountTokensAsync(prompt, cancellationToken);

            if (tokenCount <= inputBudget)
                return list;

            if (!TryTrimOldestTurn(list))
                throw new ContextLengthExceededException(tokenCount, MaxContextLength);
        }
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
        await _serverLease.DisposeAsync();
    }
}
