using System.Diagnostics;
using System.Globalization;
using LMSupply.Hardware;
using LMSupply.Llama.Server;
using LMSupply.Runtime;

namespace LMSupply.Generator.Internal.Llama;

internal sealed partial class LlamaServerGeneratorModel
{
    internal static int? ResolveAdjustedContextLength(int requestedContextLength, int effectiveContextLength)
        => effectiveContextLength != requestedContextLength ? effectiveContextLength : null;

    /// <summary>The most a context the caller did not choose asks for, whatever the model was trained to.</summary>
    internal const int DefaultContextCeiling = 32768;

    /// <summary>
    /// The context to ask llama-server for: the caller's <see cref="GeneratorOptions.MaxContextLength"/> when set;
    /// otherwise the model's trained length (GGUF <c>context_length</c>, 4096 when the file does not say) up to
    /// <see cref="DefaultContextCeiling"/>. Memory then bounds it on the backend used (VRAM on GPU, RAM on CPU).
    /// </summary>
    internal static int ResolveRequestedContextLength(int? explicitLength, int? trainedLength)
        => explicitLength ?? Math.Min(trainedLength is > 0 ? trainedLength.Value : 4096, DefaultContextCeiling);

    /// <summary>
    /// The largest context whose KV cache fits in half the machine's RAM after the weights, up to
    /// <paramref name="requestedContext"/> and never below 4096 (or the request, when smaller). Half, because RAM is
    /// shared with the host process and everything else on the machine; the KV cache is sized as llama.cpp sizes it
    /// when the file has attention metadata (<see cref="KvCacheGeometry"/>), else by a file-size heuristic.
    /// </summary>
    internal static int EstimateRamBoundContextLength(
        long modelFileSize,
        int requestedContext,
        GgufMetadata? ggufMetadata,
        string? cacheTypeK,
        string? cacheTypeV,
        int sequences,
        int ubatch,
        long systemMemoryBytes)
    {
        var floor = Math.Min(4096, requestedContext);
        if (systemMemoryBytes <= 0)
            return requestedContext;

        var budget = systemMemoryBytes / 2 - (long)(modelFileSize * 1.1);
        long kvBytesPerToken;
        var geometry = KvCacheGeometry.FromMetadata(ggufMetadata);
        if (geometry is not null)
        {
            budget -= geometry.SlidingWindowBytes(requestedContext, cacheTypeK, cacheTypeV, sequences, ubatch);
            kvBytesPerToken = geometry.BytesPerToken(cacheTypeK, cacheTypeV);
        }
        else
        {
            kvBytesPerToken = Core.Download.AvailableMemory.EstimateKvCacheBytes(modelFileSize, 1);
        }

        if (kvBytesPerToken <= 0)
            return requestedContext;

        var fits = budget <= 0 ? 0 : budget / kvBytesPerToken;
        return (int)Math.Clamp(fits, floor, requestedContext);
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
    /// <param name="modelPath">The GGUF file (its size stands in for the weights).</param>
    /// <param name="requestedContext">The context the caller asked for.</param>
    /// <param name="gpuLayerCount">GPU layers (0 = CPU, no VRAM constraint; -1 = all).</param>
    /// <param name="ggufMetadata">The file's metadata. With attention metadata the KV cache is sized as llama.cpp
    /// sizes it (<see cref="KvCacheGeometry"/>); without it a file-size heuristic stands in.</param>
    /// <param name="cacheTypeK">The <c>--cache-type-k</c> the server will run with (null = f16).</param>
    /// <param name="cacheTypeV">The <c>--cache-type-v</c> the server will run with (null = f16).</param>
    /// <param name="sequences">Parallel sequences (server slots) — sliding-window layers keep a window per sequence.</param>
    /// <param name="ubatch">The physical batch size — sliding-window layers keep one batch beyond the window.</param>
    /// <param name="gpu">The GPU as read for this load; <see cref="HardwareProfile.Current"/>'s when null.</param>
    internal static (int Context, bool Floored) EstimateSafeContextLengthDetailed(
        string modelPath,
        int requestedContext,
        int gpuLayerCount,
        GgufMetadata? ggufMetadata = null,
        string? cacheTypeK = null,
        string? cacheTypeV = null,
        int sequences = 1,
        int ubatch = KvCacheGeometry.DefaultUBatch,
        GpuInfo? gpu = null)
    {
        gpu ??= Hardware.HardwareProfile.Current.GpuInfo;
        // Use VramBudget so context cap honors LMSUPPLY_VRAM_BUDGET_MB override + safety margins.
        var budgetVram = VramBudget.GetAvailableBytes(gpu);
        var availableVram = budgetVram > 0 ? budgetVram : (gpu.EffectiveAvailableBytes ?? 0);
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

        var modelFileSize = CacheManager.GetContentLength(modelPath);
        var modelMemory = (long)(modelFileSize * 1.1);
        const long vramBuffer = 512L * 1024 * 1024; // 500MB safety buffer
        var remainingVram = Math.Max(0, availableVram - modelMemory - vramBuffer);

        // KV cache: from the file's attention metadata when it has it (layers that keep a cache × KV heads ×
        // K/V head dims × cache-type bytes, plus a fixed window for sliding-window layers); otherwise a
        // file-size heuristic, which assumes every attention head keeps K and V and so over-sizes
        // grouped-query models several times over.
        long kvBytesPerToken;
        var geometry = KvCacheGeometry.FromMetadata(ggufMetadata);
        if (geometry is not null)
        {
            remainingVram = Math.Max(0, remainingVram
                - geometry.SlidingWindowBytes(requestedContext, cacheTypeK, cacheTypeV, sequences, ubatch));
            kvBytesPerToken = geometry.BytesPerToken(cacheTypeK, cacheTypeV);
        }
        else
        {
            kvBytesPerToken = Core.Download.AvailableMemory.EstimateKvCacheBytes(modelFileSize, 1);
        }

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
    /// The llama options a load starts from: the caller's, or hardware-fitted defaults when none were
    /// given — with any <see cref="LlamaOptions.GpuOffloadRatio"/> already turned into a layer count,
    /// because everything after this point (the VRAM fit, the context estimate, the server launch) reads
    /// the count.
    /// </summary>
    /// <summary>
    /// Whether the GPU layer count is left to the VRAM fit: "all layers" on a GPU backend. Unset counts as
    /// -1, as it does for the context estimate and the server launch — a caller that passes
    /// <see cref="LlamaOptions"/> to change one knob (a timeout, the thread count) must not lose the fit and
    /// offload every layer regardless of VRAM.
    /// </summary>
    internal static bool NeedsVramFit(LlamaOptions opts, LlamaServerBackend backend) =>
        (opts.GpuLayerCount ?? -1) == -1 && backend != LlamaServerBackend.Cpu;

    internal static LlamaOptions ChooseLlamaOptions(GeneratorOptions options, string modelPath, GgufMetadata? ggufMetadata)
        => ResolveGpuOffloadRatio(
            options.LlamaOptions ?? GetVramAwareLlamaOptions(modelPath, ggufMetadata, options.Provider),
            ggufMetadata?.LayerCount);

    /// <summary>
    /// Turns <see cref="LlamaOptions.GpuOffloadRatio"/> into a layer count for this model, so the server
    /// launch — which takes a count — honours it (the ratio takes precedence over
    /// <see cref="LlamaOptions.GpuLayerCount"/>, as documented). Returns <paramref name="opts"/> unchanged
    /// when no ratio is set. 0 means no layers and 1 means all (-1, like an explicit
    /// <c>GpuLayerCount = -1</c>, so the VRAM fit may still reduce it); a ratio in between needs the model's
    /// layer count, and when the GGUF header does not carry one it cannot be applied — all layers are
    /// offloaded instead, with a warning.
    /// </summary>
    internal static LlamaOptions ResolveGpuOffloadRatio(LlamaOptions opts, int? totalLayers)
    {
        if (opts.GpuOffloadRatio is not { } ratio)
            return opts;

        if (float.IsNaN(ratio))
            throw new ArgumentException("LlamaOptions.GpuOffloadRatio must be a number from 0 to 1, not NaN.", nameof(opts));

        if (ratio > 0f && ratio < 1f && totalLayers is not > 0)
        {
            Trace.TraceWarning(
                $"[LlamaServerGeneratorModel] GpuOffloadRatio={ratio.ToString("0.##", CultureInfo.InvariantCulture)} " +
                "needs the model's layer count, which its GGUF header does not carry; offloading all layers instead.");
            return CloneLlamaOptionsWithGpuLayers(opts, gpuLayers: -1);
        }

        return CloneLlamaOptionsWithGpuLayers(opts, opts.GetEffectiveGpuLayerCount(totalLayers ?? 0));
    }

    /// <summary>
    /// Copies <paramref name="src"/> with GPU offload disabled (CPU-only). Used when Auto falls back
    /// to CPU after a floored GPU context — does not mutate the caller-supplied options object.
    /// </summary>
    internal static LlamaOptions CloneLlamaOptionsForCpuFallback(LlamaOptions src) =>
        CloneLlamaOptionsWithGpuLayers(src, gpuLayers: 0);

    /// <summary>
    /// Copies <paramref name="src"/> with <see cref="LlamaOptions.GpuLayerCount"/> replaced and
    /// <see cref="LlamaOptions.GpuOffloadRatio"/> cleared — by the time a count is chosen here the ratio has
    /// already been turned into one (<see cref="ResolveGpuOffloadRatio"/>), and a ratio left beside the new
    /// count would claim precedence over it.
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
        RequestTimeout = src.RequestTimeout,
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
        ServerVersion = source.ServerVersion,
        StartupTimeout = source.StartupTimeout,
        StartupStallTimeout = source.StartupStallTimeout,
        ShutdownTimeout = source.ShutdownTimeout,
        RequestTimeout = source.RequestTimeout,
        AdditionalArgs = source.AdditionalArgs,
        RequireApiKey = source.RequireApiKey
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
    private static LlamaOptions GetVramAwareLlamaOptions(string modelPath, GgufMetadata? ggufMetadata, ExecutionProvider provider)
    {
        // An explicit CPU provider offloads nothing, and must not probe the GPU to decide that.
        var gpu = Hardware.HardwareProfile.For(provider).GpuInfo;

        // Determine model size: prefer GGUF metadata, fall back to file size
        long modelSizeBytes;
        if (ggufMetadata is not null)
        {
            // Use file size as the most accurate measure of on-disk model weight size
            modelSizeBytes = CacheManager.GetContentLength(modelPath);
        }
        else
        {
            // No metadata: estimate from file size with runtime overhead factor
            var fileSize = CacheManager.GetContentLength(modelPath);
            modelSizeBytes = (long)(fileSize * 1.1); // ~10% runtime overhead
        }

        // Determine total layers from metadata or estimate from file size
        var totalLayers = ggufMetadata?.LayerCount ?? EstimateTotalLayers(CacheManager.GetContentLength(modelPath));

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
}
