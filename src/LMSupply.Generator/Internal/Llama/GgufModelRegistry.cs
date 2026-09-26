using LMSupply.Hardware;
using LMSupply.Runtime;

namespace LMSupply.Generator.Internal.Llama;

/// <summary>
/// Registry of well-known GGUF models with aliases for easy access.
/// Follows the same alias pattern as the ONNX ModelRegistry.
/// </summary>
public static class GgufModelRegistry
{
    private static readonly Dictionary<string, GgufModelInfo> _models = new(StringComparer.OrdinalIgnoreCase)
    {
        // ============================================================
        // Gemma 4 중심 레지스트리 (Apache 2.0, 멀티모달, 네이티브 function calling)
        // llama.cpp b8672+ 에서 Gemma 4 네이티브 지원 (GGUF 메타데이터 자동 감지)
        // ============================================================

        // Fast: Gemma 4 E2B — smallest Gemma 4, fits 4GB iGPU/mobile (~3.1GB VRAM).
        // Q4_K_M is the only tier that fits 4GB VRAM (Q8_0=4.8GB, Q5_K_M=3.2GB w/ no KV margin).
        // Tool-call schema compliance at Q4_K_M is lower than on larger models — prefer gguf:gemma4-default
        // or above for agentic workloads. Auto-selection picks E2B only when E4B doesn't fit.
        ["gguf:gemma4-fast"] = new GgufModelInfo
        {
            RepoId = "unsloth/gemma-4-E2B-it-GGUF",
            DisplayName = "Gemma 4 E2B Instruct",
            DefaultFile = "gemma-4-E2B-it-Q4_K_M.gguf",
            ChatFormat = "gemma4",
            ContextLength = 131072,
            ParameterCount = 2_300_000_000,
            EstimatedSizeBytes = 3_110_000_000L,
            QuantizationType = "Q4_K_M",
            NumLayers = 26,
            HiddenSize = 2304,
            License = LicenseTier.MIT,
            LicenseName = "Apache 2.0",
            KnownIssues = [GgufModelKnownIssues.ToolUseUnreliableQ4, GgufModelKnownIssues.InstructionFollowingUnreliableQ4],
        },

        // Default: Gemma 4 E4B — best balance of size, speed, and quality
        // NOTE (2026-08-17): this repo never published a K-quant variant — only BF16/Q4_0/Q8_0
        // (plus mmproj-*/mtp-* companion files, both excluded from selection — see
        // GgufModelDownloader.IsMmprojFile/IsMtpFile). The registry's DefaultFile previously named
        // "gemma-4-E4B-it-Q4_K_M.gguf", which never existed upstream; on hosts where that mismatch
        // caused a size/quant fallback, the (then-unfiltered) mtp-* companion could be selected
        // instead of a real model and crash llama-server (ggml-org/llama.cpp#24343). Verified via
        // huggingface.co/ggml-org/gemma-4-E4B-it-GGUF file listing before changing this.
        ["gguf:gemma4-default"] = new GgufModelInfo
        {
            RepoId = "ggml-org/gemma-4-E4B-it-GGUF",
            DisplayName = "Gemma 4 E4B Instruct",
            DefaultFile = "gemma-4-E4B-it-Q4_0.gguf",
            ChatFormat = "gemma4",
            ContextLength = 131072,
            ParameterCount = 4_500_000_000,
            EstimatedSizeBytes = 4_590_807_392L,
            QuantizationType = "Q4_0",
            NumLayers = 34,
            HiddenSize = 2560,
            License = LicenseTier.MIT,
            LicenseName = "Apache 2.0",
            KnownIssues = [GgufModelKnownIssues.ToolUseUnreliableQ4, GgufModelKnownIssues.InstructionFollowingUnreliableQ4],
        },

        // Balanced: Gemma 4 E4B Q8_0 — higher quality E4B for 10-12GB VRAM (RTX 3060 12GB, etc.)
        // Fills the gap between default (5.3GB) and quality (16.8GB)
        ["gguf:gemma4-balanced"] = new GgufModelInfo
        {
            RepoId = "ggml-org/gemma-4-E4B-it-GGUF",
            DisplayName = "Gemma 4 E4B Instruct (Q8_0)",
            DefaultFile = "gemma-4-E4B-it-Q8_0.gguf",
            ChatFormat = "gemma4",
            ContextLength = 131072,
            ParameterCount = 4_500_000_000,
            EstimatedSizeBytes = 7_500_000_000L,
            QuantizationType = "Q8_0",
            NumLayers = 34,
            HiddenSize = 2560,
            License = LicenseTier.MIT,
            LicenseName = "Apache 2.0",
            KnownIssues = [GgufModelKnownIssues.ToolUseUnreliableQ4, GgufModelKnownIssues.InstructionFollowingUnreliableQ4],
        },

        // Quality: Gemma 4 26B MoE — 31B-class performance with 4B active params
        // NOTE (2026-08-17): this repo also never published a K-quant — see gemma4-default's note
        // above for the shared root cause (ISSUE-lm-supply-20260817-gemma4-ctx-other-...).
        ["gguf:gemma4-quality"] = new GgufModelInfo
        {
            RepoId = "ggml-org/gemma-4-26B-A4B-it-GGUF",
            DisplayName = "Gemma 4 26B A4B Instruct (MoE)",
            DefaultFile = "gemma-4-26B-A4B-it-Q4_0.gguf",
            ChatFormat = "gemma4",
            ContextLength = 262144,
            ParameterCount = 26_000_000_000,
            EstimatedSizeBytes = 14_618_145_824L,
            QuantizationType = "Q4_0",
            NumLayers = 46,
            HiddenSize = 4096,
            License = LicenseTier.MIT,
            LicenseName = "Apache 2.0",
            KnownIssues = [GgufModelKnownIssues.ToolUseUnreliableQ4, GgufModelKnownIssues.InstructionFollowingUnreliableQ4],
        },

        // Large: Gemma 4 31B Dense — maximum quality single-GPU model
        // NOTE (2026-08-17): this repo also never published a K-quant — see gemma4-default's note
        // above for the shared root cause (ISSUE-lm-supply-20260817-gemma4-ctx-other-...).
        ["gguf:gemma4-large"] = new GgufModelInfo
        {
            RepoId = "ggml-org/gemma-4-31B-it-GGUF",
            DisplayName = "Gemma 4 31B Instruct",
            DefaultFile = "gemma-4-31B-it-Q4_0.gguf",
            ChatFormat = "gemma4",
            ContextLength = 262144,
            ParameterCount = 31_000_000_000,
            EstimatedSizeBytes = 17_992_313_088L,
            QuantizationType = "Q4_0",
            NumLayers = 62,
            HiddenSize = 5376,
            License = LicenseTier.MIT,
            LicenseName = "Apache 2.0",
            KnownIssues = [GgufModelKnownIssues.ToolUseUnreliableQ4, GgufModelKnownIssues.InstructionFollowingUnreliableQ4],
        },

        // XLarge: Server-grade MoE model (split GGUF — 3 shards in Q4_K_M/ subfolder)
        ["gguf:xlarge"] = new GgufModelInfo
        {
            RepoId = "unsloth/Qwen3.5-122B-A10B-GGUF",
            DisplayName = "Qwen 3.5 122B A10B (MoE)",
            DefaultFile = "Q4_K_M/Qwen3.5-122B-A10B-Q4_K_M-00001-of-00003.gguf",
            ChatFormat = "chatml",
            ContextLength = 32768,
            ParameterCount = 122_000_000_000,
            EstimatedSizeBytes = 76_536_573_608L,
            QuantizationType = "Q4_K_M",
            NumLayers = 48,
            HiddenSize = 6144,
            ShardCount = 3,
            License = LicenseTier.MIT,
            LicenseName = "Apache 2.0",
        },

        // ============================================================
        // Phi-4 Mini — MIT, strong multilingual (KO/EN), Phi3 chat format, 16K context
        // Preferred for Korean RAG workloads where Gemma 4 E2B shows silent floor at Q4_K_M.
        // GGUF repack via bartowski; matches the ONNX variant at microsoft/Phi-4-mini-instruct-onnx.
        // ============================================================

        // Phi-4-mini Q4_K_M: 3.8B params, ~2.4GB VRAM, best Korean per-GB below 5GB
        // NOTE (2026-08-17, one-time registry audit): RepoId/DefaultFile previously pointed at
        // "bartowski/Phi-4-mini-instruct-GGUF", which does not exist — bartowski's repo (and every
        // file inside it) carries the "microsoft_" prefix from the upstream model id
        // (microsoft/Phi-4-mini-instruct). Any download attempt against the old id would fail
        // outright (repo not found) rather than silently misselect, unlike the Gemma4 class of defect.
        // Verified via huggingface.co/api/models/bartowski/microsoft_Phi-4-mini-instruct-GGUF.
        ["gguf:phi-4-mini"] = new GgufModelInfo
        {
            RepoId = "bartowski/microsoft_Phi-4-mini-instruct-GGUF",
            DisplayName = "Phi-4 Mini Instruct",
            DefaultFile = "microsoft_Phi-4-mini-instruct-Q4_K_M.gguf",
            ChatFormat = "phi3",
            ContextLength = 16384,
            ParameterCount = 3_800_000_000,
            EstimatedSizeBytes = 2_393_600_000L,
            QuantizationType = "Q4_K_M",
            NumLayers = 32,
            HiddenSize = 3072,
            License = LicenseTier.MIT,
            LicenseName = "MIT",
        },

        // ============================================================
        // Qwen 2.5 — Apache 2.0, strong multilingual (KO/ZH/EN), ChatML, 32K context
        // Recommended default for Korean RAG workloads where Gemma 4 family shows silent floor
        // ============================================================

        // Qwen2.5-7B: Best KO/ZH multilingual quality per GB; 32K context; ~4.4GB VRAM
        ["gguf:qwen2.5-7b"] = new GgufModelInfo
        {
            RepoId = "bartowski/Qwen2.5-7B-Instruct-GGUF",
            DisplayName = "Qwen 2.5 7B Instruct",
            DefaultFile = "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
            ChatFormat = "chatml",
            ContextLength = 32768,
            ParameterCount = 7_620_000_000,
            EstimatedSizeBytes = 4_682_024_960L,
            QuantizationType = "Q4_K_M",
            NumLayers = 28,
            HiddenSize = 3584,
            License = LicenseTier.MIT,
            LicenseName = "Apache 2.0",
        },

        // ============================================================
        // Qwen3/3.5/3.6 티어 — Apache 2.0, ChatML, GQA/MoE 혼합
        // auto-selection pool: qwen3-fast/default/balanced/quality (qwen3-large 제외)
        // ============================================================

        // Fast: Qwen3.5-2B Q4_K_M — ~1.5GB model, ~2.25GB VRAM@4K, thinking OFF by default
        ["gguf:qwen3-fast"] = new GgufModelInfo
        {
            RepoId = "unsloth/Qwen3.5-2B-GGUF",
            DisplayName = "Qwen 3.5 2B Instruct",
            DefaultFile = "Qwen3.5-2B-Q4_K_M.gguf",
            ChatFormat = "chatml",
            ContextLength = 262144,
            ParameterCount = 2_000_000_000,
            EstimatedSizeBytes = 1_500_000_000L,
            QuantizationType = "Q4_K_M",
            NumLayers = 24,
            HiddenSize = 2048,
            License = LicenseTier.MIT,
            LicenseName = "Apache 2.0",
        },

        // Default: Qwen3.5-4B Q4_K_M — ~3.0GB model, ~4.25GB VRAM@4K; thinking ON by default
        // Use FilterReasoningTokens=true if <think> blocks should not appear in output.
        ["gguf:qwen3-default"] = new GgufModelInfo
        {
            RepoId = "bartowski/Qwen_Qwen3.5-4B-GGUF",
            DisplayName = "Qwen 3.5 4B Instruct",
            DefaultFile = "Qwen_Qwen3.5-4B-Q4_K_M.gguf",
            ChatFormat = "chatml",
            ContextLength = 262144,
            ParameterCount = 4_000_000_000,
            EstimatedSizeBytes = 3_000_000_000L,
            QuantizationType = "Q4_K_M",
            NumLayers = 32,
            HiddenSize = 2560,
            License = LicenseTier.MIT,
            LicenseName = "Apache 2.0",
            KnownIssues = [GgufModelKnownIssues.ThinkingEnabledByDefault],
        },

        // Balanced: Qwen3-8B Q4_K_M — ~5.0GB model, ~7.25GB VRAM@4K; thinking opt-in via /think token
        ["gguf:qwen3-balanced"] = new GgufModelInfo
        {
            RepoId = "Qwen/Qwen3-8B-GGUF",
            DisplayName = "Qwen3 8B Instruct",
            DefaultFile = "Qwen3-8B-Q4_K_M.gguf",
            ChatFormat = "chatml",
            ContextLength = 131072,
            ParameterCount = 8_000_000_000,
            EstimatedSizeBytes = 5_000_000_000L,
            QuantizationType = "Q4_K_M",
            NumLayers = 36,
            HiddenSize = 4096,
            License = LicenseTier.MIT,
            LicenseName = "Apache 2.0",
        },

        // Quality: Qwen3.6-35B-A3B IQ4_XS — MoE 35B total / 3B active; ~17.7GB model, ~19GB VRAM@4K
        // Thinking ON by default; use FilterReasoningTokens=true to suppress <think> blocks.
        ["gguf:qwen3-quality"] = new GgufModelInfo
        {
            RepoId = "unsloth/Qwen3.6-35B-A3B-GGUF",
            DisplayName = "Qwen 3.6 35B A3B Instruct (IQ4_XS)",
            DefaultFile = "Qwen3.6-35B-A3B-UD-IQ4_XS.gguf",
            ChatFormat = "chatml",
            ContextLength = 131072,
            ParameterCount = 35_000_000_000,
            EstimatedSizeBytes = 17_700_000_000L,
            QuantizationType = "IQ4_XS",
            NumLayers = 40,
            HiddenSize = 2048,
            License = LicenseTier.MIT,
            LicenseName = "Apache 2.0",
            KnownIssues = [GgufModelKnownIssues.ThinkingEnabledByDefault],
        },

        // Large: Qwen3.6-35B-A3B Q4_K_M — ~22.1GB model, ~23.35GB VRAM@4K
        // Excluded from auto pool: exceeds 24GB × 85% = 20.4 GB; use explicit alias only.
        ["gguf:qwen3-large"] = new GgufModelInfo
        {
            RepoId = "unsloth/Qwen3.6-35B-A3B-GGUF",
            DisplayName = "Qwen 3.6 35B A3B Instruct (Q4_K_M)",
            DefaultFile = "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
            ChatFormat = "chatml",
            ContextLength = 131072,
            ParameterCount = 35_000_000_000,
            EstimatedSizeBytes = 22_100_000_000L,
            QuantizationType = "Q4_K_M",
            NumLayers = 40,
            HiddenSize = 2048,
            License = LicenseTier.MIT,
            LicenseName = "Apache 2.0",
            KnownIssues = [GgufModelKnownIssues.ThinkingEnabledByDefault],
        },
    };

    private static readonly HashSet<string> _autoSelectionAliases =
        new(StringComparer.OrdinalIgnoreCase)
        {
            "gguf:qwen3-fast",
            "gguf:qwen3-default",
            "gguf:qwen3-balanced",
            "gguf:qwen3-quality",
            // qwen3-large excluded: 23.35 GB exceeds 24GB × 85% = 20.4 GB budget
        };

    /// <summary>
    /// Default context length used to estimate KV cache size when computing the VRAM budget.
    /// llama-server reserves the full <c>--ctx-size</c> KV cache at load time, so a model
    /// that exceeds this budget at this context will OOM. 4096 is conservative and matches
    /// the practical default for most chat workloads.
    /// </summary>
    public const int DefaultBudgetContextLength = 4096;

    /// <summary>
    /// System RAM reserved for the OS and other processes when computing the CPU fit budget.
    /// Equals the <c>ramOverhead</c> reservation in <see cref="MemoryEstimator.EstimateForGguf"/> (4 GB), but
    /// selection is stricter than that estimate on purpose: <see cref="SystemRamBudgetBytes"/> also caps the
    /// budget at <see cref="SystemRamModelShare"/> of system RAM, because a default pick must leave the machine
    /// usable while the estimate only answers whether an explicitly chosen model can load.
    /// </summary>
    public const long SystemRamReservedBytes = 4L * 1024 * 1024 * 1024;

    /// <summary>
    /// The largest share of system RAM an automatically selected model may occupy when it runs from RAM
    /// (nothing fits VRAM). Unlike VRAM, system memory is shared with the OS and every other application,
    /// so a default must not claim most of it: on a 32 GB host this keeps the pick at an 8B-class model
    /// instead of a ~20 GB one. A caller that wants the larger model names it.
    /// </summary>
    public const double SystemRamModelShare = 0.5;

    /// <summary>
    /// The RAM budget for automatic selection: system RAM less <see cref="SystemRamReservedBytes"/>,
    /// capped at <see cref="SystemRamModelShare"/> of system RAM.
    /// </summary>
    public static long SystemRamBudgetBytes(long systemRamBytes)
    {
        if (systemRamBytes <= SystemRamReservedBytes)
            return 0L;
        return Math.Min(systemRamBytes - SystemRamReservedBytes, (long)(systemRamBytes * SystemRamModelShare));
    }

    /// <summary>
    /// Resolves an alias to model information.
    /// Supports both "gguf:alias" format and plain "alias" format.
    /// </summary>
    /// <param name="aliasOrRepoId">The alias (e.g., "gguf:gemma4-default", "gemma4-default", "gguf:auto") or full repo ID.</param>
    /// <returns>Model information if found, null otherwise. The <see cref="GgufModelInfo.AliasName"/> is populated for registered aliases.</returns>
    public static GgufModelInfo? Resolve(string aliasOrRepoId)
        => Resolve(aliasOrRepoId, ExecutionProvider.Auto);

    /// <summary>
    /// Resolves an alias for a load with <paramref name="provider"/>: <c>"gguf:auto"</c> is selected with
    /// <see cref="GetAutoSelection(ExecutionProvider)"/>, the same rule as <c>"default"</c>/<c>"auto"</c>.
    /// </summary>
    public static GgufModelInfo? Resolve(string aliasOrRepoId, ExecutionProvider provider)
    {
        if (string.IsNullOrWhiteSpace(aliasOrRepoId))
            return null;

        // Handle "gguf:auto" alias - select optimal model based on hardware
        if (aliasOrRepoId.Equals("gguf:auto", StringComparison.OrdinalIgnoreCase))
            return GetAutoSelection(provider).Selected;

        // Try direct lookup with gguf: prefix
        if (_models.TryGetValue(aliasOrRepoId, out var info))
            return WithAlias(info, aliasOrRepoId);

        // Try with gguf: prefix added
        if (!aliasOrRepoId.StartsWith("gguf:", StringComparison.OrdinalIgnoreCase))
        {
            var prefixed = $"gguf:{aliasOrRepoId}";
            if (_models.TryGetValue(prefixed, out info))
                return WithAlias(info, prefixed);
        }

        return null;
    }

    private static GgufModelInfo WithAlias(GgufModelInfo info, string alias)
        => info with { AliasName = alias.ToLowerInvariant() };

    /// <summary>
    /// Gets the optimal GGUF model based on current hardware profile.
    /// Delegates to the VRAM-aware overload using the current GPU.
    /// </summary>
    public static GgufModelInfo GetAutoModel()
        => GetAutoSelection(
            HardwareProfile.Current.GpuInfo,
            HardwareProfile.Current.SystemMemoryBytes,
            DefaultBudgetContextLength,
            excludeKnownIssues: null).Selected;

    /// <summary>
    /// Gets the optimal GGUF model based on actual available VRAM, including KV cache footprint.
    /// Sorts candidates by total VRAM footprint (weights + KV @ default context) descending,
    /// selects largest that fits, falls back to smallest if nothing fits.
    /// </summary>
    public static GgufModelInfo GetAutoModel(GpuInfo gpu)
        => GetAutoSelection(gpu).Selected;

    /// <summary>
    /// Gets the set of aliases eligible for auto-selection via <see cref="GetAutoSelection(GpuInfo)"/>.
    /// </summary>
    public static IReadOnlySet<string> GetAutoSelectionAliases() => _autoSelectionAliases;

    /// <summary>
    /// Performs auto-selection and returns the full diagnostic <see cref="ModelSelectionResult"/>
    /// including budget breakdown, candidate list with fit info, and the selection reason.
    /// Uses <see cref="DefaultBudgetContextLength"/> for KV cache estimation.
    /// </summary>
    public static ModelSelectionResult GetAutoSelection(GpuInfo gpu)
        => GetAutoSelection(gpu, DefaultBudgetContextLength);

    /// <summary>
    /// Performs auto-selection with an explicit budget context length for KV cache estimation.
    /// </summary>
    public static ModelSelectionResult GetAutoSelection(GpuInfo gpu, int budgetContextLength)
        => GetAutoSelection(gpu, budgetContextLength, null);

    /// <summary>
    /// Performs auto-selection, optionally excluding models with specific known issues.
    /// Models whose <see cref="GgufModelInfo.KnownIssues"/> intersects <paramref name="excludeKnownIssues"/>
    /// are removed before selection. Pass <c>null</c> to include all models (same as the two-param overload).
    /// </summary>
    public static ModelSelectionResult GetAutoSelection(
        GpuInfo gpu,
        int budgetContextLength,
        IReadOnlyCollection<string>? excludeKnownIssues)
        // GPU-only overload: no RAM information, so RAM fallback is disabled (systemRamBytes = 0).
        => GetAutoSelection(gpu, systemRamBytes: 0, budgetContextLength, excludeKnownIssues);

    /// <summary>
    /// Performs auto-selection considering both the VRAM budget and the system RAM budget.
    /// Prefers the largest model that fits VRAM (full GPU); if none fits VRAM, picks the largest
    /// that fits system RAM (CPU / partial offload) so a low-VRAM, high-RAM machine is not stuck
    /// with the smallest model; otherwise falls back to the smallest. Pass <paramref name="systemRamBytes"/>
    /// = 0 to disable the RAM path (pure VRAM selection).
    /// </summary>
    public static ModelSelectionResult GetAutoSelection(
        GpuInfo gpu,
        long systemRamBytes,
        int budgetContextLength,
        IReadOnlyCollection<string>? excludeKnownIssues)
    {
        var safetyMargin = VramBudget.GetRecommendedSafetyMargin(gpu);
        var availableVram = VramBudget.GetAvailableBytes(gpu, safetyMargin);
        return Select(availableVram, safetyMargin, systemRamBytes, budgetContextLength, excludeKnownIssues);
    }

    /// <summary>
    /// The selection behind <c>"default"</c>, <c>"auto"</c> and <c>"gguf:auto"</c> for a load with
    /// <paramref name="provider"/> — one rule for all three names. The profile is
    /// <see cref="HardwareProfile.For"/>: an explicit <see cref="ExecutionProvider.Cpu"/> selects from system
    /// memory alone (the VRAM budget, including <see cref="VramBudget.BudgetOverrideEnvVar"/>, does not apply to a
    /// load that ruled the GPU out) and never probes the GPU; any other provider selects from the detected GPU and
    /// falls back to system memory when no candidate fits VRAM.
    /// </summary>
    public static ModelSelectionResult GetAutoSelection(ExecutionProvider provider)
    {
        var profile = HardwareProfile.For(provider);
        return provider == ExecutionProvider.Cpu
            ? Select(availableVram: 0, safetyMargin: 0, profile.SystemMemoryBytes, DefaultBudgetContextLength, excludeKnownIssues: null)
            : GetAutoSelection(profile.GpuInfo, profile.SystemMemoryBytes, DefaultBudgetContextLength, excludeKnownIssues: null);
    }

    private static ModelSelectionResult Select(
        long availableVram,
        double safetyMargin,
        long systemRamBytes,
        int budgetContextLength,
        IReadOnlyCollection<string>? excludeKnownIssues)
    {
        var availableRam = SystemRamBudgetBytes(systemRamBytes);

        var poolFiltered = _models.Where(kv => _autoSelectionAliases.Contains(kv.Key));

        var eligible = excludeKnownIssues is { Count: > 0 }
            ? poolFiltered.Where(kv => !kv.Value.KnownIssues.Any(excludeKnownIssues.Contains))
            : poolFiltered;

        var candidates = eligible
            .Select(kv => EvaluateCandidate(WithAlias(kv.Value, kv.Key), availableVram, availableRam, budgetContextLength))
            .OrderByDescending(c => c.TotalBytes)
            .ToList();

        GgufModelInfo selected;
        ModelSelectionReason reason;

        var fittingVram = candidates.FirstOrDefault(c => c.Fits);
        var fittingRam = candidates.FirstOrDefault(c => c.FitsInSystemRam);

        if (fittingVram is not null)
        {
            selected = fittingVram.Model;
            reason = ModelSelectionReason.Fits;
        }
        else if (fittingRam is not null)
        {
            // Nothing fits VRAM, but RAM can hold it — run on CPU/partial offload instead of
            // dropping to the smallest model on a machine with ample RAM.
            selected = fittingRam.Model;
            reason = ModelSelectionReason.FitsInSystemRam;
        }
        else
        {
            var smallest = candidates.LastOrDefault()?.Model
                ?? WithAlias(_models["gguf:qwen3-fast"], "gguf:qwen3-fast");
            selected = smallest;
            reason = ModelSelectionReason.FallbackToSmallest;
        }

        return new ModelSelectionResult
        {
            Selected = selected,
            Reason = reason,
            AvailableVramBytes = availableVram,
            AvailableSystemRamBytes = availableRam,
            SafetyMargin = safetyMargin,
            BudgetContextLength = budgetContextLength,
            Candidates = candidates,
        };
    }

    private static ModelSelectionCandidate EvaluateCandidate(
        GgufModelInfo model, long availableVram, long availableRam, int budgetContextLength)
    {
        var weights = ModelMemoryEstimator.EstimateModelSizeBytes(
            model.ParameterCount,
            model.QuantizationType,
            model.EstimatedSizeBytes);

        // KV cache only computable when architecture fields are set.
        long kvCache = (model.NumLayers > 0 && model.HiddenSize > 0)
            ? ModelMemoryEstimator.EstimateKvCacheBytes(
                budgetContextLength, model.NumLayers, model.HiddenSize)
            : 0L;

        var total = weights + kvCache;
        return new ModelSelectionCandidate
        {
            Model = model,
            WeightsBytes = weights,
            KvCacheBytes = kvCache,
            Fits = total <= availableVram,
            FitsInSystemRam = availableRam > 0 && total <= availableRam,
        };
    }

    /// <summary>
    /// Gets all registered GGUF models.
    /// </summary>
    public static IReadOnlyList<GgufModelInfo> GetAllModels() =>
        _models.Values.ToList();

    /// <summary>
    /// Gets GGUF models filtered by license tier.
    /// </summary>
    public static IReadOnlyList<GgufModelInfo> GetModelsByLicense(LicenseTier tier) =>
        _models.Values.Where(m => m.License == tier).ToList();

    /// <summary>
    /// Checks if a string is a known GGUF alias.
    /// Only matches "gguf:"-prefixed aliases (e.g., "gguf:gemma4-default", "gguf:gemma4-fast", "gguf:auto").
    /// Plain aliases like "default" or "fast" are reserved for ONNX models.
    /// </summary>
    public static bool IsAlias(string value)
    {
        if (string.IsNullOrWhiteSpace(value))
            return false;

        // Only match if it starts with "gguf:" prefix
        // Plain aliases without prefix are reserved for ONNX ModelRegistry
        if (!value.StartsWith("gguf:", StringComparison.OrdinalIgnoreCase))
            return false;

        // Handle "gguf:auto" special alias
        if (value.Equals("gguf:auto", StringComparison.OrdinalIgnoreCase))
            return true;

        return _models.ContainsKey(value);
    }

    /// <summary>
    /// Gets all available alias names including "gguf:auto".
    /// </summary>
    public static IReadOnlyList<string> GetAliases()
    {
        var aliases = new List<string> { "gguf:auto" };
        aliases.AddRange(_models.Keys);
        return aliases;
    }
}
