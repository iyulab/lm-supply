using LMSupply.Generator.Abstractions;
using LMSupply.Hardware;
using LMSupply.Runtime;

namespace LMSupply.Generator;

/// <summary>
/// Factory class for creating local text generation models.
/// </summary>
public static class LocalGenerator
{
    /// <summary>
    /// Gets the model registry for the Generator domain.
    /// Provides access to model resolution, alias management, and model enumeration.
    /// </summary>
    public static IModelRegistry<ModelInfo> Registry => GeneratorModelRegistry.Default;

    /// <summary>
    /// Shared pool for named model management. Supports GetOrLoadAsync / UnloadAsync by model ID.
    /// </summary>
    public static GeneratorPool Pool { get; } = new(Internal.DefaultGeneratorFactory.Instance);

    /// <summary>
    /// Loads a text generator from a HuggingFace model repository or alias.
    /// </summary>
    /// <param name="modelId">
    /// One of:
    /// <list type="bullet">
    ///   <item><c>"default"</c> / <c>"auto"</c> — hardware-aware selection (see remarks).</item>
    ///   <item>A GGUF alias (<c>"gguf:gemma4-default"</c>, <c>"gguf:gemma4-fast"</c>, <c>"gguf:gemma4-quality"</c>, <c>"gguf:auto"</c>, etc.).</item>
    ///   <item>An ONNX alias (<c>"phi-4-mini"</c>, <c>"fast"</c>, <c>"quality"</c>, <c>"phi-3.5-mini"</c>).</item>
    ///   <item>A HuggingFace repo ID (e.g., <c>"microsoft/Phi-4-mini-instruct-onnx"</c>).</item>
    ///   <item>A local file path (GGUF) or directory (ONNX).</item>
    /// </list>
    /// </param>
    /// <param name="options">Model loading options.</param>
    /// <param name="progress">Progress callback for model downloading.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>A text generator instance.</returns>
    /// <remarks>
    /// For <c>"default"</c> and <c>"auto"</c>, the backend and model are selected from the host:
    /// <list type="bullet">
    ///   <item>NVIDIA GPU / CPU / macOS / Linux → GGUF via llama.cpp (Gemma 4 by default, VRAM-aware).</item>
    ///   <item>Windows with DirectML and a non-NVIDIA GPU → ONNX (Phi-4 Mini).</item>
    /// </list>
    /// The selection is logged via <c>Trace.TraceInformation</c> with a <c>[LocalGenerator.auto]</c> prefix.
    /// </remarks>
    public static Task<IGeneratorModel> LoadAsync(
        string modelId,
        GeneratorOptions? options = null,
        IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(modelId);
        options ??= new GeneratorOptions();

        // User alias translation precedes ALL format detection: an alias is a name
        // substitution, so the gguf/default/path checks below must see the TARGET
        // (e.g. "my-writer" -> "gguf:qwen3-quality" must enter the GGUF path).
        if (GeneratorModelRegistry.Default.TryGetUserAliasTarget(modelId, out var userAliasTarget))
        {
            modelId = userAliasTarget!;
        }

        // Any "gguf:"-prefixed string is a GGUF domain identifier — never split on ':'.
        // IsAlias only matches registered aliases, so unregistered "gguf:phi-4-mini" would
        // fall through to SplitQualifier, producing ("gguf", "phi-4-mini") and then calling
        // GgufModelDownloader with repoId="gguf", triggering a HF 401.
        if (modelId.StartsWith("gguf:", StringComparison.OrdinalIgnoreCase) ||
            Internal.Llama.GgufModelRegistry.IsAlias(modelId))
        {
            return Internal.GeneratorModelLoader.LoadAsync(modelId, options, progress, cancellationToken);
        }

        // Parse variant qualifier (e.g., "default:fp16" → modelId="default", hint="fp16")
        var (baseId, qualifier) = LMSupplyOptionsBase.SplitQualifier(modelId);
        modelId = baseId;
        options.QuantizationHint ??= qualifier;

        // "default" and "auto" both delegate to hardware-aware selection.
        if (modelId.Equals("default", StringComparison.OrdinalIgnoreCase) ||
            modelId.Equals("auto", StringComparison.OrdinalIgnoreCase))
        {
            return LoadAutoAsync(options, progress, cancellationToken);
        }

        // Handle other standard aliases via the registry
        if (GeneratorModelRegistry.Default.TryResolve(modelId, out var resolvedModel))
        {
            modelId = resolvedModel!.ModelId;
        }

        // Check if it's a local file path (e.g., C:\models\model.gguf or /path/to/model.gguf)
        if (File.Exists(modelId))
        {
            return Internal.GeneratorModelLoader.LoadFromPathAsync(modelId, options, modelId);
        }

        // Check if it's a local directory path
        if (Directory.Exists(modelId))
        {
            return Internal.GeneratorModelLoader.LoadFromPathAsync(modelId, options, modelId);
        }

        return Internal.GeneratorModelLoader.LoadAsync(modelId, options, progress, cancellationToken);
    }

    /// <summary>
    /// Downloads the weights <see cref="LoadAsync"/> would load for <paramref name="modelId"/> and
    /// returns their local path — without loading the model, provisioning a runtime or starting
    /// llama-server. For installers, first-run screens and CI cache warming: a later
    /// <see cref="LoadAsync"/> with the same id and options then downloads nothing.
    /// </summary>
    /// <param name="modelId">
    /// Anything <see cref="LoadAsync"/> accepts: a user alias, a <c>gguf:</c> alias, a registry alias,
    /// <c>"default"</c>/<c>"auto"</c> (resolved with the same hardware-aware selection), a HuggingFace
    /// repo id, or a local path (returned as-is).
    /// </param>
    /// <param name="options">
    /// The options the later load will use. <see cref="LMSupplyOptionsBase.Provider"/> matters: GGUF
    /// auto-quantization picks the file from the provider's memory budget, so passing a different
    /// provider here than at load time can warm the wrong file.
    /// </param>
    /// <param name="progress">Download progress.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>The local file (GGUF) or directory (ONNX) the model will be loaded from.</returns>
    public static async Task<string> DownloadModelAsync(
        string modelId,
        GeneratorOptions? options = null,
        IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(modelId);
        options ??= new GeneratorOptions();

        // Mirrors LoadAsync's resolution step for step — see the comments there for why each
        // check sits where it does. Divergence here would warm a file the load never opens.
        if (GeneratorModelRegistry.Default.TryGetUserAliasTarget(modelId, out var userAliasTarget))
        {
            modelId = userAliasTarget!;
        }

        if (modelId.StartsWith("gguf:", StringComparison.OrdinalIgnoreCase) ||
            Internal.Llama.GgufModelRegistry.IsAlias(modelId))
        {
            return await Internal.GeneratorModelLoader.DownloadAsync(modelId, options, progress, cancellationToken).ConfigureAwait(false);
        }

        var (baseId, qualifier) = LMSupplyOptionsBase.SplitQualifier(modelId);
        modelId = baseId;
        options.QuantizationHint ??= qualifier;

        if (modelId.Equals("default", StringComparison.OrdinalIgnoreCase) ||
            modelId.Equals("auto", StringComparison.OrdinalIgnoreCase))
        {
            modelId = !string.IsNullOrEmpty(options.PreferredAutoModelId)
                ? options.PreferredAutoModelId
                : SelectAutoModel(options).ModelId;
            return await DownloadModelAsync(modelId, options, progress, cancellationToken).ConfigureAwait(false);
        }

        if (GeneratorModelRegistry.Default.TryResolve(modelId, out var resolvedModel))
        {
            modelId = resolvedModel!.ModelId;
        }

        if (File.Exists(modelId) || Directory.Exists(modelId))
        {
            return modelId;
        }

        return await Internal.GeneratorModelLoader.DownloadAsync(modelId, options, progress, cancellationToken).ConfigureAwait(false);
    }

    /// <summary>
    /// Loads a text generator from a local model path.
    /// </summary>
    /// <param name="modelPath">The path to the local model directory or GGUF file.</param>
    /// <param name="options">Model loading options.</param>
    /// <returns>A text generator instance.</returns>
    public static Task<IGeneratorModel> LoadFromPathAsync(
        string modelPath,
        GeneratorOptions? options = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(modelPath);

        // Support both directory (ONNX) and file (GGUF) paths
        if (!Directory.Exists(modelPath) && !File.Exists(modelPath))
        {
            throw new FileNotFoundException($"Model path not found: {modelPath}");
        }

        options ??= new GeneratorOptions();

        return Internal.GeneratorModelLoader.LoadFromPathAsync(modelPath, options);
    }

    /// <summary>
    /// Tries each model ID in <paramref name="candidates"/> in order and returns
    /// the first one that loads successfully.
    /// </summary>
    /// <param name="candidates">
    /// Ordered list of model IDs to try (aliases, repo IDs, or paths).
    /// Must contain at least one entry. Use <c>"auto"</c> as a final fallback to
    /// trigger hardware-aware selection.
    /// </param>
    /// <param name="options">Model loading options shared across all candidates.</param>
    /// <param name="onFailure">
    /// Optional callback invoked when a candidate fails.
    /// Receives the failed model ID and the exception. Useful for logging.
    /// </param>
    /// <param name="progress">Progress callback for model downloading.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>The loaded model from the first successful candidate.</returns>
    /// <exception cref="AggregateException">
    /// Thrown when all candidates fail. Inner exceptions contain per-candidate failures.
    /// </exception>
    public static async Task<IGeneratorModel> LoadWithFallbackChainAsync(
        IReadOnlyList<string> candidates,
        GeneratorOptions? options = null,
        Action<string, Exception>? onFailure = null,
        IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        if (candidates is null || candidates.Count == 0)
            throw new ArgumentException("At least one candidate model ID is required.", nameof(candidates));

        var failures = new List<Exception>(candidates.Count);

        foreach (var modelId in candidates)
        {
            try
            {
                return await LoadAsync(modelId, options, progress, cancellationToken).ConfigureAwait(false);
            }
            catch (Exception ex) when (ex is not OperationCanceledException)
            {
                failures.Add(ex);
                onFailure?.Invoke(modelId, ex);
            }
        }

        throw new AggregateException(
            $"All {candidates.Count} candidate model(s) failed to load: {string.Join(", ", candidates)}",
            failures);
    }

    /// <summary>
    /// Loads a text generator using the default model.
    /// </summary>
    /// <param name="options">Model loading options.</param>
    /// <param name="progress">Progress callback for model downloading.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>A text generator instance.</returns>
    public static Task<IGeneratorModel> LoadDefaultAsync(
        GeneratorOptions? options = null,
        IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        return LoadAsync("default", options, progress, cancellationToken);
    }

    /// <summary>
    /// Auto-selects the optimal model based on hardware platform.
    /// GGUF for most environments (CPU, CUDA, Metal).
    /// ONNX only for Windows DirectML (non-NVIDIA) or NPU.
    /// </summary>
    private static async Task<IGeneratorModel> LoadAutoAsync(
        GeneratorOptions options,
        IProgress<DownloadProgress>? progress,
        CancellationToken cancellationToken)
    {
        // Honour caller-specified quality floor: try the preferred model before hardware selection.
        if (!string.IsNullOrEmpty(options.PreferredAutoModelId))
        {
            try
            {
                var preferred = await Internal.GeneratorModelLoader.LoadAsync(
                    options.PreferredAutoModelId, options, progress, cancellationToken)
                    .ConfigureAwait(false);

                System.Diagnostics.Trace.TraceInformation(
                    $"[LocalGenerator.auto] Preferred model loaded: {options.PreferredAutoModelId}");
                return preferred;
            }
            catch (Exception ex) when (ex is not OperationCanceledException)
            {
                System.Diagnostics.Trace.TraceWarning(
                    $"[LocalGenerator.auto] Preferred model '{options.PreferredAutoModelId}' failed " +
                    $"({ex.GetType().Name}: {ex.Message}); falling back to hardware-aware selection.");
            }
        }

        var (selectedModelId, diagnostics) = SelectAutoModel(options);

        var loaded = await Internal.GeneratorModelLoader.LoadAsync(
            selectedModelId, options, progress, cancellationToken).ConfigureAwait(false);

        if (loaded is Internal.IDiagnosticsSink sink)
            sink.SetDiagnostics(diagnostics);

        return loaded;
    }

    private static SelectionDiagnostics BuildOnnxDiagnostics(HardwareProfile profile)
        => new()
        {
            TotalVramBytes = profile.GpuInfo.TotalMemoryBytes,
            FreeVramBytes = profile.GpuInfo.FreeMemoryBytes,
            BudgetVramBytes = VramBudget.GetAvailableBytes(profile.GpuInfo),
            SafetyMargin = VramBudget.GetRecommendedSafetyMargin(profile.GpuInfo),
            EnvOverrideApplied = VramBudget.TryGetEnvOverrideBytes(out _)
        };

    private static SelectionDiagnostics BuildGgufDiagnostics(
        HardwareProfile profile, Internal.Llama.ModelSelectionResult selection)
        => new()
        {
            TotalVramBytes = profile.GpuInfo.TotalMemoryBytes,
            FreeVramBytes = profile.GpuInfo.FreeMemoryBytes,
            BudgetVramBytes = selection.AvailableVramBytes,
            SafetyMargin = selection.SafetyMargin,
            SelectionReason = selection.Reason.ToString(),
            EnvOverrideApplied = VramBudget.TryGetEnvOverrideBytes(out _)
        };

    /// <summary>
    /// Hardware-aware model selection behind <c>"default"</c>/<c>"auto"</c>: the backend decision
    /// (ONNX vs GGUF) and the model within it, with the diagnostics a loaded model reports. Shared by
    /// <see cref="LoadAutoAsync"/> and <see cref="DownloadModelAsync"/> so warming and loading agree.
    /// Logs the selection with the <c>[LocalGenerator.auto]</c> prefix.
    /// </summary>
    private static (string ModelId, SelectionDiagnostics Diagnostics) SelectAutoModel(GeneratorOptions options)
    {
        var profile = HardwareProfile.Current;
        var useOnnx = Internal.GeneratorRoutingPolicy.ShouldUseOnnx(
            profile.GpuInfo, profile.RecommendedProvider);

        if (useOnnx)
        {
            var model = GeneratorModelRegistry.Default.Resolve("auto");
            LogOnnxAutoSelection(profile, model.ModelId);
            return (model.ModelId, BuildOnnxDiagnostics(profile));
        }

        var selection = Internal.Llama.GgufModelRegistry.GetAutoSelection(profile.GpuInfo);
        // Pass the alias (e.g. "gguf:gemma4-fast") rather than RepoId so the downstream
        // loader can re-resolve the registry entry and use its DefaultFile. Passing
        // RepoId would lose the DefaultFile and fall back to GgufFileSelector, which
        // can pick larger variants (e.g. bf16) that fit in VRAM+RAM but blow VRAM.
        var selectedModelId = !string.IsNullOrEmpty(selection.Selected.AliasName)
            ? selection.Selected.AliasName
            : selection.Selected.RepoId;
        LogGgufAutoSelection(profile, selection);
        return (selectedModelId, BuildGgufDiagnostics(profile, selection));
    }

    private static void LogOnnxAutoSelection(HardwareProfile profile, string modelId)
    {
        var vramMb = VramBudget.GetAvailableBytes(profile.GpuInfo) / (1024 * 1024);
        System.Diagnostics.Trace.TraceInformation(
            $"[LocalGenerator.auto] Provider={profile.RecommendedProvider}, " +
            $"GPU={profile.GpuInfo.Vendor} {profile.GpuInfo.DeviceName ?? "n/a"}, " +
            $"VRAM={vramMb}MB → ONNX path, selected={modelId}");
    }

    private static void LogGgufAutoSelection(
        HardwareProfile profile, Internal.Llama.ModelSelectionResult selection)
    {
        const double mb = 1024.0 * 1024.0;
        var vramTotalMb = (profile.GpuInfo.TotalMemoryBytes ?? 0) / mb;
        var vramFreeMb = (profile.GpuInfo.FreeMemoryBytes ?? profile.GpuInfo.TotalMemoryBytes ?? 0) / mb;
        var budgetMb = selection.AvailableVramBytes / mb;
        var marginPct = selection.SafetyMargin * 100;

        System.Diagnostics.Trace.TraceInformation(
            $"[LocalGenerator.auto] Provider={profile.RecommendedProvider}, " +
            $"GPU={profile.GpuInfo.Vendor} {profile.GpuInfo.DeviceName ?? "n/a"}, " +
            $"VRAM total={vramTotalMb:F0}MB free={vramFreeMb:F0}MB → budget={budgetMb:F0}MB (margin={marginPct:F0}%, " +
            $"KV ctx={selection.BudgetContextLength}) → GGUF path, " +
            $"selected={selection.Selected.AliasName} ({selection.Selected.RepoId}), " +
            $"reason={selection.Reason}");

        // Warn when free VRAM is the binding constraint (total-based cap would have been larger).
        if (profile.GpuInfo.TotalMemoryBytes is > 0 && profile.GpuInfo.FreeMemoryBytes is > 0)
        {
            var totalCapMb = profile.GpuInfo.TotalMemoryBytes.Value * (1.0 - selection.SafetyMargin) / mb;
            var freeCapMb = profile.GpuInfo.FreeMemoryBytes.Value * Hardware.VramBudget.FreeVramSafetyFactor / mb;
            if (freeCapMb < totalCapMb * 0.99)
            {
                System.Diagnostics.Trace.TraceWarning(
                    $"[LocalGenerator.auto] Free VRAM cap binding: budget capped at {budgetMb:F0}MB " +
                    $"by free VRAM ({vramFreeMb:F0}MB × {Hardware.VramBudget.FreeVramSafetyFactor:P0}), " +
                    $"not by total × margin ({totalCapMb:F0}MB). " +
                    $"Another process may be occupying GPU memory. " +
                    $"Set {Hardware.VramBudget.BudgetOverrideEnvVar}=<MB> to override.");
            }
        }

        // Verbose breakdown of all candidates considered
        foreach (var c in selection.Candidates)
        {
            var weightsMb = c.WeightsBytes / mb;
            var kvMb = c.KvCacheBytes / mb;
            var totalMb = c.TotalBytes / mb;
            System.Diagnostics.Trace.TraceInformation(
                $"[LocalGenerator.auto]   candidate {c.Model.AliasName,-14} " +
                $"weights={weightsMb,7:F0}MB + KV={kvMb,6:F0}MB = {totalMb,7:F0}MB " +
                $"({(c.Fits ? "fits" : "OVER BUDGET")})");
        }

        if (selection.Reason == Internal.Llama.ModelSelectionReason.FallbackToSmallest)
        {
            System.Diagnostics.Trace.TraceWarning(
                $"[LocalGenerator.auto] WARNING: no registered model fits in {budgetMb:F0}MB budget. " +
                $"Selected smallest ({selection.Selected.AliasName}) as fallback — " +
                $"runtime may OOM or partial-offload to CPU. " +
                $"Override with explicit alias (e.g., \"phi-4-mini\") for low-VRAM hosts.");
        }
    }
}
