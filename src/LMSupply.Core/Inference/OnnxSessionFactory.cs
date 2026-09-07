using System.Diagnostics;
using System.Runtime.InteropServices;
using LMSupply.Download;
using LMSupply.Exceptions;
using LMSupply.Runtime;
using Microsoft.ML.OnnxRuntime;

namespace LMSupply.Inference;

/// <summary>
/// Result of session creation including metadata about the active execution provider.
/// </summary>
public sealed class SessionCreationResult
{
    /// <summary>
    /// The created inference session.
    /// </summary>
    public required InferenceSession Session { get; init; }

    /// <summary>
    /// The execution provider that was requested.
    /// </summary>
    public required ExecutionProvider RequestedProvider { get; init; }

    /// <summary>
    /// The execution providers actually active in the session.
    /// </summary>
    public required IReadOnlyList<string> ActiveProviders { get; init; }

    /// <summary>
    /// GPU providers the load-time chain tried for this model and that failed to produce a working
    /// session (initialization threw, or the provider library did not load). Empty when the first
    /// candidate succeeded. A provider that was merely skipped — missing CUDA libraries, or excluded
    /// by the caller — is not listed: it was never tried. <see cref="RecoverableOnnxSession"/> seeds
    /// its <see cref="ProviderBlacklist"/> from this so sibling sessions of the same model do not
    /// try a provider that already refused it.
    /// </summary>
    public IReadOnlyList<ExecutionProvider> FailedProviders { get; init; } = Array.Empty<ExecutionProvider>();

    /// <summary>
    /// Whether GPU acceleration is actually being used.
    /// </summary>
    public bool IsGpuActive => ActiveProviders.Any(p =>
        p.Contains("CUDA", StringComparison.OrdinalIgnoreCase) ||
        p.Contains("DML", StringComparison.OrdinalIgnoreCase) ||
        p.Contains("DirectML", StringComparison.OrdinalIgnoreCase) ||
        p.Contains("CoreML", StringComparison.OrdinalIgnoreCase) ||
        p.Contains("TensorRT", StringComparison.OrdinalIgnoreCase));
}

/// <summary>
/// Factory for creating ONNX Runtime inference sessions with proper execution provider configuration.
/// Supports lazy loading of native runtime binaries via RuntimeManager.
/// </summary>
public static class OnnxSessionFactory
{
    /// <summary>
    /// Creates an ONNX Runtime inference session asynchronously, ensuring runtime binaries are available.
    /// This is the recommended method as it downloads required binaries on first use.
    /// When provider is Auto, uses fallback chain: CUDA → DirectML → CoreML → CPU.
    /// </summary>
    /// <param name="modelPath">Path to the ONNX model file.</param>
    /// <param name="provider">The execution provider to use.</param>
    /// <param name="configureOptions">Optional callback to configure additional session options.</param>
    /// <param name="progress">Optional progress reporter for binary downloads.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>A configured inference session.</returns>
    public static async Task<InferenceSession> CreateAsync(
        string modelPath,
        ExecutionProvider provider = ExecutionProvider.Auto,
        Action<SessionOptions>? configureOptions = null,
        IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        // Use CreateWithInfoAsync which handles the fallback chain and return just the session
        var result = await CreateWithInfoAsync(modelPath, provider, configureOptions, progress, cancellationToken);
        return result.Session;
    }

    /// <summary>
    /// Creates an ONNX Runtime inference session with detailed information about active providers.
    /// Use this when you need to verify GPU acceleration is actually working.
    /// When provider is Auto, uses fallback chain: CUDA → DirectML → CoreML → CPU.
    /// </summary>
    /// <param name="modelPath">Path to the ONNX model file.</param>
    /// <param name="provider">The execution provider to use.</param>
    /// <param name="configureOptions">Optional callback to configure additional session options.</param>
    /// <param name="progress">Optional progress reporter for binary downloads.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>A session creation result with provider information.</returns>
    public static async Task<SessionCreationResult> CreateWithInfoAsync(
        string modelPath,
        ExecutionProvider provider = ExecutionProvider.Auto,
        Action<SessionOptions>? configureOptions = null,
        IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        return await CreateWithInfoAsync(modelPath, provider, skipProviders: null,
            configureOptions, progress, cancellationToken);
    }

    /// <summary>
    /// Same as <see cref="CreateWithInfoAsync(string, ExecutionProvider, Action{SessionOptions}?, IProgress{DownloadProgress}?, CancellationToken)"/>
    /// but accepts a set of providers to exclude from the Auto fallback chain.
    /// Used by inference-time fallback recovery when a previously-selected provider crashed at run time.
    /// </summary>
    public static async Task<SessionCreationResult> CreateWithInfoAsync(
        string modelPath,
        ExecutionProvider provider,
        IReadOnlyCollection<ExecutionProvider>? skipProviders,
        Action<SessionOptions>? configureOptions = null,
        IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        // Ensure runtime binaries are available
        await RuntimeManager.Instance.InitializeAsync(cancellationToken);

        if (provider == ExecutionProvider.Auto)
        {
            // Use fallback chain: CUDA → DirectML → CoreML → CPU
            return await CreateWithFallbackChainAsync(modelPath, skipProviders, configureOptions, progress, cancellationToken);
        }

        // Explicit provider specified
        var providerString = provider switch
        {
            ExecutionProvider.Cuda => "cuda12",  // Try CUDA 12 first
            ExecutionProvider.DirectML => "directml",
            ExecutionProvider.CoreML => "coreml",
            _ => "cpu"
        };
        var isGpuRequested = provider is ExecutionProvider.Cuda or ExecutionProvider.DirectML or ExecutionProvider.CoreML;

        // Download runtime binaries if needed. This is provisioning, not session construction --
        // a platform that cannot provision the requested provider at all (e.g. DirectML has no
        // native binaries for linux-x64/osx-*) fails loud here with an actionable message. The
        // CPU-fallback catch below is reserved for a narrower failure class: the runtime WAS
        // successfully provisioned but session construction itself failed transiently (a missing
        // DX12 device, a CUDA driver mismatch). Conflating the two would let a caller silently run
        // on CPU while believing an explicitly-requested GPU provider is active.
        try
        {
            await RuntimeManager.Instance.EnsureRuntimeAsync(
                "onnxruntime",
                provider: providerString,
                progress: progress,
                cancellationToken: cancellationToken);
        }
        catch (OperationCanceledException)
        {
            throw;
        }
        catch (Exception ex) when (isGpuRequested)
        {
            throw WrapProvisioningFailure(provider, modelPath, ex);
        }

        InferenceSession session;
        bool gpuEpAppended;
        try
        {
            // Create session on a dedicated thread to avoid thread pool starvation.
            // InferenceSession construction blocks for several seconds on large models.
            // Capture whether the requested GPU EP was actually appended (Try* may silently fail).
            var created = await Task.Factory.StartNew(
                () =>
                {
                    var s = Create(modelPath, provider, configureOptions, out var appended);
                    return (s, appended);
                },
                cancellationToken,
                TaskCreationOptions.LongRunning,
                TaskScheduler.Default);
            session = created.s;
            gpuEpAppended = created.appended;
        }
        catch (OperationCanceledException)
        {
            throw;
        }
        catch (Exception ex) when (isGpuRequested)
        {
            // GPU session initialization failed (e.g., DirectML DX12 device unavailable,
            // CUDA driver mismatch). Fall back to CPU so the caller gets a working session.
            var msg = ex.Message.Length > 100 ? ex.Message[..100] + "..." : ex.Message;
            Trace.TraceWarning($"[OnnxSessionFactory] {provider} session init failed: {msg}. Falling back to CPU.");

            await RuntimeManager.Instance.EnsureRuntimeAsync(
                "onnxruntime", provider: "cpu", progress: progress, cancellationToken: cancellationToken);

            session = await Task.Factory.StartNew(
                () => Create(modelPath, ExecutionProvider.Cpu, configureOptions),
                cancellationToken,
                TaskCreationOptions.LongRunning,
                TaskScheduler.Default);

            return new SessionCreationResult
            {
                Session = session,
                RequestedProvider = provider,
                ActiveProviders = new[] { "CPUExecutionProvider" },
                FailedProviders = new[] { provider }
            };
        }

        // Build the active-provider list from what was actually appended (not a heuristic re-probe).
        var activeProviders = ResolveActiveProviders(provider, gpuEpAppended);

        if (isGpuRequested && !activeProviders.Any(p => p != "CPUExecutionProvider"))
        {
            Trace.TraceWarning($"[OnnxSessionFactory] Warning: GPU provider {provider} was requested but only CPU is active. " +
                $"Active providers: {string.Join(", ", activeProviders)}");
        }

        return new SessionCreationResult
        {
            Session = session,
            RequestedProvider = provider,
            ActiveProviders = activeProviders
        };
    }

    /// <summary>
    /// Creates a session using the fallback chain until one succeeds.
    /// For CUDA, verifies runtime libraries are available before attempting.
    /// For DirectML, trusts the session creation result (Windows manages DirectML).
    /// </summary>
    /// <remarks>
    /// Does not call <see cref="EnsureOnnxRuntimeAvailable"/> itself — <see cref="Create(string,
    /// ExecutionProvider,Action{SessionOptions}?,out bool)"/> already does, and it must run after
    /// each provider's <c>EnsureRuntimeAsync</c> call below, not before this method's own fallback
    /// loop starts. RuntimeManager/NativeLoader registration is in-memory and per-process, so on a
    /// process that has never provisioned a runtime yet, checking availability before the first
    /// <c>EnsureRuntimeAsync</c> call always reports unavailable — a false negative that used to
    /// fail every first-time Auto-provider load outright, before it ever reached the download step.
    /// </remarks>
    private static async Task<SessionCreationResult> CreateWithFallbackChainAsync(
        string modelPath,
        IReadOnlyCollection<ExecutionProvider>? skipProviders,
        Action<SessionOptions>? configureOptions,
        IProgress<DownloadProgress>? progress,
        CancellationToken cancellationToken)
    {
        var fallbackChain = RuntimeManager.Instance.Gpu?.GetFallbackProviders()
            ?? new[] { ExecutionProvider.Cpu };

        Exception? lastException = null;
        var triedProviders = new List<string>();
        var failedProviders = new List<ExecutionProvider>();

        foreach (var providerToTry in fallbackChain)
        {
            if (skipProviders is not null && skipProviders.Contains(providerToTry))
            {
                Trace.TraceInformation($"[Fallback] {providerToTry}: skipped (caller-blacklisted)");
                triedProviders.Add($"{providerToTry}(blacklisted)");
                continue;
            }

            try
            {
                // For CUDA, check if runtime libraries are available FIRST
                if (providerToTry == ExecutionProvider.Cuda)
                {
                    var (cudaAvailable, missingLibs) = CheckCudaRuntimeAvailability();
                    if (!cudaAvailable)
                    {
                        // Warning, not Information: CUDA was the recommended provider on this hardware
                        // but a runtime dependency is missing. Operators need to see this in ASP.NET logs
                        // because the silent skip means inference will fall back to DML/CPU and may
                        // surface as a confusing "wrong provider was used" later.
                        Trace.TraceWarning(
                            $"[OnnxSessionFactory] CUDA skipped (missing: {string.Join(", ", missingLibs)}). " +
                            $"Falling back to next provider in chain. " +
                            $"Install the missing libraries to enable CUDA acceleration.");
                        triedProviders.Add("CUDA(skipped)");
                        continue;
                    }
                }

                // Get provider string for binary download
                var providerString = providerToTry switch
                {
                    ExecutionProvider.Cuda => "cuda12",  // Try CUDA 12 first
                    ExecutionProvider.DirectML => "directml",
                    ExecutionProvider.CoreML => "coreml",
                    _ => "cpu"
                };

                // Download runtime binaries
                await RuntimeManager.Instance.EnsureRuntimeAsync(
                    "onnxruntime",
                    provider: providerString,
                    progress: progress,
                    cancellationToken: cancellationToken);

                // Create session on a dedicated thread to avoid thread pool starvation.
                // InferenceSession construction blocks for several seconds on large models.
                var session = await Task.Factory.StartNew(
                    () => Create(modelPath, providerToTry, configureOptions),
                    cancellationToken,
                    TaskCreationOptions.LongRunning,
                    TaskScheduler.Default);

                // Determine active providers based on what was requested
                var activeProviders = new List<string>();

                if (providerToTry == ExecutionProvider.Cuda)
                {
                    // For CUDA, verify the provider DLL was actually loaded
                    if (IsCudaProviderLoaded())
                    {
                        activeProviders.Add("CUDAExecutionProvider");
                        Trace.TraceInformation($"[Fallback] CUDA: success");
                    }
                    else
                    {
                        Trace.TraceInformation($"[Fallback] CUDA: failed (provider not loaded), trying next...");
                        triedProviders.Add("CUDA(failed)");
                        failedProviders.Add(ExecutionProvider.Cuda);
                        session.Dispose();
                        continue;
                    }
                }
                else if (providerToTry == ExecutionProvider.DirectML)
                {
                    // DirectML is managed by Windows - trust the session creation
                    // If it fails, ONNX Runtime would have thrown or fallen back internally
                    activeProviders.Add("DmlExecutionProvider");
                    Trace.TraceInformation($"[Fallback] DirectML: success");
                }
                else if (providerToTry == ExecutionProvider.CoreML)
                {
                    activeProviders.Add("CoreMLExecutionProvider");
                    Trace.TraceInformation($"[Fallback] CoreML: success");
                }

                activeProviders.Add("CPUExecutionProvider");

                return new SessionCreationResult
                {
                    Session = session,
                    RequestedProvider = ExecutionProvider.Auto,
                    ActiveProviders = activeProviders,
                    FailedProviders = failedProviders
                };
            }
            catch (OperationCanceledException)
            {
                throw;
            }
            catch (Exception ex) when (providerToTry != ExecutionProvider.Cpu)
            {
                // Provide helpful error messages based on the error type
                if (ex.Message.Contains("CUDNN_STATUS_NOT_INITIALIZED"))
                {
                    var cudaEnv = Runtime.CudaEnvironment.Instance;
                    var cuda = cudaEnv.PrimaryCuda;
                    var cudaMajor = cuda?.MajorVersion ?? 12;
                    var cudnnMajor = cudaMajor >= 12 ? 9 : 8;

                    Trace.TraceInformation($"[Fallback] {providerToTry}: cuDNN initialization failed. Ensure cuDNN {cudnnMajor}.x bin is in PATH.");

                    // Show detected cuDNN installations or suggest installation
                    var cudnn = cudaEnv.GetBestCuDnn(cudaMajor);
                    if (cudnn is not null)
                    {
                        Trace.TraceInformation($"Found: cuDNN {cudnn.Version} at {cudnn.LibraryPath}");
                        if (cudnn.MajorVersion < 9)
                            Trace.TraceInformation($"Note: zlibwapi.dll may be required for cuDNN 8.x");
                    }
                    else
                    {
                        Trace.TraceInformation($"Install cuDNN {cudnnMajor}.x from https://developer.nvidia.com/cudnn");
                    }
                }
                else
                {
                    // Try to parse missing library names from CUDA errors
                    var missingLibs = ParseMissingLibrariesFromError(ex.Message);
                    if (missingLibs.Length > 0)
                    {
                        Trace.TraceInformation($"[Fallback] {providerToTry}: missing {string.Join(", ", missingLibs)}");
                    }
                    else
                    {
                        // Truncate long error messages
                        var msg = ex.Message.Length > 100 ? ex.Message[..100] + "..." : ex.Message;
                        Trace.TraceInformation($"[Fallback] {providerToTry}: {msg}");
                    }
                }
                triedProviders.Add($"{providerToTry}(error)");
                failedProviders.Add(providerToTry);
                lastException = ex;
            }
        }

        // Final fallback to CPU
        Trace.TraceInformation($"[Fallback] Using CPU (tried: {string.Join(" -> ", triedProviders)})");

        await RuntimeManager.Instance.EnsureRuntimeAsync(
            "onnxruntime", provider: "cpu", progress: progress, cancellationToken: cancellationToken);

        var cpuSession = Create(modelPath, ExecutionProvider.Cpu, configureOptions);
        return new SessionCreationResult
        {
            Session = cpuSession,
            RequestedProvider = ExecutionProvider.Auto,
            ActiveProviders = new[] { "CPUExecutionProvider" },
            FailedProviders = failedProviders
        };
    }

    /// <summary>
    /// Checks CUDA runtime availability using CudaEnvironment for dynamic detection.
    /// This works before ONNX Runtime is loaded.
    /// </summary>
    /// <returns>Tuple of (isAvailable, missingLibraries)</returns>
    private static (bool Available, string[] MissingLibraries) CheckCudaRuntimeAvailability()
    {
        if (!OperatingSystem.IsWindows() && !OperatingSystem.IsLinux())
            return (false, new[] { "CUDA not supported on this platform" });

        var cudaEnv = Runtime.CudaEnvironment.Instance;
        cudaEnv.Initialize();

        // Check for CUDA installation
        var cuda = cudaEnv.PrimaryCuda;
        if (cuda is null)
        {
            return (false, new[] { "CUDA_PATH not set or CUDA not installed" });
        }

        var cudaMajorVersion = cuda.MajorVersion;
        var missing = new List<string>();

        // Check for required CUDA libraries (cuBLAS)
        var (cudaAvailable, cudaMissing) = cudaEnv.CheckCudaLibraries(cudaMajorVersion);
        if (!cudaAvailable)
        {
            missing.AddRange(cudaMissing);
        }

        // Check for cuDNN (required for CUDA provider)
        // ONNX Runtime 1.22+ requires cuDNN 9.x for CUDA 12.x
        var requiredCudnnMajor = cudaMajorVersion >= 12 ? 9 : 8;
        var (cudnnAvailable, cudnnMissing, _) = cudaEnv.CheckCuDnnLibraries(cudaMajorVersion);

        if (!cudnnAvailable)
        {
            // Provide helpful error message with expected DLL name
            var expectedCudnnDll = Runtime.CudaEnvironment.GetCuDnnLibraryName(requiredCudnnMajor);
            if (cudnnMissing.Any(m => m.Contains("not found")))
            {
                missing.Add($"{expectedCudnnDll} (cuDNN {requiredCudnnMajor}.x required for CUDA {cudaMajorVersion}.x)");
            }
            else
            {
                missing.AddRange(cudnnMissing);
            }
        }

        if (missing.Count > 0)
            return (false, missing.ToArray());

        return (true, Array.Empty<string>());
    }

    /// <summary>
    /// Checks if the ONNX Runtime native library can be loaded.
    /// This catches missing VC++ Redistributable (Windows) or missing shared libraries (Linux)
    /// BEFORE SessionOptions construction, which would otherwise cause a fatal process crash (0xC0000005).
    /// </summary>
    /// <returns>Tuple of (isAvailable, errorMessage). errorMessage is null when available.</returns>
    public static (bool Available, string? ErrorMessage) CheckOnnxRuntimeAvailability()
    {
        // Fast path: NativeLoader already has it loaded
        if (NativeLoader.Instance.IsLoaded("onnxruntime"))
            return (true, null);

        // Try to load via NativeLoader (checks registered paths)
        if (NativeLoader.Instance.TryLoad("onnxruntime", out _))
            return (true, null);

        // Last resort: try direct load
        try
        {
            var libName = OperatingSystem.IsWindows() ? "onnxruntime"
                : OperatingSystem.IsLinux() ? "libonnxruntime.so"
                : OperatingSystem.IsMacOS() ? "libonnxruntime.dylib"
                : "onnxruntime";

            if (NativeLibrary.TryLoad(libName, out var handle))
            {
                NativeLibrary.Free(handle);
                return (true, null);
            }
        }
        catch
        {
            // Swallow - fall through to error message
        }

        var message = OperatingSystem.IsWindows()
            ? "ONNX Runtime native library failed to load. This usually means Visual C++ Redistributable 2015-2022 is not installed. "
              + "Download from: https://aka.ms/vs/17/release/vc_redist.x64.exe"
            : OperatingSystem.IsLinux()
                ? "ONNX Runtime native library failed to load. Ensure required shared libraries are installed (libstdc++, libgomp)."
                : "ONNX Runtime native library failed to load. Ensure required system libraries are installed.";

        return (false, message);
    }

    /// <summary>
    /// Throws ModelLoadException if ONNX Runtime native library is not available.
    /// </summary>
    private static void EnsureOnnxRuntimeAvailable(string? modelPath = null)
    {
        var (available, errorMessage) = CheckOnnxRuntimeAvailability();
        if (!available)
        {
            Trace.TraceError($"[OnnxSessionFactory] {errorMessage}");
            throw new ModelLoadException(
                errorMessage ?? "ONNX Runtime native library is not available.",
                modelPath);
        }
    }

    /// <summary>
    /// Wraps a provisioning-stage failure (the requested provider has no native binaries for this
    /// platform) into a clear, actionable exception, instead of letting the provisioning layer's
    /// raw message (e.g. "No native binaries found for {rid} in {package}") leak to the caller.
    /// </summary>
    internal static ModelLoadException WrapProvisioningFailure(
        ExecutionProvider provider, string modelPath, Exception inner) =>
        new(
            $"{provider} is not supported on this platform ({RuntimeInformation.RuntimeIdentifier}). " +
            "Use ExecutionProvider.Auto or ExecutionProvider.Cpu instead.",
            modelPath,
            inner);

    /// <summary>
    /// Parses ONNX Runtime error messages to extract missing library names.
    /// Example: '...depends on "cublasLt64_12.dll" which is missing...'
    /// </summary>
    private static string[] ParseMissingLibrariesFromError(string errorMessage)
    {
        var missing = new List<string>();

        // Pattern: depends on "xxx.dll" which is missing
        var regex = new System.Text.RegularExpressions.Regex(
            @"depends on ""([^""]+\.dll)"" which is missing",
            System.Text.RegularExpressions.RegexOptions.IgnoreCase);

        var matches = regex.Matches(errorMessage);
        foreach (System.Text.RegularExpressions.Match match in matches)
        {
            if (match.Groups.Count > 1)
                missing.Add(match.Groups[1].Value);
        }

        return missing.ToArray();
    }

    /// <summary>
    /// Checks if CUDA provider DLL was successfully loaded into the process.
    /// </summary>
    private static bool IsCudaProviderLoaded()
    {
        try
        {
            var modules = System.Diagnostics.Process.GetCurrentProcess().Modules;
            foreach (System.Diagnostics.ProcessModule module in modules)
            {
                if (module.ModuleName?.Contains("onnxruntime_providers_cuda", StringComparison.OrdinalIgnoreCase) == true)
                    return true;
            }
        }
        catch (Exception ex)
        {
            Trace.TraceInformation($"[Fallback] Failed to enumerate process modules: {ex.Message}");
        }
        return false;
    }

    /// <summary>
    /// Creates an ONNX Runtime inference session with the specified execution provider.
    /// Note: This assumes runtime binaries are already available. For lazy loading, use CreateAsync.
    /// </summary>
    /// <param name="modelPath">Path to the ONNX model file.</param>
    /// <param name="provider">The execution provider to use.</param>
    /// <param name="configureOptions">Optional callback to configure additional session options.</param>
    /// <returns>A configured inference session.</returns>
    public static InferenceSession Create(
        string modelPath,
        ExecutionProvider provider = ExecutionProvider.Auto,
        Action<SessionOptions>? configureOptions = null)
        => Create(modelPath, provider, configureOptions, out _);

    /// <summary>
    /// Same as <see cref="Create(string, ExecutionProvider, Action{SessionOptions}?)"/> but reports
    /// whether a GPU execution provider was actually appended to the session options.
    /// <paramref name="gpuEpAppended"/> is false for CPU/Auto-to-CPU and when a GPU EP append failed
    /// silently — letting callers report accurate active providers instead of a loadability heuristic.
    /// </summary>
    public static InferenceSession Create(
        string modelPath,
        ExecutionProvider provider,
        Action<SessionOptions>? configureOptions,
        out bool gpuEpAppended)
    {
        // Pre-check: prevent fatal crash from NativeMethods..cctor() on missing native dependencies
        EnsureOnnxRuntimeAvailable(modelPath);

        var options = new SessionOptions();

        // Default to Error level to suppress expected ORT warnings
        // (e.g., "Some nodes were not assigned to the preferred execution providers").
        // Callers can override via configureOptions callback.
        options.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR;

        // Apply user configuration (may override log level)
        configureOptions?.Invoke(options);

        // Configure execution provider
        gpuEpAppended = ConfigureExecutionProvider(options, provider);

        return new InferenceSession(modelPath, options);
    }

    /// <summary>
    /// Configures the execution provider for the session options.
    /// Returns true when a GPU execution provider was successfully appended.
    /// </summary>
    public static bool ConfigureExecutionProvider(SessionOptions options, ExecutionProvider provider)
    {
        return provider switch
        {
            ExecutionProvider.Auto => TryAddBestAvailableProvider(options),
            ExecutionProvider.Cuda => TryAddCuda(options),
            ExecutionProvider.DirectML => TryAddDirectML(options),
            ExecutionProvider.CoreML => TryAddCoreML(options),
            ExecutionProvider.Cpu => false, // CPU is always available as fallback; no GPU EP appended
            _ => throw new ArgumentOutOfRangeException(nameof(provider), provider, "Unknown execution provider")
        };
    }

    /// <summary>
    /// Tries to add the best available GPU provider. Returns true if a GPU EP was appended,
    /// false when none was available (CPU fallback is automatic).
    /// </summary>
    private static bool TryAddBestAvailableProvider(SessionOptions options)
    {
        // Try providers in order of preference
        if (TryAddCuda(options)) return true;
        if (TryAddDirectML(options)) return true;
        if (TryAddCoreML(options)) return true;
        // CPU fallback is automatic
        return false;
    }

    /// <summary>
    /// Resolves the active-provider list for an explicitly-requested provider from the actual append
    /// result (as reported by <see cref="Create(string, ExecutionProvider, Action{SessionOptions}?, out bool)"/>
    /// or <see cref="ConfigureExecutionProvider"/>). For CUDA, additionally verifies the provider DLL
    /// was loaded into the process so an appended-but-not-loaded CUDA EP is not reported as active.
    /// This replaces the former session-introspection heuristic, which reported false positives.
    /// </summary>
    public static IReadOnlyList<string> ResolveActiveProviders(ExecutionProvider provider, bool gpuEpAppended)
    {
        var providers = new List<string>();

        if (gpuEpAppended)
        {
            switch (provider)
            {
                case ExecutionProvider.Cuda when IsCudaProviderLoaded():
                    providers.Add("CUDAExecutionProvider");
                    break;
                case ExecutionProvider.Cuda:
                    // Appended but the CUDA provider DLL is not in the process — not actually active.
                    break;
                case ExecutionProvider.DirectML:
                    providers.Add("DmlExecutionProvider");
                    break;
                case ExecutionProvider.CoreML:
                    providers.Add("CoreMLExecutionProvider");
                    break;
            }
        }

        providers.Add("CPUExecutionProvider");
        return providers;
    }

    private static bool TryAddCuda(SessionOptions options)
    {
        try
        {
            options.AppendExecutionProvider_CUDA();
            Trace.TraceInformation("[OnnxSessionFactory] CUDA provider added successfully");
            return true;
        }
        catch (Exception ex)
        {
            Trace.TraceWarning($"[OnnxSessionFactory] Failed to add CUDA provider: {ex.Message}");
            return false;
        }
    }

    private static bool TryAddDirectML(SessionOptions options)
    {
        try
        {
            // DirectML requires these settings to work properly and avoid hangs
            // See: https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html
            options.EnableMemoryPattern = false;
            options.ExecutionMode = ExecutionMode.ORT_SEQUENTIAL;
            
            options.AppendExecutionProvider_DML();
            Trace.TraceInformation("[OnnxSessionFactory] DirectML provider added successfully");
            return true;
        }
        catch (Exception ex)
        {
            Trace.TraceWarning($"[OnnxSessionFactory] Failed to add DirectML provider: {ex.Message}");
            return false;
        }
    }

    private static bool TryAddCoreML(SessionOptions options)
    {
        try
        {
            options.AppendExecutionProvider_CoreML();
            return true;
        }
        catch (Exception ex)
        {
            Trace.TraceInformation($"[OnnxSessionFactory] CoreML provider not available: {ex.Message}");
            return false;
        }
    }

    /// <summary>
    /// Gets the list of available execution providers on the current system.
    /// </summary>
    public static IEnumerable<ExecutionProvider> GetAvailableProviders()
    {
        // Pre-check: if ONNX Runtime native library is not available, return CPU only
        var (available, _) = CheckOnnxRuntimeAvailability();
        if (!available)
        {
            yield return ExecutionProvider.Cpu;
            yield break;
        }

        // CPU is always available
        yield return ExecutionProvider.Cpu;

        // Check GPU providers
        var testOptions = new SessionOptions();

        if (TryAddCuda(testOptions))
            yield return ExecutionProvider.Cuda;

        testOptions = new SessionOptions();
        if (TryAddDirectML(testOptions))
            yield return ExecutionProvider.DirectML;

        testOptions = new SessionOptions();
        if (TryAddCoreML(testOptions))
            yield return ExecutionProvider.CoreML;
    }

}
