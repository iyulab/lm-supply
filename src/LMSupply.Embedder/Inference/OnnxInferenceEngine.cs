using System.Diagnostics;
using LMSupply.Download;
using LMSupply.Exceptions;
using LMSupply.Inference;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace LMSupply.Embedder.Inference;

/// <summary>
/// ONNX Runtime inference engine for embedding models.
/// </summary>
internal sealed class OnnxInferenceEngine : IDisposable
{
    /// <summary>
    /// An inference session paired with the gate that serializes access to it. The pair is
    /// immutable and swapped atomically on provider fallback, so a caller that acquired the gate
    /// of a handle that has since been replaced can detect it (reference mismatch) and re-dispatch
    /// onto the current one instead of running on a stale session.
    /// </summary>
    private sealed class SessionHandle
    {
        public SessionHandle(InferenceSession session, IReadOnlyList<string> activeProviders, bool isGpuActive)
        {
            Session = session;
            ActiveProviders = activeProviders;
            IsGpuActive = isGpuActive;
        }

        public InferenceSession Session { get; }
        public SemaphoreSlim Gate { get; } = new(1, 1);
        public IReadOnlyList<string> ActiveProviders { get; }
        public bool IsGpuActive { get; }
    }

    private volatile SessionHandle _handle;
    private readonly bool _hasTokenTypeIds;
    private readonly string _outputName;
    private bool _isGpuProvider;
    private readonly ExecutionProvider _requestedProvider;
    private readonly string _modelPath;
    private readonly List<ExecutionProvider> _blacklistedProviders = new();

    // Handles abandoned by TryRecoverAfterTimeout: their native run may still be blocked, so they
    // are neither disposed nor reused; Dispose() releases them only once their gate is free.
    private readonly List<SessionHandle> _abandonedHandles = new();
    private readonly object _recoveryGate = new();

    public int HiddenSize { get; }

    /// <summary>
    /// Gets whether GPU acceleration is being used for inference.
    /// </summary>
    public bool IsGpuActive => _handle.IsGpuActive;

    /// <summary>
    /// Gets the list of active execution providers.
    /// </summary>
    public IReadOnlyList<string> ActiveProviders => _handle.ActiveProviders;

    /// <summary>
    /// Gets the execution provider that was requested.
    /// </summary>
    public ExecutionProvider RequestedProvider => _requestedProvider;

    private OnnxInferenceEngine(
        InferenceSession session,
        int hiddenSize,
        bool hasTokenTypeIds,
        string outputName,
        bool isGpuProvider,
        bool isGpuActive,
        IReadOnlyList<string> activeProviders,
        ExecutionProvider requestedProvider,
        string modelPath)
    {
        _handle = new SessionHandle(session, activeProviders, isGpuActive);
        HiddenSize = hiddenSize;
        _hasTokenTypeIds = hasTokenTypeIds;
        _outputName = outputName;
        _isGpuProvider = isGpuProvider;
        _requestedProvider = requestedProvider;
        _modelPath = modelPath;
    }

    /// <summary>
    /// Creates an inference engine from an ONNX model file asynchronously.
    /// This method ensures runtime binaries are available before creating the session.
    /// </summary>
    /// <param name="modelPath">Path to the ONNX model file.</param>
    /// <param name="provider">The execution provider to use.</param>
    /// <param name="progress">Optional progress reporter for binary downloads.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>A configured inference engine.</returns>
    public static async Task<OnnxInferenceEngine> CreateAsync(
        string modelPath,
        ExecutionProvider provider,
        IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        if (!File.Exists(modelPath))
            throw new ModelNotFoundException("Model file not found", modelPath);

        var result = await OnnxSessionFactory.CreateWithInfoAsync(
            modelPath,
            provider,
            ConfigureOptions,
            progress,
            cancellationToken);

        return CreateFromSessionResult(result, modelPath);
    }

    /// <summary>
    /// Creates an inference engine from an ONNX model file.
    /// Note: This assumes runtime binaries are already available. For lazy loading, use CreateAsync.
    /// </summary>
    public static OnnxInferenceEngine Create(string modelPath, ExecutionProvider provider)
    {
        if (!File.Exists(modelPath))
            throw new ModelNotFoundException("Model file not found", modelPath);

        var session = OnnxSessionFactory.Create(modelPath, provider, ConfigureOptions, out var gpuEpAppended);
        var activeProviders = OnnxSessionFactory.ResolveActiveProviders(provider, gpuEpAppended);
        var isGpuActive = activeProviders.Any(p => p != "CPUExecutionProvider");

        return CreateFromSession(session, IsGpuProvider(provider), isGpuActive, activeProviders, provider, modelPath);
    }

    private static bool IsGpuProvider(ExecutionProvider provider)
    {
        return provider is ExecutionProvider.Cuda
            or ExecutionProvider.DirectML
            or ExecutionProvider.CoreML
            or ExecutionProvider.Auto; // Auto may select GPU, so treat as GPU for safety
    }

    private static void ConfigureOptions(SessionOptions options)
    {
        options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
        options.ExecutionMode = ExecutionMode.ORT_SEQUENTIAL;
        options.EnableCpuMemArena = true;
        options.EnableMemoryPattern = true;
        options.IntraOpNumThreads = Environment.ProcessorCount;
        options.InterOpNumThreads = 1;
    }

    private static OnnxInferenceEngine CreateFromSessionResult(SessionCreationResult result, string modelPath)
    {
        return CreateFromSession(
            result.Session,
            IsGpuProvider(result.RequestedProvider),
            result.IsGpuActive,
            result.ActiveProviders,
            result.RequestedProvider,
            modelPath);
    }

    private static OnnxInferenceEngine CreateFromSession(
        InferenceSession session,
        bool isGpuProvider,
        bool isGpuActive,
        IReadOnlyList<string> activeProviders,
        ExecutionProvider requestedProvider,
        string modelPath)
    {
        // Detect model configuration from metadata
        var inputNames = session.InputMetadata.Keys.ToHashSet();
        bool hasTokenTypeIds = inputNames.Contains("token_type_ids");

        // Get output name and hidden size
        var outputMeta = session.OutputMetadata.First();
        string outputName = outputMeta.Key;
        int hiddenSize = (int)outputMeta.Value.Dimensions[^1]; // Last dimension is hidden size

        return new OnnxInferenceEngine(
            session,
            hiddenSize,
            hasTokenTypeIds,
            outputName,
            isGpuProvider,
            isGpuActive,
            activeProviders,
            requestedProvider,
            modelPath);
    }

    /// <summary>
    /// Runs inference for a single sequence.
    /// </summary>
    /// <param name="inputIds">Token ids for the sequence.</param>
    /// <param name="attentionMask">Attention mask for the sequence.</param>
    /// <param name="cancellationToken">
    /// Cancellation token. When cancelled, the native ONNX run is asked to terminate
    /// cooperatively via <see cref="RunOptions.Terminate"/> (best-effort — honored only between
    /// operators, so an intra-kernel hang such as a cold DirectML init may not be preempted).
    /// </param>
    public float[] RunInference(long[] inputIds, long[] attentionMask, CancellationToken cancellationToken = default)
    {
        return RunInferenceInternal(inputIds, attentionMask, allowRetry: true, cancellationToken);
    }

    private float[] RunInferenceInternal(
        long[] inputIds, long[] attentionMask, bool allowRetry, CancellationToken cancellationToken)
    {
        int seqLength = inputIds.Length;

        // Create input tensors
        var inputIdsTensor = new DenseTensor<long>(inputIds, [1, seqLength]);
        var attentionMaskTensor = new DenseTensor<long>(attentionMask, [1, seqLength]);

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input_ids", inputIdsTensor),
            NamedOnnxValue.CreateFromTensor("attention_mask", attentionMaskTensor)
        };

        // Add token_type_ids if model expects it
        if (_hasTokenTypeIds)
        {
            var tokenTypeIds = new long[seqLength]; // All zeros
            var tokenTypeIdsTensor = new DenseTensor<long>(tokenTypeIds, [1, seqLength]);
            inputs.Add(NamedOnnxValue.CreateFromTensor("token_type_ids", tokenTypeIdsTensor));
        }

        // Serialize access to the InferenceSession (not thread-safe for concurrent Run calls).
        // The session and its gate travel together as one handle: if the handle was replaced by a
        // timeout recovery while this thread was waiting on the old gate, re-dispatch onto the
        // current handle rather than running on the abandoned session.
        SessionHandle handle;
        while (true)
        {
            handle = _handle;
            // Honor cancellation while waiting for the lock; Wait throws before the try, so the
            // finally/Release below is only reached when the lock was actually acquired.
            handle.Gate.Wait(cancellationToken);
            if (ReferenceEquals(handle, _handle))
                break;
            handle.Gate.Release();
        }

        try
        {
            // Per-run options so a cancelled token asks the native run to terminate cooperatively.
            // Terminate is checked between operators; it cannot preempt a hang inside a single
            // kernel (e.g. cold DirectML init). Control-return on such hangs is guaranteed by the
            // caller-side WaitAsync(ct) wrapper, not here.
            using var runOptions = new RunOptions();
            using var ctRegistration = cancellationToken.CanBeCanceled
                ? cancellationToken.Register(static state => ((RunOptions)state!).Terminate = true, runOptions)
                : default;
            try
            {
                using var results = handle.Session.Run(inputs, [_outputName], runOptions);
                var output = results[0].AsTensor<float>();

                // Output shape: [1, seqLength, hiddenSize]
                // Copy to flat array
                var outputArray = new float[seqLength * HiddenSize];
                int idx = 0;
                for (int seq = 0; seq < seqLength; seq++)
                {
                    for (int dim = 0; dim < HiddenSize; dim++)
                    {
                        outputArray[idx++] = output[0, seq, dim];
                    }
                }

                return outputArray;
            }
            catch (OnnxRuntimeException) when (cancellationToken.IsCancellationRequested)
            {
                // RunOptions.Terminate surfaces as an OnnxRuntimeException; translate it to a
                // cancellation so callers see OperationCanceledException, not a provider crash,
                // and so the fallback path below is not mistakenly entered.
                throw new OperationCanceledException(cancellationToken);
            }
            catch (OnnxRuntimeException ex) when (allowRetry && _requestedProvider != ExecutionProvider.Cpu && TryFallback(ex))
            {
                // After successful fallback, retry exactly once on the new session.
                // Pass allowRetry: false to avoid runaway recursion if the next provider also fails
                // on the same input (the outer call will see the propagated exception).
                return RunInferenceInternal(inputIds, attentionMask, allowRetry: false, cancellationToken);
            }
        }
        finally
        {
            handle.Gate.Release();
        }
    }

    /// <summary>
    /// Attempts to recreate the inference session on the next provider in the fallback chain
    /// after the current provider's session crashed at run time. Must be called by the thread
    /// holding the current handle's gate (the run has returned, so the old session is safe to
    /// dispose). Returns true on success; false if no remaining provider can run.
    /// </summary>
    private bool TryFallback(OnnxRuntimeException ex)
    {
        lock (_recoveryGate)
        {
            var current = _handle;
            var failedProvider = MapActiveToProvider(current.ActiveProviders);
            if (failedProvider is null)
            {
                Trace.TraceWarning(
                    "[OnnxInferenceEngine] Inference failed but active provider could not be identified; not attempting fallback.");
                return false;
            }

            var replacement = TryCreateReplacementHandle(
                failedProvider.Value,
                $"Inference failed on {failedProvider} ({ex.GetType().Name})",
                ex.Message);
            if (replacement is null)
                return false;

            // The crashed run has returned, so the old session can be released right away.
            current.Session.Dispose();
            _handle = replacement;
            return true;
        }
    }

    /// <summary>
    /// Attempts to move inference to the next provider in the fallback chain after a run on the
    /// current provider exceeded the caller-side timeout (see <see cref="CancellableInference"/>).
    /// Unlike <see cref="TryFallback"/>, the timed-out native run may still be blocked and still
    /// holds the old handle's gate, so the old session is abandoned rather than disposed and a
    /// fresh session/gate pair is published for subsequent calls. Returns true when a replacement
    /// provider is active and the caller should retry once; false when nothing can be done (CPU
    /// was requested, CPU was already the active provider, the provider was already blacklisted,
    /// or session recreation failed).
    /// </summary>
    public bool TryRecoverAfterTimeout(InferenceTimeoutException ex)
    {
        if (_requestedProvider == ExecutionProvider.Cpu)
            return false;

        lock (_recoveryGate)
        {
            var stale = _handle;
            var failedProvider = MapActiveToProvider(stale.ActiveProviders);
            if (failedProvider is null or ExecutionProvider.Cpu)
            {
                // Nothing below CPU to fall back to; surface the timeout as-is.
                return false;
            }

            var replacement = TryCreateReplacementHandle(
                failedProvider.Value,
                $"Inference timed out on {failedProvider} after {ex.Timeout.TotalSeconds:F0}s (likely a cold GPU kernel initialization hang)",
                ex.Message);
            if (replacement is null)
                return false;

            // The timed-out run may still be inside native code holding the old gate: never
            // dispose or reuse that session. Park it and let Dispose() reclaim it if it ever
            // returns. Waiters on the old gate re-dispatch onto the new handle (see
            // RunInferenceInternal).
            _abandonedHandles.Add(stale);
            _handle = replacement;
            return true;
        }
    }

    /// <summary>
    /// Shared fallback step: blacklists <paramref name="failedProvider"/>, then creates a session on
    /// the next provider in the Auto chain. Returns null when the provider was already blacklisted,
    /// when no different provider could be selected, or when session creation failed. Caller must
    /// hold <see cref="_recoveryGate"/>.
    /// </summary>
    private SessionHandle? TryCreateReplacementHandle(ExecutionProvider failedProvider, string reason, string originalMessage)
    {
        if (_blacklistedProviders.Contains(failedProvider))
        {
            // Already tried recovery for this provider — give up.
            return null;
        }

        _blacklistedProviders.Add(failedProvider);

        Trace.TraceWarning(
            $"[OnnxInferenceEngine] {reason}. " +
            $"Attempting fallback to next provider. Original message: {Truncate(originalMessage, 200)}");

        try
        {
            var result = OnnxSessionFactory.CreateWithInfoAsync(
                _modelPath,
                ExecutionProvider.Auto,
                _blacklistedProviders.ToArray(),
                ConfigureOptions).GetAwaiter().GetResult();

            // Verify a different provider was actually selected.
            var newProvider = MapActiveToProvider(result.ActiveProviders);
            if (newProvider is null || _blacklistedProviders.Contains(newProvider.Value))
            {
                Trace.TraceWarning(
                    $"[OnnxInferenceEngine] Fallback produced no usable alternative provider " +
                    $"(blacklist=[{string.Join(",", _blacklistedProviders)}]). Surfacing original exception.");
                result.Session.Dispose();
                return null;
            }

            _isGpuProvider = IsGpuProvider(newProvider.Value);

            Trace.TraceWarning(
                $"[OnnxInferenceEngine] Recovered: now running on {string.Join("+", result.ActiveProviders)}.");
            return new SessionHandle(result.Session, result.ActiveProviders, result.IsGpuActive);
        }
        catch (Exception fallbackEx)
        {
            Trace.TraceError(
                $"[OnnxInferenceEngine] Fallback session creation failed: {Truncate(fallbackEx.Message, 200)}");
            return null;
        }
    }

    private static ExecutionProvider? MapActiveToProvider(IReadOnlyList<string> activeProviders)
    {
        // The first non-CPU provider in the list is the primary.
        foreach (var p in activeProviders)
        {
            if (p.Contains("CUDA", StringComparison.OrdinalIgnoreCase))
                return ExecutionProvider.Cuda;
            if (p.Contains("DML", StringComparison.OrdinalIgnoreCase) ||
                p.Contains("DirectML", StringComparison.OrdinalIgnoreCase))
                return ExecutionProvider.DirectML;
            if (p.Contains("CoreML", StringComparison.OrdinalIgnoreCase))
                return ExecutionProvider.CoreML;
        }
        if (activeProviders.Any(p => p.Contains("CPU", StringComparison.OrdinalIgnoreCase)))
            return ExecutionProvider.Cpu;
        return null;
    }

    private static string Truncate(string s, int max) =>
        s.Length <= max ? s : s[..max] + "...";

    public void Dispose()
    {
        var current = _handle;
        current.Session?.Dispose();
        current.Gate.Dispose();

        lock (_recoveryGate)
        {
            foreach (var abandoned in _abandonedHandles)
            {
                // Only reclaim a session whose timed-out run has actually returned; a session still
                // blocked in native code must be leaked rather than disposed under it.
                if (abandoned.Gate.Wait(0))
                {
                    abandoned.Session?.Dispose();
                    abandoned.Gate.Release();
                    abandoned.Gate.Dispose();
                }
                else
                {
                    Trace.TraceWarning(
                        $"[OnnxInferenceEngine] Leaking a session on {string.Join("+", abandoned.ActiveProviders)}: " +
                        "its timed-out native run has not returned, so it cannot be disposed safely.");
                }
            }
            _abandonedHandles.Clear();
        }
    }
}
