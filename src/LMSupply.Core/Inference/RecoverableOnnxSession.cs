using System.Diagnostics;
using LMSupply.Exceptions;
using Microsoft.ML.OnnxRuntime;

namespace LMSupply.Inference;

/// <summary>
/// An ONNX <see cref="InferenceSession"/> that recovers from a failing execution provider at run
/// time by moving to the next provider in the Auto fallback chain, for both failure shapes a GPU
/// provider produces: a run-time crash (<see cref="OnnxRuntimeException"/>, e.g. an unsupported
/// kernel) and a hang that only the caller-side bound in <see cref="CancellableInference"/> can
/// detect (<see cref="InferenceTimeoutException"/>).
/// </summary>
/// <remarks>
/// <para>
/// The session and the gate that serializes access to it travel together as one immutable handle
/// that is swapped atomically. This matters for the timeout case: the timed-out native run may
/// still be blocked inside <c>Run</c> holding the old gate, so the old session is never disposed
/// or reused — it is abandoned, a fresh session/gate pair is published, and a thread that later
/// acquires the old gate notices the handle changed and re-dispatches onto the current one.
/// </para>
/// <para>
/// Originally implemented inside <c>LMSupply.Embedder</c>'s inference engine and promoted here so
/// every ONNX-backed module (Reranker, Transcriber, Translator, Synthesizer, Segmenter, Ocr,
/// ImageGenerator, Detector, Captioner, …) shares one recovery path instead of each holding a
/// bare <c>InferenceSession</c> that fails outright on the first GPU crash or hang.
/// </para>
/// </remarks>
public sealed class RecoverableOnnxSession : IDisposable
{
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
    private readonly ExecutionProvider _requestedProvider;
    private readonly string _modelPath;
    private readonly Action<SessionOptions>? _configureOptions;
    private readonly string _logPrefix;
    private readonly List<ExecutionProvider> _blacklistedProviders = new();
    private readonly List<SessionHandle> _abandonedHandles = new();
    private readonly object _recoveryGate = new();
    private bool _disposed;

    /// <summary>
    /// Wraps an already-created session.
    /// </summary>
    /// <param name="session">The live session (may be null only in tests that never run inference).</param>
    /// <param name="activeProviders">Execution providers active on <paramref name="session"/>, primary first.</param>
    /// <param name="isGpuActive">Whether a GPU provider is active on <paramref name="session"/>.</param>
    /// <param name="requestedProvider">The provider the caller asked for; <see cref="ExecutionProvider.Cpu"/> disables recovery.</param>
    /// <param name="modelPath">Model file, needed to create a replacement session on the next provider.</param>
    /// <param name="configureOptions">Session options to reapply to a replacement session.</param>
    /// <param name="logPrefix">Prefix for the <see cref="Trace"/> diagnostics this class emits.</param>
    public RecoverableOnnxSession(
        InferenceSession session,
        IReadOnlyList<string> activeProviders,
        bool isGpuActive,
        ExecutionProvider requestedProvider,
        string modelPath,
        Action<SessionOptions>? configureOptions = null,
        string logPrefix = "[RecoverableOnnxSession]")
    {
        _handle = new SessionHandle(session, activeProviders, isGpuActive);
        _requestedProvider = requestedProvider;
        _modelPath = modelPath;
        _configureOptions = configureOptions;
        _logPrefix = logPrefix;
    }

    /// <summary>
    /// Wraps the session produced by <see cref="OnnxSessionFactory"/>'s <c>CreateWithInfoAsync</c>.
    /// </summary>
    public static RecoverableOnnxSession FromResult(
        SessionCreationResult result,
        string modelPath,
        Action<SessionOptions>? configureOptions = null,
        string logPrefix = "[RecoverableOnnxSession]")
        => new(result.Session, result.ActiveProviders, result.IsGpuActive, result.RequestedProvider, modelPath, configureOptions, logPrefix);

    /// <summary>Execution providers active on the current session, primary first.</summary>
    public IReadOnlyList<string> ActiveProviders => _handle.ActiveProviders;

    /// <summary>Whether a GPU provider is active on the current session.</summary>
    public bool IsGpuActive => _handle.IsGpuActive;

    /// <summary>The provider the caller asked for.</summary>
    public ExecutionProvider RequestedProvider => _requestedProvider;

    /// <summary>
    /// The current session, for diagnostics only. Do not run inference on it directly — use
    /// <see cref="Run{T}"/> so the gate, cancellation translation and provider fallback apply.
    /// </summary>
    public InferenceSession Session => _handle.Session;

    /// <summary>
    /// Runs <paramref name="work"/> against the current session under its gate. A cancelled token
    /// asks the native run to terminate cooperatively via <see cref="RunOptions.Terminate"/> and
    /// surfaces as <see cref="OperationCanceledException"/>. An <see cref="OnnxRuntimeException"/>
    /// on a non-CPU provider triggers one fallback to the next provider and exactly one retry.
    /// </summary>
    /// <param name="work">Receives the session and per-run options; must call <c>session.Run(..., runOptions)</c>.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    public T Run<T>(Func<InferenceSession, RunOptions, T> work, CancellationToken cancellationToken = default)
        => RunInternal(work, allowRetry: true, cancellationToken);

    /// <summary>
    /// Runs <paramref name="work"/> under <see cref="CancellableInference"/>'s bound as well as
    /// <see cref="Run{T}"/>'s crash fallback. If the bound elapses while a GPU provider is active,
    /// the session moves to the next provider (<see cref="TryRecoverAfterTimeout"/>) and the work
    /// is retried exactly once; a timeout on CPU, or a second timeout, surfaces unchanged.
    /// </summary>
    public async Task<T> RunWithRecoveryAsync<T>(
        Func<InferenceSession, RunOptions, T> work,
        TimeSpan? timeout = null,
        CancellationToken cancellationToken = default)
    {
        try
        {
            return await CancellableInference.RunAsync(() => Run(work, cancellationToken), cancellationToken, timeout);
        }
        catch (InferenceTimeoutException ex) when (TryRecoverAfterTimeout(ex))
        {
            return await CancellableInference.RunAsync(() => Run(work, cancellationToken), cancellationToken, timeout);
        }
    }

    private T RunInternal<T>(Func<InferenceSession, RunOptions, T> work, bool allowRetry, CancellationToken cancellationToken)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        // If the handle was replaced by a timeout recovery while this thread waited on the old
        // gate, re-dispatch onto the current handle rather than running on the abandoned session.
        SessionHandle handle;
        while (true)
        {
            handle = _handle;
            handle.Gate.Wait(cancellationToken);
            if (ReferenceEquals(handle, _handle))
                break;
            handle.Gate.Release();
        }

        try
        {
            // Terminate is checked between operators; it cannot preempt a hang inside a single
            // kernel (e.g. cold DirectML init) — that is what RunWithRecoveryAsync's bound is for.
            using var runOptions = new RunOptions();
            using var ctRegistration = cancellationToken.CanBeCanceled
                ? cancellationToken.Register(static state => ((RunOptions)state!).Terminate = true, runOptions)
                : default;
            try
            {
                return work(handle.Session, runOptions);
            }
            catch (OnnxRuntimeException) when (cancellationToken.IsCancellationRequested)
            {
                // RunOptions.Terminate surfaces as an OnnxRuntimeException; translate it so callers
                // see a cancellation, not a provider crash, and so fallback is not entered.
                throw new OperationCanceledException(cancellationToken);
            }
            catch (OnnxRuntimeException ex) when (allowRetry && _requestedProvider != ExecutionProvider.Cpu && TryFallback(ex))
            {
                // Retry exactly once on the replacement session (allowRetry: false prevents
                // runaway recursion if the next provider also fails on the same input).
                return RunInternal(work, allowRetry: false, cancellationToken);
            }
        }
        finally
        {
            handle.Gate.Release();
        }
    }

    /// <summary>
    /// After a run on the current provider threw, blacklists that provider and recreates the
    /// session on the next one in the Auto chain. The crashed run has returned, so the old session
    /// is disposed. Returns false when nothing can be done.
    /// </summary>
    public bool TryFallback(OnnxRuntimeException ex)
    {
        lock (_recoveryGate)
        {
            var current = _handle;
            var failedProvider = MapActiveToProvider(current.ActiveProviders);
            if (failedProvider is null)
            {
                Trace.TraceWarning($"{_logPrefix} Inference failed but active provider could not be identified; not attempting fallback.");
                return false;
            }

            var replacement = TryCreateReplacementHandle(
                failedProvider.Value,
                $"Inference failed on {failedProvider} ({ex.GetType().Name})",
                ex.Message);
            if (replacement is null)
                return false;

            current.Session?.Dispose();
            _handle = replacement;
            return true;
        }
    }

    /// <summary>
    /// After a run exceeded the caller-side bound, moves to the next provider without touching the
    /// old session (its native run may still be blocked holding the old gate — it is abandoned and
    /// reclaimed by <see cref="Dispose"/> only once its gate is free). Returns true when the caller
    /// should retry once; false when CPU was requested, CPU is already active, the provider was
    /// already blacklisted, or session recreation failed.
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
                return false;   // nothing below CPU to fall back to

            var replacement = TryCreateReplacementHandle(
                failedProvider.Value,
                $"Inference timed out on {failedProvider} after {ex.Timeout.TotalSeconds:F0}s (likely a cold GPU kernel initialization hang)",
                ex.Message);
            if (replacement is null)
                return false;

            _abandonedHandles.Add(stale);
            _handle = replacement;
            return true;
        }
    }

    // Caller must hold _recoveryGate.
    private SessionHandle? TryCreateReplacementHandle(ExecutionProvider failedProvider, string reason, string originalMessage)
    {
        if (_blacklistedProviders.Contains(failedProvider))
            return null;   // already tried recovery for this provider — give up

        _blacklistedProviders.Add(failedProvider);

        Trace.TraceWarning($"{_logPrefix} {reason}. Attempting fallback to next provider. Original message: {Truncate(originalMessage, 200)}");

        try
        {
            var result = OnnxSessionFactory.CreateWithInfoAsync(
                _modelPath,
                ExecutionProvider.Auto,
                _blacklistedProviders.ToArray(),
                _configureOptions).GetAwaiter().GetResult();

            var newProvider = MapActiveToProvider(result.ActiveProviders);
            if (newProvider is null || _blacklistedProviders.Contains(newProvider.Value))
            {
                Trace.TraceWarning($"{_logPrefix} Fallback produced no usable alternative provider (blacklist=[{string.Join(",", _blacklistedProviders)}]). Surfacing original exception.");
                result.Session.Dispose();
                return null;
            }

            Trace.TraceWarning($"{_logPrefix} Recovered: now running on {string.Join("+", result.ActiveProviders)}.");
            return new SessionHandle(result.Session, result.ActiveProviders, result.IsGpuActive);
        }
        catch (Exception fallbackEx)
        {
            Trace.TraceError($"{_logPrefix} Fallback session creation failed: {Truncate(fallbackEx.Message, 200)}");
            return null;
        }
    }

    private static ExecutionProvider? MapActiveToProvider(IReadOnlyList<string> activeProviders)
    {
        foreach (var p in activeProviders)
        {
            if (p.Contains("CUDA", StringComparison.OrdinalIgnoreCase))
                return ExecutionProvider.Cuda;
            if (p.Contains("DML", StringComparison.OrdinalIgnoreCase) || p.Contains("DirectML", StringComparison.OrdinalIgnoreCase))
                return ExecutionProvider.DirectML;
            if (p.Contains("CoreML", StringComparison.OrdinalIgnoreCase))
                return ExecutionProvider.CoreML;
        }
        if (activeProviders.Any(p => p.Contains("CPU", StringComparison.OrdinalIgnoreCase)))
            return ExecutionProvider.Cpu;
        return null;
    }

    private static string Truncate(string s, int max) => s.Length <= max ? s : s[..max] + "...";

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        var current = _handle;
        current.Session?.Dispose();
        current.Gate.Dispose();

        lock (_recoveryGate)
        {
            foreach (var abandoned in _abandonedHandles)
            {
                // Reclaim only a session whose timed-out run has actually returned; one still blocked
                // in native code must be leaked rather than disposed under it.
                if (abandoned.Gate.Wait(0))
                {
                    abandoned.Session?.Dispose();
                    abandoned.Gate.Release();
                    abandoned.Gate.Dispose();
                }
                else
                {
                    Trace.TraceWarning($"{_logPrefix} Leaking a session on {string.Join("+", abandoned.ActiveProviders)}: its timed-out native run has not returned, so it cannot be disposed safely.");
                }
            }
            _abandonedHandles.Clear();
        }
    }
}
